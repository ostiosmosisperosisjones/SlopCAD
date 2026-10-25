"""
cad/operations/fillet_proc.py

Process-isolated, killable fillet execution.

OCCT's BRepFilletAPI_MakeFillet.Build() can spin forever on certain
edge/radius combinations (e.g. a partial circular edge between a
surface-of-revolution and a plane at a specific radius). The spin is in C++
holding the GIL, so neither a Python-level timeout nor running it on a worker
thread can interrupt it — the whole app freezes.

The only robust defense is to run the kernel call in a *child process* that can
be killed.  Builds run in the warm workers of cad.kernel_pool with a hard
wall-clock timeout; on overrun the worker is killed and FilletTimeout is
raised so callers surface a clean message instead of hanging.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor


# Default hard cap for a single fillet build. Generous enough for legitimately
# heavy fillets (a long BSpline edge can take tens of seconds) but bounded so a
# true kernel hang can't lock the app indefinitely.
DEFAULT_TIMEOUT_S = 25.0

# Per-probe cap for the fit-largest search. On filletbug.vc a *buildable*
# radius takes ~6s; tighter caps rejected every candidate.
ATTEMPT_TIMEOUT_S = 15.0


class FilletTimeout(RuntimeError):
    """Raised when a fillet build exceeds its wall-clock budget (likely an OCCT
    kernel hang). Carries the radius so callers can suggest trying another."""

    def __init__(self, radius: float, timeout_s: float):
        self.radius = radius
        self.timeout_s = timeout_s
        super().__init__(
            f"Fillet timed out after {timeout_s:.0f}s at radius {radius:g} — "
            f"this edge can't be filleted at that radius (try a smaller one)")


class FilletProcessError(RuntimeError):
    """The worker process died or returned no result for a non-timeout reason."""


class FilletCanceled(RuntimeError):
    """Raised when the caller's should_cancel() predicate asked to abort an
    in-flight build (the child process is killed)."""


def _edge_midpoint(occ_edge):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    a = BRepAdaptor_Curve(occ_edge)
    p = a.Value((a.FirstParameter() + a.LastParameter()) * 0.5)
    return (round(p.X(), 4), round(p.Y(), 4), round(p.Z(), 4))


def _own_edges(shape, edge_occs):
    """Edges not found in `shape` (a caller passed a copy, not a sub-shape)
    are swapped for the body's edge with the same midpoint."""
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap
    m = IMap()
    TopExp.MapShapes_s(getattr(shape, "wrapped", shape), TopAbs_EDGE, m)
    if all(m.FindIndex(e) for e in edge_occs):
        return edge_occs
    by_mid = {_edge_midpoint(m.FindKey(i)): m.FindKey(i) for i in range(1, m.Size() + 1)}
    return [e if m.FindIndex(e) else by_mid.get(_edge_midpoint(e), e) for e in edge_occs]


def _fillet_task(shape, face_indices, edge_occs, radius_arg, validate_mesh):
    """Worker-side build.  edge_occs arrive as sub-shapes of the worker's copy
    of `shape` (see kernel_pool), aligned with any per-edge radii."""
    from cad.operations.fillet import fillet_edges
    edge_occs = _own_edges(shape, [getattr(e, "wrapped", e) for e in edge_occs])
    result = fillet_edges(shape, list(face_indices), edge_occs, radius_arg)
    # OCCT happily produces fillet solids the mesher then can't triangulate
    # (tessellator face count != shape.faces()), which render broken.  For the
    # fit-largest search such a radius is as useless as a failed build.
    if validate_mesh:
        from viewer.mesh import Mesh
        Mesh(result)
    return result


def _max_search_ceiling(edge_occs):
    """A geometry-derived upper bound for the fit search: a fillet can't exceed
    roughly half the shortest selected edge's length (and we cap generously).
    Independent of any current radius so repeated 'Fit largest' is idempotent."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_AbscissaPoint
    lengths = []
    for e in (edge_occs or []):
        try:
            a = BRepAdaptor_Curve(e)
            lengths.append(GCPnts_AbscissaPoint.Length_s(a))
        except Exception:
            pass
    if not lengths:
        return 50.0
    return max(0.5 * min(lengths), 0.5)


def fillet_edges_find_max(shape, face_indices, edge_occs,
                          ceiling: float | None = None,
                          attempt_timeout_s: float = ATTEMPT_TIMEOUT_S,
                          tol_fraction: float = 0.05,
                          max_iters: int = 8,
                          repeats: int = 3,
                          edge_radii=None,
                          progress=None, should_cancel=None):
    """Find (and build at) the largest *reliably* buildable fillet radius.

    OCCT's fillet near its limit on a given edge is unstable: the SAME radius can
    build in one process and fail/hang in another (floating-point / kernel-state
    dependent). So a radius is only trusted as buildable when it builds in
    `repeats` independent child processes — this rejects borderline radii that
    would validate once and then fail on the separate commit run.

    Deterministic and idempotent: the range is anchored to a geometry-derived
    ceiling, not any current/typed radius. Each probe is killable, so a hanging
    radius is killed and counts as 'not buildable'.

    Returns (shape, applied_radius). Raises FilletTimeout if nothing in the
    range builds reliably.
    """
    def _scaled(r):
        if not edge_radii:
            return None
        out = []
        base = ceiling or r
        f = (r / base) if base else 1.0
        for er in edge_radii:
            if er is None:
                out.append(None)
            elif isinstance(er, (list, tuple)):
                out.append([float(er[0]) * f, float(er[1]) * f])
            else:
                out.append(float(er) * f)
        return out

    # Rough denominator for progress: the ceiling probe, the low bound, then up
    # to max_iters bisections, each repeated `repeats` times.
    total = (2 + max_iters) * repeats
    done = [0]

    def _build_once(r):
        # validate_mesh: a fit-largest candidate must build AND tessellate, or
        # it's no good (would commit to an unrenderable body).
        if should_cancel is not None and should_cancel():
            raise FilletCanceled("fit canceled")
        done[0] += 1
        if progress is not None:
            # Pass the radius (in mm) as value_mm so the UI can format it in the
            # user's chosen unit; text is the label prefix.
            progress("trying radius", done[0], total, r)
        return fillet_edges_isolated(shape, face_indices, edge_occs, r,
                                     timeout_s=attempt_timeout_s,
                                     edge_radii=_scaled(r), validate_mesh=True,
                                     should_cancel=should_cancel)

    def _reliable(r):
        """Build r in `repeats` independent worker processes at once; return a
        result only if ALL succeed (rejects unstable borderline radii)."""
        with ThreadPoolExecutor(max_workers=repeats) as ex:
            futs = [ex.submit(_build_once, r) for _ in range(repeats)]
            results = [f.result() for f in futs]   # re-raises any failure
        return results[0]

    hi = ceiling if ceiling and ceiling > 0 else _max_search_ceiling(edge_occs)
    lo = 0.0

    # If the ceiling reliably builds, that's the answer.
    try:
        return _reliable(hi), round(hi, 4)
    except (FilletTimeout, RuntimeError):
        pass

    # Binary search in (0, hi): largest r that *reliably* builds. No up-front
    # lower-bound probe — tiny radii on long BSpline edges are the *slowest*
    # builds (0.14mm took 27s on filletbug.vc vs ~9s at 2mm), so they're only
    # reached if nothing larger builds.
    best = None
    for _ in range(max_iters):
        if (hi - lo) <= tol_fraction * hi:
            break
        mid = round(0.5 * (lo + hi), 4)
        if mid <= lo:
            break
        try:
            res = _reliable(mid)
            best = (res, mid)
            lo = mid          # mid reliably builds → push lower bound up
        except (FilletTimeout, RuntimeError):
            hi = mid          # mid failed at least once → pull upper bound down

    if best is None:
        raise FilletTimeout(hi, attempt_timeout_s)
    return best[0], round(best[1], 4)


# Back-compat: the old name routed through here. Preview no longer auto-fits, so
# this is only reached via the explicit "Fit largest" action.
def fillet_edges_autofit_isolated(shape, face_indices, edge_occs, radius: float,
                                  timeout_s: float = DEFAULT_TIMEOUT_S,
                                  attempt_timeout_s: float = ATTEMPT_TIMEOUT_S,
                                  edge_radii=None, **_legacy):
    """Find the largest buildable radius, capped at *radius* (the requested
    value — we never fit larger than asked). Thin wrapper over
    fillet_edges_find_max."""
    return fillet_edges_find_max(
        shape, face_indices, edge_occs, ceiling=radius,
        attempt_timeout_s=attempt_timeout_s, edge_radii=edge_radii)


def fillet_edges_isolated(shape, face_indices, edge_occs, radius: float,
                          timeout_s: float = DEFAULT_TIMEOUT_S,
                          edge_radii=None, validate_mesh: bool = False,
                          should_cancel=None):
    """Run fillet_edges() in a killable kernel worker with a hard timeout.

    Same signature/return as cad.operations.fillet.fillet_edges (a build123d
    Compound), but a kernel hang is bounded: on timeout the worker is killed and
    FilletTimeout is raised.

    edge_radii (optional): per-edge radius overrides aligned with edge_occs —
    each a float, an [r1, r2] taper, or None to use the scalar `radius`.

    validate_mesh: when True, the worker also tessellates the result and fails
    if it can't be meshed — used by the fit-largest search so it never reports a
    radius that builds but can't render.
    """
    from build123d import Compound
    from cad import kernel_pool as kp

    edge_occs = list(edge_occs or [])
    radius_arg = radius
    # Per-edge radii only apply to picked edges (no face_indices in that mode).
    if edge_radii and any(r is not None for r in edge_radii) and not face_indices:
        radius_arg = [radius if r is None else
                      (tuple(r) if isinstance(r, (list, tuple)) else r)
                      for r in edge_radii]
    args = (shape, list(face_indices), edge_occs, radius_arg, validate_mesh)
    if not kp.enabled():
        return Compound(_fillet_task(*args).wrapped)
    try:
        r = kp.pool().call(_fillet_task, *args, timeout=timeout_s,
                           should_cancel=should_cancel)
    except kp.KernelTimeout:
        raise FilletTimeout(radius, timeout_s)
    except kp.KernelCanceled:
        raise FilletCanceled("fillet canceled")
    except kp.KernelWorkerDied as ex:
        raise FilletProcessError(str(ex))
    return r if isinstance(r, Compound) else Compound(r.wrapped)
