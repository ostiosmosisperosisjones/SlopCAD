"""
cad/operations/fillet_proc.py

Process-isolated, killable fillet execution.

OCCT's BRepFilletAPI_MakeFillet.Build() can spin forever on certain
edge/radius combinations (e.g. a partial circular edge between a
surface-of-revolution and a plane at a specific radius). The spin is in C++
holding the GIL, so neither a Python-level timeout nor running it on a worker
thread can interrupt it — the whole app freezes.

The only robust defense is to run the kernel call in a *child process* that can
be killed. This module serializes the shape to BREP, ships the request to a
worker process, and enforces a hard wall-clock timeout; on overrun the worker is
terminated and FilletTimeout is raised so callers surface a clean message
instead of hanging.

Edges are identified across the process boundary by geometry (midpoint), not by
index — BREP round-trip does not preserve topology order.
"""

from __future__ import annotations
import os
import tempfile
import multiprocessing as _mp


# Default hard cap for a single fillet build. Generous enough for legitimately
# heavy fillets (a long BSpline edge can take tens of seconds) but bounded so a
# true kernel hang can't lock the app indefinitely.
DEFAULT_TIMEOUT_S = 25.0


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


def _write_status(status_path: str, msg: str):
    try:
        with open(status_path, "w") as f:
            f.write(msg)
    except OSError:
        pass


def _worker(brep_path: str, face_indices, edge_mids, edge_radii, radius,
            out_path: str, status_path: str, validate_mesh: bool = False):
    """Child-process entry point. Reads the shape, re-matches edges by midpoint,
    runs the (heal-retrying) fillet, writes the result BREP. Writes a short
    status string to status_path the parent reads back.

    Status goes through a file rather than a multiprocessing.Array because the
    latter allocates a POSIX semaphore per call; preview fires one process per
    value change, and those semaphores leaked and eventually exhausted the
    system (the "leaked semaphore objects" warning + lockup).

    edge_radii (optional, aligned with edge_mids) carries a per-edge radius
    override (float, or [r1,r2] taper, or None). Matched to the rebuilt edges by
    midpoint so it survives BREP round-trip (which doesn't preserve edge order).
    When all overrides are None, the scalar `radius` is used for everything.
    """
    try:
        from OCP.BRepTools import BRepTools
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Shape
        from build123d import Compound
        from cad.operations.fillet import fillet_edges

        builder = BRep_Builder()
        occ = TopoDS_Shape()
        BRepTools.Read_s(occ, brep_path, builder)
        shape = Compound(occ)

        # Map midpoint → override radius (None entries fall back to scalar).
        mid_to_r = {m: r for m, r in zip(edge_mids, edge_radii or [])}
        wanted = set(edge_mids)
        edge_occs, occ_radii = [], []
        for e in shape.edges():
            m = _edge_midpoint(e.wrapped)
            if m in wanted:
                edge_occs.append(e.wrapped)
                r = mid_to_r.get(m)
                occ_radii.append(radius if r is None else
                                 (tuple(r) if isinstance(r, (list, tuple)) else r))

        has_override = any(m in mid_to_r and mid_to_r[m] is not None
                           for m in (edge_mids or []))
        # Per-edge radii only apply to picked edges (no face_indices in that
        # mode); pass the aligned list, else the plain scalar.
        radius_arg = occ_radii if (has_override and not face_indices) else radius

        result = fillet_edges(shape, list(face_indices), edge_occs, radius_arg)

        # Validate that the result can actually be tessellated. OCCT happily
        # produces fillet solids that the mesher then can't triangulate (the
        # "tessellator gave N faces but shape.faces() gave N+1" failure), which
        # render as an empty/broken body. Such a radius is useless, so for the
        # fit-largest search we reject it the same as a build failure — this is
        # what keeps fit-largest from returning a radius that builds but can't
        # be shown or used.
        if validate_mesh:
            from viewer.mesh import Mesh
            Mesh(result)   # raises on face-count mismatch / tessellation failure

        result_occ = result.wrapped if hasattr(result, "wrapped") else result
        BRepTools.Write_s(result_occ, out_path)
        _write_status(status_path, "ok")
    except Exception as ex:  # propagate a short reason, not a traceback object
        _write_status(status_path, f"err:{ex}"[:255])


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
                          attempt_timeout_s: float = 8.0,
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
        """Build r in `repeats` independent processes; return a result only if
        ALL succeed (rejects unstable borderline radii)."""
        result = None
        for _ in range(repeats):
            result = _build_once(r)   # raises on any failure → caller treats r as bad
        return result

    hi = ceiling if ceiling and ceiling > 0 else _max_search_ceiling(edge_occs)
    lo = max(hi * 0.02, 1e-3)

    # If the ceiling reliably builds, that's the answer.
    try:
        return _reliable(hi), round(hi, 4)
    except (FilletTimeout, RuntimeError):
        pass

    # Known-good lower bound (must also be reliable); else nothing builds.
    best = None
    try:
        best = (_reliable(lo), lo)
    except (FilletTimeout, RuntimeError):
        raise FilletTimeout(hi, attempt_timeout_s)

    # Binary search in (lo, hi): largest r that *reliably* builds.
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

    return best[0], round(best[1], 4)


# Back-compat: the old name routed through here. Preview no longer auto-fits, so
# this is only reached via the explicit "Fit largest" action.
def fillet_edges_autofit_isolated(shape, face_indices, edge_occs, radius: float,
                                  timeout_s: float = DEFAULT_TIMEOUT_S,
                                  attempt_timeout_s: float = 8.0,
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
    """Run fillet_edges() in a killable child process with a hard timeout.

    Same signature/return as cad.operations.fillet.fillet_edges (a build123d
    Compound), but a kernel hang is bounded: on timeout the worker is killed and
    FilletTimeout is raised.

    edge_radii (optional): per-edge radius overrides aligned with edge_occs —
    each a float, an [r1, r2] taper, or None to use the scalar `radius`.

    validate_mesh: when True, the worker also tessellates the result and fails
    if it can't be meshed — used by the fit-largest search so it never reports a
    radius that builds but can't render.
    """
    from OCP.BRepTools import BRepTools
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shape
    from build123d import Compound

    src_occ = shape.wrapped if hasattr(shape, "wrapped") else shape
    edge_mids = [_edge_midpoint(e) for e in (edge_occs or [])]
    edge_radii = list(edge_radii) if edge_radii else [None] * len(edge_mids)

    tmpdir = tempfile.mkdtemp(prefix="cadapp_fillet_")
    brep_path = os.path.join(tmpdir, "src.brep")
    out_path = os.path.join(tmpdir, "out.brep")
    status_path = os.path.join(tmpdir, "status.txt")
    BRepTools.Write_s(src_occ, brep_path)

    # Status flows through a file, not a multiprocessing.Array — the latter
    # allocates a POSIX semaphore per call which leaked under rapid preview
    # firing and eventually locked the app.
    ctx = _mp.get_context("spawn")
    proc = ctx.Process(
        target=_worker,
        args=(brep_path, list(face_indices), edge_mids, edge_radii, radius,
              out_path, status_path, validate_mesh),
        daemon=True)
    proc.start()
    # Poll instead of one blocking join so the caller can cancel mid-build (and
    # so a hung worker is still bounded by timeout_s). Cheap: ~20 wakeups/sec.
    import time
    deadline = time.monotonic() + timeout_s
    canceled = False
    while proc.is_alive():
        if should_cancel is not None and should_cancel():
            canceled = True
            break
        if time.monotonic() >= deadline:
            break
        proc.join(0.05)

    try:
        if proc.is_alive():
            proc.terminate()
            proc.join(2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(1.0)
            if canceled:
                raise FilletCanceled("fillet canceled")
            raise FilletTimeout(radius, timeout_s)

        try:
            with open(status_path) as f:
                st = f.read().strip()
        except OSError:
            st = ""
        if st.startswith("err:"):
            raise RuntimeError(st[4:])
        if st != "ok" or not os.path.exists(out_path):
            raise FilletProcessError(
                f"fillet worker exited (code {proc.exitcode}) without a result")

        builder = BRep_Builder()
        result_occ = TopoDS_Shape()
        BRepTools.Read_s(result_occ, out_path, builder)
        return Compound(result_occ)
    finally:
        # Reap the process so its OS resources are released (avoids zombie/leak
        # accumulation across many preview calls).
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)
            proc.close()
        except Exception:
            pass
        for p in (brep_path, out_path, status_path):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
