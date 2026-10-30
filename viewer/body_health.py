"""
viewer/body_health.py

BodyHealth — background health check of every history entry's result solid
(cad/body_check.py).  A damaged result (inside-out, non-manifold,
self-intersecting) puts its entry in the orange warning state via
entry.health_warning; it never blocks or errors the op.

  - one check in flight at a time, in a kernel worker (PreviewJob thread
    waiting on the pool, so the UI stays live)
  - results are cached by shape fingerprint, so a replay that rebuilds the
    same geometry costs nothing
  - only the entry that FIRST breaks a body is checked for self-intersection:
    later ops on an already-flagged body (and pieces split off it) inherit
    the damage, and on bodies like that the check runs for minutes
  - the self-intersection pass has a timeout; a timed-out check just isn't
    reported (better silent than a false alarm)
"""

from __future__ import annotations

from viewer.preview_job import PreviewJob

# Ops whose shape_after isn't a new result of their own.
_SKIP_OPS = {"sketch", "offset_plane"}
SELF_INTERSECT_TIMEOUT = 120.0


class BodyHealth:
    def __init__(self, viewport):
        self._vp = viewport
        self._results: dict[tuple, tuple[str, bool]] = {}   # fp → (msg, self-intersecting)
        self._fp_by_id: dict[int, tuple] = {}               # id(shape) → (shape, fp)
        self._inflight_fp: tuple | None = None
        self._job = PreviewJob(self._on_done, "body-check")

    # ------------------------------------------------------------------

    def _fp(self, shape):
        from cad.body_check import fingerprint
        hit = self._fp_by_id.get(id(shape))
        if hit is not None and hit[0] is shape:
            return hit[1]
        fp = fingerprint(shape)
        self._fp_by_id[id(shape)] = (shape, fp)
        return fp

    def scan(self) -> None:
        """Apply known results to entries, then start the next unchecked one.
        Called on every history change — cheap when everything is cached."""
        entries = self._vp.history.entries
        live_ids = set()
        flagged_bodies: set[str] = set()     # bodies already self-intersecting
        todo = None
        for e in entries:
            s = e.shape_after
            if e.operation in _SKIP_OPS or e.error or s is None:
                e.health_warning = ""
                continue
            live_ids.add(id(s))
            # Split-off pieces of a damaged body carry its damage too.
            if e.params.get("split_from") in flagged_bodies:
                flagged_bodies.add(e.body_id)
            inherited = e.body_id in flagged_bodies
            fp = self._fp(s)
            known = self._results.get(fp) if fp is not None else None
            if known is not None:
                msg, si = known
                e.health_warning = "" if inherited else msg
                if si:
                    flagged_bodies.add(e.body_id)
            else:
                e.health_warning = ""
                if todo is None and fp is not None and fp != self._inflight_fp:
                    todo = (s, fp, inherited)
        # Forget shapes no entry holds any more (keeps ids from being reused).
        self._fp_by_id = {k: v for k, v in self._fp_by_id.items() if k in live_ids}
        if todo is not None and not self._job.busy:
            shape, fp, skip_si = todo
            self._inflight_fp = fp
            self._job.submit(lambda: (fp, _check(shape, skip_si)))

    def _on_done(self, result, error) -> None:
        self._inflight_fp = None
        if error is None and result is not None:
            fp, found = result
            self._results[fp] = found
        self._vp.history_changed.emit()      # refresh rows → scan() → next

    def cancel(self) -> None:
        self._job.cancel()
        self._inflight_fp = None


def _check(shape, skip_si: bool) -> tuple[str, bool]:
    """(warning text, self-intersecting) — runs on the PreviewJob thread."""
    from cad import body_check
    from cad.kernel_pool import pool, enabled, KernelTimeout
    problems = body_check.quick_problems(shape)
    si = False
    if not skip_si:
        try:
            n = (pool().call(body_check.self_intersections, shape,
                             timeout=SELF_INTERSECT_TIMEOUT)
                 if enabled() else body_check.self_intersections(shape))
        except KernelTimeout:
            n = 0
            print(f"[Body check] self-intersection check timed out "
                  f"({SELF_INTERSECT_TIMEOUT:.0f}s) — skipped")
        if n:
            si = True
            problems.append(f"self-intersecting ({n} "
                            f"intersection{'s' * (n != 1)})")
    return body_check.describe(problems), si
