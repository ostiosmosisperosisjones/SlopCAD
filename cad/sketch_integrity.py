"""Sketch integrity check (PROVENANCE_AUDIT_PLAN.md step 2).

check_sketch(se) reads the sketch's final geometry, independent of the
solver, and lists what no longer holds:

  constraint   a geometric constraint / dimension is not satisfied
  degenerate   a zero-length line or zero-radius arc
  bad-index    a constraint or snap names an entity that isn't there, or a
               degenerate one
  snap         a MIDPOINT / NEAREST snap point left its target
  include-mid  a point linked at t = 0.5 of a straight include is off its middle
  coincidence  (with a snapshot) ends that met before an edit no longer meet

Only constraint kinds it can read exactly are checked; anything else is
skipped, not guessed.  Used on replay (SketchOp sets entry.warning) and by
tests and the stress walker.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

_DEGEN = 1e-3          # = sketch._ATTACH_TOL
_DIR_TOL = 1e-6        # on unit vectors, as in SketchEntry._constraint_holds
_POS_TOL = 1e-6        # mm, scaled by max(1, size)
_MEET_TOL = 1e-6


class Problem(NamedTuple):
    kind:    str
    msg:     str
    indices: tuple = ()


def _scale(*vals) -> float:
    return max([1.0] + [abs(float(v)) for v in vals])


def _line(ent):
    from cad.sketch import line_ends
    try:
        e = line_ends(ent)
    except Exception:
        return None
    return None if e is None else (np.asarray(e[0], float), np.asarray(e[1], float))


def _circle(ent):
    from cad.sketch import ArcEntity, circle_of
    if isinstance(ent, ArcEntity):
        return np.asarray(ent.center, float), float(ent.radius)
    try:
        c = circle_of(ent)
    except Exception:
        return None
    return None if c is None else (np.asarray(c[0], float), float(c[1]))


def _name(se, i) -> str:
    from cad.sketch import ArcEntity, LineEntity, ReferenceEntity
    e = se.entities[i]
    kind = ("line" if isinstance(e, LineEntity) else "arc" if isinstance(e, ArcEntity)
            else "include" if isinstance(e, ReferenceEntity) else type(e).__name__.lower())
    return f"{kind} {i}"


def _degenerate(ent) -> bool:
    from cad.sketch import ArcEntity, LineEntity
    if isinstance(ent, LineEntity):
        return float(np.linalg.norm(ent.p1 - ent.p0)) < _DEGEN
    if isinstance(ent, ArcEntity):
        return ent.radius < _DEGEN
    return False


def _check_constraint(se, con):
    """None when it holds or isn't a kind read here, else a message."""
    idx = con.indices
    ents = [se.entities[i] for i in idx]
    t = con.type
    if t in ("horizontal", "vertical") and len(idx) == 1:
        ln = _line(ents[0])
        if ln is None:
            return None
        d = ln[1] - ln[0]
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return None
        u = d / n
        off = abs(u[1]) if t == "horizontal" else abs(u[0])
        return None if off < _DIR_TOL else f"{t} on {_name(se, idx[0])} no longer holds"
    if t in ("parallel", "perpendicular") and len(idx) == 2:
        ls = [_line(e) for e in ents]
        if None in ls:
            return None
        u = []
        for a, b in ls:
            n = float(np.linalg.norm(b - a))
            if n < 1e-9:
                return None
            u.append((b - a) / n)
        off = (abs(u[0][0] * u[1][1] - u[0][1] * u[1][0]) if t == "parallel"
               else abs(float(u[0] @ u[1])))
        return None if off < _DIR_TOL else \
            f"{t} between {_name(se, idx[0])} and {_name(se, idx[1])} no longer holds"
    if t == "distance" and len(idx) == 1 and con.points is None:
        ln = _line(ents[0])
        if ln is None:
            return None
        L = float(np.linalg.norm(ln[1] - ln[0]))
        return None if abs(L - con.value) < _POS_TOL * _scale(con.value) else \
            f"length of {_name(se, idx[0])} is {L:.6g}, dimension says {con.value:.6g}"
    if t == "diameter" and len(idx) == 1:
        c = _circle(ents[0])
        if c is None:
            return None
        return None if abs(2 * c[1] - con.value) < _POS_TOL * _scale(con.value) else \
            f"diameter of {_name(se, idx[0])} is {2 * c[1]:.6g}, dimension says {con.value:.6g}"
    if t == "concentric" and len(idx) == 2:
        cs = [_circle(e) for e in ents]
        if None in cs:
            return None
        d = float(np.linalg.norm(cs[0][0] - cs[1][0]))
        return None if d < _POS_TOL * _scale(cs[0][1], cs[1][1]) else \
            f"concentric {_name(se, idx[0])}/{_name(se, idx[1])} centres {d:.3g} mm apart"
    if t == "equal" and len(idx) == 2:
        cs = [_circle(e) for e in ents]
        if None not in cs:
            a, b = cs[0][1], cs[1][1]
        else:
            ls = [_line(e) for e in ents]
            if None in ls:
                return None
            a, b = (float(np.linalg.norm(q - p)) for p, q in ls)
        return None if abs(a - b) < _POS_TOL * _scale(a, b) else \
            f"equal {_name(se, idx[0])}/{_name(se, idx[1])}: {a:.6g} vs {b:.6g}"
    return None


def _on_segment(p, a, b) -> float:
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-24 else min(1.0, max(0.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(a + t * ab - p))


def _check_snaps(se, out):
    from cad.sketch import LineEntity
    from cad.sketch_tools.snap import SnapType
    n = len(se.entities)
    for i, ent in enumerate(se.entities):
        if not isinstance(ent, LineEntity):
            continue
        for which in ("p0", "p1"):
            snap = getattr(ent, f"{which}_snap", None)
            if snap is None or snap[1] not in (SnapType.MIDPOINT, SnapType.NEAREST):
                continue
            j = snap[0]
            if not (0 <= j < n):
                out.append(Problem("bad-index", f"{_name(se, i)} {which} snaps to "
                                   f"entity {j}, which doesn't exist", (i,)))
                continue
            p = np.asarray(getattr(ent, which), float)
            tgt = se.entities[j]
            # Only what the solver holds (_add_snap_coincidences): snaps on
            # drawn lines.  Arc snaps are pinned to their stored spot, a
            # curved include's midpoint is on-curve only (known gap).
            if not isinstance(tgt, LineEntity):
                continue
            ln = (np.asarray(tgt.p0, float), np.asarray(tgt.p1, float))
            if j == i or min(float(np.linalg.norm(p - q)) for q in ln) < _DEGEN:
                continue        # stale: the point is an end of its own target
            size = float(np.linalg.norm(ln[1] - ln[0]))
            d = (float(np.linalg.norm(p - (ln[0] + ln[1]) / 2))
                 if snap[1] == SnapType.MIDPOINT else _on_segment(p, *ln))
            if d > _POS_TOL * _scale(size):
                out.append(Problem("snap", f"{_name(se, i)} {which}: "
                                   f"{snap[1].name.lower()} snap to {_name(se, j)} "
                                   f"is {d:.3g} mm off", (i, j)))


def _check_include_mids(se, out):
    from cad.sketch import ReferenceEntity
    links = getattr(se, "links", None) or {}
    if not links:
        return
    by_uid = {e.uid: (k, e) for k, e in enumerate(se.entities)}
    for (uid, end), (ref_uid, kind, data, n_pts) in links.items():
        if kind != "s" or n_pts != 2 or abs(data[1] - 0.5) > 1e-12:
            continue
        if uid not in by_uid or ref_uid not in by_uid:
            continue
        i, ent = by_uid[uid]
        ri, ref = by_uid[ref_uid]
        if not isinstance(ref, ReferenceEntity) or len(ref.points) != 2:
            continue
        a, b = (np.asarray(p, float) for p in ref.points)
        p = np.asarray(getattr(ent, end), float)
        d = float(np.linalg.norm(p - (a + b) / 2))
        if d > _POS_TOL * _scale(float(np.linalg.norm(b - a))):
            out.append(Problem("include-mid", f"{_name(se, i)} {end} should be at the "
                               f"middle of {_name(se, ri)}; {d:.3g} mm off", (i, ri)))


def integrity_snapshot(se) -> dict:
    """Which drawn ends meet, keyed by (uid, end), for check_sketch(before=)."""
    from cad.sketch import ArcEntity, LineEntity
    pts = [((e.uid, w), np.asarray(getattr(e, w), float))
           for e in se.entities if isinstance(e, (LineEntity, ArcEntity))
           for w in ("p0", "p1")]
    pairs = set()
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            if pts[a][0][0] != pts[b][0][0] and \
                    float(np.linalg.norm(pts[a][1] - pts[b][1])) < _MEET_TOL:
                pairs.add((pts[a][0], pts[b][0]))
    return {"meet": pairs}


def _check_coincidence(se, before, out):
    from cad.sketch import ArcEntity, LineEntity
    now = {(e.uid, w): np.asarray(getattr(e, w), float)
           for e in se.entities if isinstance(e, (LineEntity, ArcEntity))
           for w in ("p0", "p1")}
    idx = {e.uid: k for k, e in enumerate(se.entities)}
    for ka, kb in sorted(before.get("meet", ())):
        if ka not in now or kb not in now:
            continue                       # an end the edit removed or split away
        d = float(np.linalg.norm(now[ka] - now[kb]))
        if d > _MEET_TOL * 10:
            i, j = idx[ka[0]], idx[kb[0]]
            out.append(Problem("coincidence", f"{_name(se, i)} {ka[1]} and "
                               f"{_name(se, j)} {kb[1]} met before, now {d:.3g} mm apart",
                               (i, j)))


def check_sketch(se, before: dict | None = None) -> list[Problem]:
    out: list[Problem] = []
    n = len(se.entities)
    degen = set()
    for i, e in enumerate(se.entities):
        if _degenerate(e):
            degen.add(i)
            out.append(Problem("degenerate", f"{_name(se, i)} has collapsed to nothing", (i,)))
    for con in se.constraints:
        # sharp_distance: indices[1:3] are optional neighbours, -1 = none.
        named = con.indices[:1] if con.type == "sharp_distance" else con.indices
        bad = [i for i in named if not (0 <= i < n)]
        if bad:
            out.append(Problem("bad-index", f"{con.type} names entity {bad[0]}, which "
                               f"doesn't exist", tuple(con.indices)))
            continue
        if any(i in degen for i in named):
            out.append(Problem("bad-index", f"{con.type} is on collapsed "
                               f"{_name(se, next(i for i in named if i in degen))}",
                               tuple(con.indices)))
            continue
        try:
            msg = _check_constraint(se, con)
        except Exception as ex:          # a reader bug must not break replay
            msg = None
            print(f"[sketch_integrity] {con.type}: {ex}")
        if msg:
            out.append(Problem("constraint", msg, tuple(con.indices)))
    _check_snaps(se, out)
    _check_include_mids(se, out)
    if before is not None:
        _check_coincidence(se, before, out)
    return out


def warning_text(problems) -> str:
    if not problems:
        return ""
    more = f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""
    return f"Sketch: {problems[0].msg}{more}"
