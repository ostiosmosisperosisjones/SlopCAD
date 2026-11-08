"""How each reference resolved during a replay.

Every site that re-finds geometry (a face, edge, vertex, plane, sketch region
or solid) notes how it found it:

    EXACT  provenance (an origin, a region boundary) found it
    REF    the stored geometric ref matched strictly; the producer has no
           provenance kind (shell, sweep, loft, ...)
    FUZZY  a fallback matched: coplanar, moved, nearest centroid, category
           score, positional, bare index
    STALE  not found; the last position/value was kept

Matchers note only their hits; the site that gives up notes STALE (a raise
is already entry.error).  Recording is active only inside
History.replay_from, for the entry being replayed; note() anywhere else is
a no-op, so commit and preview paths are untouched.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple

EXACT, REF, FUZZY, STALE = "EXACT", "REF", "FUZZY", "STALE"
_RANK = {EXACT: 0, REF: 1, FUZZY: 2, STALE: 3}


class Note(NamedTuple):
    kind:   str
    site:   str
    detail: str = ""


class Ledger:
    def __init__(self):
        self.by_entry: dict[int, list[Note]] = {}
        self.entry: int | None = None


_active: ContextVar[Ledger | None] = ContextVar("resolve_ledger", default=None)


def note(kind: str, site: str, detail: str = "") -> None:
    led = _active.get()
    if led is None or led.entry is None:
        return
    led.by_entry.setdefault(led.entry, []).append(Note(kind, site, detail))


@contextmanager
def recording():
    """A ledger for one replay; a nested replay records into the outer one."""
    led = _active.get()
    if led is not None:
        yield led
        return
    led = Ledger()
    token = _active.set(led)
    try:
        yield led
    finally:
        _active.reset(token)


@contextmanager
def for_entry(index: int):
    """Attribute notes to history entry *index* while inside."""
    led = _active.get()
    if led is None:
        yield
        return
    prev, led.entry = led.entry, index
    led.by_entry.pop(index, None)        # a re-replay replaces, not appends
    try:
        yield
    finally:
        led.entry = prev


def worst(notes) -> str | None:
    return max((n.kind for n in notes), key=_RANK.__getitem__, default=None)


def warning_text(notes) -> str:
    """User-facing warning for an entry's FUZZY/STALE notes, or ""."""
    bad = [n for n in notes if _RANK[n.kind] >= _RANK[FUZZY]]
    if not bad:
        return ""
    stale = [n for n in bad if n.kind == STALE]
    fuzzy = [n for n in bad if n.kind == FUZZY]
    parts = []
    if stale:
        parts.append(f"{len(stale)} reference(s) not found, kept last position")
    if fuzzy:
        parts.append(f"{len(fuzzy)} reference(s) matched only approximately")
    first = bad[0]
    where = f"{first.site}: {first.detail}" if first.detail else first.site
    return ("After an upstream change, " + " and ".join(parts)
            + f" ({where}). Check the result; re-picking the reference may fix it.")


def describe(notes) -> str:
    """One line per note, for tests and debug output."""
    return "\n".join(f"  {n.kind:5s} {n.site}" + (f"  {n.detail}" if n.detail else "")
                     for n in notes)


def assert_exact(history, allow_ref: bool = True) -> None:
    """Raise AssertionError if any entry resolved FUZZY/STALE (or REF, when
    allow_ref is False) in its last replay."""
    limit = _RANK[FUZZY] if allow_ref else _RANK[REF]
    bad = []
    for i, e in enumerate(history._entries):
        notes = [n for n in getattr(e, "resolution", []) if _RANK[n.kind] >= limit]
        if notes:
            bad.append(f"entry {i} '{e.label}':\n{describe(notes)}")
    if bad:
        raise AssertionError("non-exact resolution:\n" + "\n".join(bad))
