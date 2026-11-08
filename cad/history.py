"""
cad/history.py

Single flat workspace history with cursor-based undo/redo and parametric replay.
Each entry is tagged with a body_id so operations are scoped to their body.

Every entry gets a stable UUID (entry_id) at push() time.  Cross-entry
references (sketch→extrude, parent→split-child) use these UUIDs instead of
fragile integer indices, so insert/delete/reorder never corrupts the graph.

Parametric replay
-----------------
replay_from(index) re-executes every entry from *index* onward in list order,
maintaining per-body shape state so that cross-body operations (e.g.
CrossBodyCutOp) always see the up-to-date shape of dependency bodies that
were already processed earlier in the same pass.

Two special cases beyond the REGISTRY executors:

  "sketch" entries
      Re-resolve the sketch plane via SketchEntry.plane_source and
      re-project every ReferenceEntity that carries a parametric EdgeSource.

  "extrude" / "cut" entries with op.from_sketch_id set
      These are sketch-profile extrudes.  The executor rebuilds the sketch
      profile faces from the SketchEntry and calls extrude_face_direct.
      After the extrude, any split-body import entries whose
      op.source_entry_id matches this entry's entry_id are updated to
      receive the correct solid from the new result.
"""

from __future__ import annotations
from cad.solid_ref import as_compound
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from cad.face_ref import FaceRef


@dataclass
class HistoryEntry:
    label:        str
    operation:    str
    params:       dict
    body_id:      str        # which body this operation targets
    face_ref:     FaceRef | None
    shape_before: Any        # shape of the body before this op
    shape_after:  Any        # shape of the body after this op
    # Stable identity — never changes regardless of list position
    entry_id:     str  = field(default_factory=lambda: str(uuid.uuid4()))
    # Authoritative op class name — survives operation string mutations
    op_type:      str  = ""
    # Non-destructive history state
    diverged:     bool = False   # True when a branch was inserted before this entry
    error:        bool = False   # True when this entry failed to replay
    error_msg:    str  = ""
    # Non-fatal: the entry replayed but not as authored (e.g. an included
    # edge couldn't be re-found and kept its last position).  Set by replay.
    warning:      str  = ""
    # Non-fatal: the result solid is damaged (inside-out, non-manifold,
    # self-intersecting).  Set by viewer/body_health.py, never saved.
    health_warning: str = ""
    # How each reference this entry re-found resolved in its last replay
    # (cad.resolve_ledger Notes).  Set by replay, never saved.
    resolution:   list = field(default_factory=list, repr=False)
    editing:      bool = False   # True while the ExtrudePanel is open editing this entry
    # Typed operation object — mirrors params, set by history.push()
    op:           Any  = field(default=None, repr=False)


class History:
    def __init__(self):
        self._entries:   list[HistoryEntry] = []
        self._cursor:    int = -1
        self._workspace: Any = None   # set by Workspace after construction
        from cad.params import ParamTable
        self.params = ParamTable()
        # Recent parameter sets the document rebuilt cleanly with, oldest
        # first (see "Known-good states" below).  Saved with the file.
        self.good_states: list[dict] = []
        # Evaluated parameter values, cached for the duration of one replay
        # (sketch replay reads them via param_values()).
        self._param_cache: tuple | None = None
        # Imported STEP files carried inside the project: {zip name: bytes}.
        # Import entries point here via params["embed"] (cad/importer.py).
        self.embedded: dict[str, bytes] = {}
        # (entry_id, index) armed by an edit's teardown: see reuse_entry_id().
        self._reuse_id: tuple[str, int] | None = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def push(self, label, operation, params, body_id,
             face_ref, shape_before, shape_after,
             entry_id: str | None = None) -> HistoryEntry:
        from cad.op_types import Op
        insert_at = self._cursor + 1
        reuse, self._reuse_id = getattr(self, "_reuse_id", None), None
        if entry_id is None and reuse is not None and reuse[1] == insert_at \
                and self.id_to_index(reuse[0]) is None:
            entry_id = reuse[0]
        op = Op.from_params(operation, params)
        entry = HistoryEntry(
            label=label, operation=operation, params=params,
            body_id=body_id, face_ref=face_ref,
            shape_before=shape_before, shape_after=shape_after,
            op=op,
            op_type=(type(op).__name__ if op is not None else operation),
        )
        if entry_id is not None:
            entry.entry_id = entry_id
        from cad.op_extrude import bind_along_normal
        bind_along_normal(self, entry)
        if insert_at < len(self._entries):
            self._entries.insert(insert_at, entry)
            for i in range(insert_at + 1, len(self._entries)):
                self._entries[i].diverged = True
            self._cursor = insert_at
            return entry
        self._entries.append(entry)
        self._cursor = insert_at
        return entry

    def reuse_entry_id(self, entry_id: str, index: int) -> None:
        """Arm an edit's re-commit: the next push, if it lands at *index*,
        takes *entry_id* (the deleted entry it replaces).  An edit tears the
        old entry down and commits a fresh op; without this the new entry gets
        a new uuid and every reference to the old one (face / edge origins
        naming the revolve, extrude or chamfer that made a face) dangles.
        Any push disarms it."""
        self._reuse_id = (entry_id, index)

    @property
    def is_mid_history(self) -> bool:
        """True when cursor is not at the last entry (diverged entries exist)."""
        return self._cursor < len(self._entries) - 1

    # ------------------------------------------------------------------
    # Op mutation
    # ------------------------------------------------------------------

    def update_entry_op(self, index: int, new_params: dict,
                        new_distance: float | None = None) -> None:
        """
        Update an entry's params (and optionally distance/sign) and
        rebuild entry.op and entry.operation atomically.

        new_distance is signed: positive → extrude, negative → cut.
        Caller triggers replay_from() after this.
        """
        from cad.op_types import Op
        if index < 0 or index >= len(self._entries):
            return
        entry = self._entries[index]
        entry.params.update(new_params)

        if new_distance is not None:
            entry.params["distance"] = abs(new_distance)
            # The sign lives in the operation ("cut" / "extrude"), which
            # Op.from_params turns back into a signed distance.
            is_cut = new_distance < 0
            entry.operation = "cut" if is_cut else "extrude"

        entry.op = Op.from_params(entry.operation, entry.params)
        entry.op_type = type(entry.op).__name__ if entry.op is not None else entry.operation

    # ------------------------------------------------------------------
    # UUID ↔ index resolution
    # ------------------------------------------------------------------

    def id_to_index(self, entry_id: str) -> int | None:
        """Return the current list index for a given entry_id, or None."""
        for i, e in enumerate(self._entries):
            if e.entry_id == entry_id:
                return i
        return None

    def index_to_id(self, index: int) -> str | None:
        """Return the entry_id at a given index, or None if out of range."""
        if 0 <= index < len(self._entries):
            return self._entries[index].entry_id
        return None

    def op_group(self, index: int) -> list[int]:
        """Sorted indices of everything one user action produced: the entry,
        its fan-out siblings (a no-target cut pushes one entry per body it hit,
        all tagged with the same params["fanout_id"]) and their split-body
        children (params["source_entry_id"])."""
        entries = self._entries
        if not 0 <= index < len(entries):
            return []
        members = self.fanout_members(index)
        ids = {entries[i].entry_id for i in members}
        children = [i for i, e in enumerate(entries)
                    if e.params.get("source_entry_id") in ids]
        return sorted(set(members) | set(children))

    def fanout_members(self, index: int) -> list[int]:
        """*index* plus its fan-out siblings (no split children)."""
        fid = (self._entries[index].params.get("fanout_id")
               if 0 <= index < len(self._entries) else None)
        if not fid:
            return [index]
        return [i for i, e in enumerate(self._entries)
                if e.params.get("fanout_id") == fid]

    def fanout_span(self, index: int) -> tuple[int, int] | None:
        """(first, last) index of the fan-out group *index* belongs to (as a
        member or a member's split child), or None if it isn't in one."""
        if not 0 <= index < len(self._entries):
            return None
        e = self._entries[index]
        if "fanout_id" not in e.params:
            src = self.id_to_index(e.params.get("source_entry_id") or "")
            if src is None or "fanout_id" not in self._entries[src].params:
                return None
            index = src
        g = self.op_group(index)
        return g[0], g[-1]

    def delete(self, index: int) -> None:
        """
        Remove the entry at *index* and clamp the cursor.
        No index-patching needed — cross-references use entry_id UUIDs.
        Caller is responsible for triggering replay_from() and rebuilding meshes.
        """
        if index < 0 or index >= len(self._entries):
            return
        del self._entries[index]
        if self._cursor >= len(self._entries):
            self._cursor = len(self._entries) - 1

    def reorder(self, src: int, dst: int) -> None:
        """
        Move the entry at *src* to *dst*.
        No index-patching needed — cross-references use entry_id UUIDs.
        Caller is responsible for triggering replay_from() and rebuilding meshes.
        """
        n = len(self._entries)
        if src == dst or src < 0 or src >= n or dst < 0 or dst >= n:
            return
        entry = self._entries.pop(src)
        self._entries.insert(dst, entry)
        if self._cursor == src:
            self._cursor = dst
        elif src < dst and src < self._cursor <= dst:
            self._cursor -= 1
        elif dst < src and dst <= self._cursor < src:
            self._cursor += 1
        self._cursor = max(0, min(self._cursor, len(self._entries) - 1))

    def step_target(self, delta: int) -> int:
        """Cursor index one undo (-1) / redo (+1) step away.  A fan-out group
        is one step: never land part-way through it."""
        t = self._cursor + delta
        span = self.fanout_span(t)
        if span is not None and t < span[1]:
            t = span[0] - 1 if delta < 0 else span[1]
        return t

    def undo(self) -> HistoryEntry | None:
        if self._cursor <= 0:
            return None
        entry = self._entries[self._cursor]
        self._cursor = max(self.step_target(-1), 0)
        return entry

    def redo(self) -> HistoryEntry | None:
        if self._cursor >= len(self._entries) - 1:
            return None
        self._cursor = self.step_target(+1)
        return self._entries[self._cursor]

    def seek(self, index: int) -> bool:
        """Move cursor to index. Returns True if index is valid."""
        if index < 0 or index >= len(self._entries):
            return False
        self._cursor = index
        return True

    # ------------------------------------------------------------------
    # Shape lookup (used by plane_ref / edge_ref during replay)
    # ------------------------------------------------------------------

    def _shape_for_body_at(self, body_id: str, before_index: int) -> Any | None:
        """
        Return the most recent shape for *body_id* up to (not including)
        *before_index*.

        During an active replay_from() call, _replay_shape_cache is set and
        contains already-computed shapes for all bodies seen so far — O(1).
        Outside of replay (e.g. initial plane resolve at sketch creation),
        fall back to a linear scan of the entry list.

        For force_new_body child bodies (no history entries of their own),
        falls back to workspace.source_shape when the parent entry is visible
        at before_index.
        """
        cache = getattr(self, '_replay_shape_cache', None)
        if cache is not None:
            entry = cache.get(body_id)
            if entry is not None:
                cached_idx, cached_shape = entry
                if cached_idx < before_index and cached_shape is not None:
                    return cached_shape
            # Cache miss or entry is at/after before_index — fall back to linear
            # scan so before_index is respected (e.g. sketch plane resolve called
            # with before_index < the current replay cursor).

        # Linear scan: most-recent shape for body_id strictly before before_index.
        result = None
        for i, e in enumerate(self._entries):
            if i >= before_index:
                break
            if e.body_id == body_id and e.shape_after is not None:
                result = e.shape_after
        if result is not None:
            return result

        # Last resort: workspace.source_shape for force_new_body children.
        # Only return it if the body's creation entry is visible at before_index.
        if self._workspace is not None:
            body = self._workspace.bodies.get(body_id)
            if body is not None and body.source_shape is not None:
                if body.created_at_entry_id is None:
                    return body.source_shape
                created_idx = self.id_to_index(body.created_at_entry_id)
                if created_idx is not None and created_idx < before_index:
                    return body.source_shape
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def entries_for_body(self, body_id: str) -> list[tuple[int, HistoryEntry]]:
        """Return [(global_index, entry)] for all entries belonging to body_id."""
        return [(i, e) for i, e in enumerate(self._entries)
                if e.body_id == body_id]

    # ------------------------------------------------------------------
    # Parametric replay
    # ------------------------------------------------------------------

    def replay_from(self, index: int) -> tuple[bool, str, set[str]]:
        """
        Re-execute every entry from *index* onward in list order, maintaining
        per-body shape state so cross-body ops (e.g. CrossBodyCutOp) see the
        up-to-date shape of dependency bodies already processed earlier in
        the same pass.

        On failure, the failing entry is marked with error=True/error_msg and
        downstream entries on the same body cascade-error.  Other bodies'
        chains continue independently.

        Returns (all_ok, first_error_message, mutated_body_ids) where
        mutated_body_ids is the set of body IDs whose shape_after changed —
        callers can use this to rebuild only the affected meshes.
        """
        from cad.op_types import ImportOp, SketchOp, OffsetPlaneOp

        if index < 0 or index >= len(self._entries):
            return True, "", set()

        # Build shape cache from all entries strictly before index.
        shape_cache: dict[str, tuple[int, Any]] = {}
        for j, e in enumerate(self._entries[:index]):
            if e.shape_after is not None:
                shape_cache[e.body_id] = (j, e.shape_after)
            # Non-primary split children of an earlier entry (seeded in-loop
            # below when the split itself is replayed).
            if e.shape_after is not None and self._workspace is not None:
                for cbid in e.params.get("child_body_ids", []):
                    body = self._workspace.bodies.get(cbid)
                    if (cbid != e.body_id and body is not None
                            and body.source_shape is not None):
                        shape_cache[cbid] = (j, body.source_shape)

        # Per-body state: current_shape and chain_broken flag.
        current_shapes: dict[str, Any]  = {bid: sv[1] for bid, sv in shape_cache.items()}
        chain_broken:   dict[str, bool] = {}

        first_error    = ""
        mutated_bodies: set[str] = set()

        self._replay_shape_cache = shape_cache
        self._param_cache = self.params.evaluate()
        from cad import resolve_ledger
        ledger_cm = resolve_ledger.recording()
        ledger = ledger_cm.__enter__()
        outer_entry = ledger.entry
        try:
            for i in range(index, len(self._entries)):
                ledger.entry = i
                ledger.by_entry.pop(i, None)
                entry  = self._entries[i]
                bid    = entry.body_id
                op     = entry.op
                broken = chain_broken.get(bid, False)

                entry.error     = False
                entry.error_msg = ""

                # Import / force_new_body child: shape is authoritative.
                if isinstance(op, ImportOp) or (
                        entry.shape_before is None
                        and entry.params.get("source_entry_id") is not None):
                    current_shapes[bid] = entry.shape_after
                    shape_cache[bid]    = (i, entry.shape_after)
                    continue

                if broken:
                    entry.error     = True
                    entry.error_msg = "Dependency error: a prior operation in this chain failed"
                    entry.shape_before = current_shapes.get(bid)
                    entry.shape_after  = None
                    _null_split_dependents(self._entries, i)
                    continue

                if op is None:
                    err = f"Entry {i} '{entry.label}': unknown operation '{entry.operation}'"
                    entry.error = True; entry.error_msg = err
                    entry.shape_before = current_shapes.get(bid); entry.shape_after = None
                    _null_split_dependents(self._entries, i)
                    chain_broken[bid] = True
                    if not first_error: first_error = err
                    mutated_bodies.add(bid)
                    continue

                if entry.params.get("exprs"):
                    try:
                        self._apply_bindings(i)
                    except ValueError as ex:
                        err = f"Entry {i} '{entry.label}': {ex}"
                        entry.error = True; entry.error_msg = err
                        entry.shape_before = current_shapes.get(bid)
                        entry.shape_after = None
                        _null_split_dependents(self._entries, i)
                        chain_broken[bid] = True
                        if not first_error: first_error = err
                        mutated_bodies.add(bid)
                        continue
                    op = entry.op

                current_shape = current_shapes.get(bid)

                if isinstance(op, SketchOp):
                    try:
                        op.execute(current_shape, self, i)
                    except Exception as ex:
                        entry.error = True; entry.error_msg = str(ex)
                        entry.shape_before = current_shape; entry.shape_after = None
                        chain_broken[bid] = True
                        if not first_error: first_error = str(ex)
                        mutated_bodies.add(bid)
                        continue
                    entry.shape_before = current_shape
                    entry.shape_after  = current_shape
                    continue

                if isinstance(op, OffsetPlaneOp) or getattr(op, "DATUM", False):
                    # Datum plane — no shape change.  Resolve to surface
                    # parametric errors (e.g. parent face vanished).  body_id
                    # is None for offset planes, so no chain-broken cascade.
                    entry.warning = ""
                    try:
                        op.execute(current_shape, self, i)
                    except Exception as ex:
                        entry.error = True; entry.error_msg = str(ex)
                        entry.shape_before = None; entry.shape_after = None
                        if not first_error: first_error = str(ex)
                        continue
                    entry.shape_before = None
                    entry.shape_after  = None
                    continue

                if current_shape is None and not op.creates_body_from_nothing(self, i):
                    err = f"Entry {i} '{entry.label}': no input shape (prior operation failed)"
                    entry.error = True; entry.error_msg = err
                    entry.shape_before = None; entry.shape_after = None
                    _null_split_dependents(self._entries, i)
                    chain_broken[bid] = True
                    if not first_error: first_error = err
                    mutated_bodies.add(bid)
                    continue

                entry.warning = ""
                try:
                    new_shape = op.execute(current_shape, self, i)
                except Exception as ex:
                    err = f"Entry {i} '{entry.label}' failed: {ex}"
                    entry.error = True; entry.error_msg = err
                    entry.shape_before = current_shape; entry.shape_after = None
                    _null_split_dependents(self._entries, i)
                    chain_broken[bid] = True
                    if not first_error: first_error = err
                    mutated_bodies.add(bid)
                    continue

                # Ops saved before region boundaries existed matched their
                # sketch region by centroid; persist the boundary that match
                # found so later parameter changes match by identity.
                from cad.profile_source import persist_backfilled_boundaries
                persist_backfilled_boundaries(op, entry.params)
                # Likewise a rim saved before rim origins (cad.face_origin).
                if getattr(op, "edge_refs", None) and "edge_refs" in entry.params:
                    entry.params["edge_refs"] = [r.to_dict() for r in op.edge_refs]

                # Split op: the parent keeps only its own solid(s). execute()
                # returns the FULL multi-solid boolean result (both halves of a
                # cut). Before trimming the parent, hand each split-away piece to
                # its ImportOp child body (their baked shape_after isn't
                # persisted across save/load — no STEP path — so replay must
                # re-derive it from the full result here). THEN trim the parent
                # to the retained solids by SolidRef; otherwise the parent body
                # renders every disconnected piece and a split reads as one
                # object. (Commit does the same in _push_result; this makes it
                # survive replay/reload.)
                retained_dicts = entry.params.get("retained_solid_refs")
                if retained_dicts and new_shape is not None:
                    _seed_split_import_children(self._entries, i, new_shape,
                                                current_shapes, shape_cache,
                                                mutated_bodies)
                    new_shape = _trim_to_retained_solids(new_shape, retained_dicts)

                entry.shape_before = current_shape
                entry.shape_after  = new_shape
                current_shapes[bid] = new_shape
                shape_cache[bid]    = (i, new_shape)
                entry.label         = _make_label(entry.operation, entry.params)
                mutated_bodies.add(bid)

                # Split ops produce extra child bodies whose geometry the op
                # wrote onto body.source_shape (the parent entry is tagged only
                # to the primary child). Seed those into the replay state at
                # this position so downstream ops targeting a non-primary child
                # find an input shape — without this they'd see None and error
                # with "no input shape".
                child_body_ids = entry.params.get("child_body_ids", [])
                for cbid in child_body_ids:
                    mutated_bodies.add(cbid)
                    if cbid == bid:
                        continue  # primary child already seeded via new_shape
                    body = (self._workspace.bodies.get(cbid)
                            if self._workspace is not None else None)
                    if body is not None and body.source_shape is not None:
                        current_shapes[cbid] = body.source_shape
                        shape_cache[cbid]    = (i, body.source_shape)

            ledger.entry = outer_entry
            for i in range(index, len(self._entries)):
                entry = self._entries[i]
                entry.resolution = ledger.by_entry.get(i, [])
                if not entry.error and not entry.warning:
                    entry.warning = resolve_ledger.warning_text(entry.resolution)
                    if entry.warning:
                        print(f"[Replay] Entry {i}: {entry.warning}")

        finally:
            ledger.entry = outer_entry
            ledger_cm.__exit__(None, None, None)
            self._replay_shape_cache = None
            self._param_cache = None

        if not first_error and len(self.params.params):
            self._remember_good(self.params.evaluate()[0], self.params.to_list())
        return (not first_error), first_error, mutated_bodies

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def param_values(self) -> dict:
        """Evaluated parameter values (name -> Quantity).  Failed params are
        absent, so a binding that uses one errors with "Unknown name"."""
        if self._param_cache is not None:
            return self._param_cache[0]
        return self.params.evaluate()[0]

    def _apply_bindings(self, index: int) -> None:
        """Recompute an entry's bound fields from their expressions and
        rebuild entry.op if anything changed.  Raises ValueError."""
        from cad.params import eval_binding
        entry  = self._entries[index]
        values = self.param_values()
        new = {}
        for key, b in entry.params.get("exprs", {}).items():
            try:
                v = eval_binding(b, values)
            except ValueError as ex:
                raise ValueError(f"{key} = {b.get('expr')!r}: {ex}") from ex
            # Distances are stored as magnitudes; the sign is the op type.
            # Bound fields are magnitudes: the panel fields that create
            # bindings only take positive values, and some ops store the
            # value signed (CrossBodyCutOp distance, flipped thicken) — the
            # sign is the op's, not the expression's.
            if v < 0:
                raise ValueError(f"{key} = {b.get('expr')!r} evaluates to "
                                 f"{v:g} — must be positive")
            field, _, slot = key.partition(".")
            if slot:
                # "edge_radii.2": a slot of a list field (fillet per-edge
                # radius overrides).
                lst = new.get(field, list(entry.params.get(field) or []))
                pos = int(slot)
                if pos >= len(lst):
                    lst.extend([None] * (pos + 1 - len(lst)))
                if lst[pos] != v:
                    lst[pos] = v
                    new[field] = lst
                continue
            old = entry.params.get(key, 0)
            if isinstance(old, (int, float)) and old < 0:
                v = -v
            if entry.params.get(key) != v:
                new[key] = v
        if new:
            self.update_entry_op(index, new)

    def param_users(self, names: set[str]) -> list[int]:
        """Indices of entries whose op bindings or sketch dimensions
        reference any of *names*."""
        from cad.params import binding_refs, _safe_refs
        out = []
        for i, e in enumerate(self._entries):
            refs: set[str] = set()
            for b in e.params.get("exprs", {}).values():
                refs |= binding_refs(b)
            se = e.params.get("sketch_entry")
            for c in getattr(se, "constraints", None) or []:
                if getattr(c, "expr", None):
                    refs |= _safe_refs(c.expr)
            if refs & names:
                out.append(i)
        return out

    def _changed_params(self, before: dict, after: dict) -> set[str]:
        return {n for n in set(before) | set(after)
                if before.get(n) != after.get(n)}

    def set_param(self, param) -> tuple[bool, str, set[str]]:
        """Add/replace a parameter and replay everything it affects.
        Raises ParamError if the parameter itself doesn't evaluate."""
        before, _ = self.params.evaluate()
        items = self.params.to_list()
        self.params.set(param)
        return self._replay_for_params(before, items)

    def remove_param(self, name: str) -> None:
        """Remove an unused parameter.  Raises ParamError if params or
        history entries still reference it."""
        from cad.params import ParamError
        users = self.param_users({name})
        if users:
            labels = ", ".join(self._entries[i].label.strip() for i in users[:5])
            raise ParamError(f"'{name}' is used by history: {labels}")
        self.params.remove(name)

    def rename_param(self, old: str, new: str) -> None:
        """Rename a parameter everywhere it's referenced.  Values don't
        change, so no replay is needed."""
        from cad.expression import rename_ref
        self.params.rename(old, new)
        self.good_states.clear()         # keyed by the old name
        for e in self._entries:
            for b in e.params.get("exprs", {}).values():
                b["expr"] = rename_ref(b["expr"], old, new)
            se = e.params.get("sketch_entry")
            for c in getattr(se, "constraints", None) or []:
                if getattr(c, "expr", None):
                    c.expr = rename_ref(c.expr, old, new)

    def replace_params(self, items: list[dict]) -> tuple[bool, str, set[str]]:
        """Swap in a whole parameter table (undo/redo) and replay whatever
        changed value."""
        from cad.params import ParamTable
        before, _ = self.params.evaluate()
        old_items = self.params.to_list()
        self.params = ParamTable.from_list(items)
        return self._replay_for_params(before, old_items)

    def _replay_for_params(self, before: dict,
                           before_items: list | None = None) -> tuple[bool, str, set[str]]:
        """Replay what the change from values *before* affects.
        before_items: the table those values came from; the state the
        document is leaving is remembered under it if it was a good one."""
        if before_items is not None:
            self._remember_good(before, before_items)
        after, _ = self.params.evaluate()
        users = self.param_users(self._changed_params(before, after))
        if not users:
            return True, "", set()
        state = self._good_state_for(after)
        if state is not None:
            self._restore_good(state, users[0])
        return self.replay_from(users[0])

    # ------------------------------------------------------------------
    # Known-good states
    # ------------------------------------------------------------------
    # A sketch's solved coordinates are the only record of where its free
    # geometry was, and each solve starts from the last one, so the outcome
    # can depend on the path taken and a failed replay leaves its mark.  So
    # the sketches are remembered each time the document builds cleanly, by
    # parameter values: entering those values again brings back exactly that
    # state, and revert_to_good() gets back to the latest one from anywhere.

    MAX_GOOD_STATES = 12

    def _sketches(self):
        return [(i, e, e.params["sketch_entry"]) for i, e in enumerate(self._entries)
                if e.params.get("sketch_entry") is not None]

    def _doc_signature(self) -> list:
        """Which entries and which version of each sketch a state is for."""
        return [[e.entry_id, se.rev if (se := e.params.get("sketch_entry")) else None]
                for e in self._entries]

    def _doc_is_good(self) -> bool:
        """Built with nothing failed or patched up: no errors, no warnings,
        no lost includes, no line crushed to nothing."""
        from cad.sketch import LineEntity
        if any(e.error or e.warning for e in self._entries):
            return False
        for _, _, se in self._sketches():
            if getattr(se, "stale_includes", None):
                return False
            if any(isinstance(l, LineEntity) and np.linalg.norm(l.p1 - l.p0) < 1e-6
                   for l in se.entities):
                return False
        return True

    def _remember_good(self, values: dict, items: list) -> None:
        if not self._entries or not items or not self._doc_is_good():
            return
        from cad.sketch import sketch_geometry
        state = {
            "values":    {k: float(v.value) for k, v in values.items()},
            "items":     items,
            "signature": self._doc_signature(),
            # Only sketches a parameter can move need restoring: bound
            # dimensions, or includes of something a parameter moves.
            "sketches":  {e.entry_id: sketch_geometry(se) for _, e, se in self._sketches()
                          if se.links is not None
                          or any(getattr(c, "expr", None) for c in se.constraints)},
        }
        self.good_states = [s for s in self.good_states
                            if s["values"] != state["values"]]
        self.good_states.append(state)
        del self.good_states[:-self.MAX_GOOD_STATES]

    def _good_state_for(self, values: dict) -> dict | None:
        """The state remembered for these parameter values, if it still fits
        this document.  Parameters added or removed since don't matter."""
        sig = self._doc_signature()
        for s in reversed(self.good_states):
            common = set(s["values"]) & set(values)
            if s["signature"] == sig and common and all(
                    abs(s["values"][k] - values[k].value) < 1e-9 for k in common):
                return s
        return None

    def _restore_good(self, state: dict, from_index: int = 0) -> None:
        from cad.sketch import restore_sketch_geometry
        for i, e, se in self._sketches():
            if i >= from_index and e.entry_id in state["sketches"]:
                restore_sketch_geometry(se, state["sketches"][e.entry_id])

    def last_good(self) -> dict | None:
        """The latest good state that fits this document, or None."""
        sig = self._doc_signature()
        return next((s for s in reversed(self.good_states) if s["signature"] == sig), None)

    def revert_to_good(self) -> tuple[bool, str, set[str]]:
        """Back to the parameters and sketches of the latest good state."""
        from cad.params import ParamTable
        state = self.last_good()
        if state is None:
            return False, "No working state remembered for this document", set()
        self.params = ParamTable.from_list(state["items"])
        self._restore_good(state)
        return self.replay_from(0)

    # Legacy name — kept so existing viewer/serializer call sites keep working.
    # Both names invoke the same unified replay; pick replay_from() in new code.
    replay_all_from = replay_from

    def replay_sketch_dependents(self, sketch_entry_id: str) -> tuple[bool, str, set[str]]:
        """
        Replay every body chain that contains an op with a profile drawn from
        sketch_entry_id (extrude, cut, revolve, loft, sweep …).  Called after
        a sketch is re-edited so that ops built from it are recomputed.

        Returns (all_ok, first_error, mutated_body_ids).
        """
        earliest = None
        for i, entry in enumerate(self._entries):
            ids = getattr(entry.op, "from_sketch_ids", None)
            if not ids or sketch_entry_id not in ids:
                continue
            if earliest is None or i < earliest:
                earliest = i

        if earliest is None:
            return True, "", set()

        all_ok, first_err, mutated = self.replay_from(earliest)

        return all_ok, first_err, mutated

    @property
    def entries(self) -> list[HistoryEntry]:
        return list(self._entries)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._entries) - 1

    def __len__(self):
        return len(self._entries)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _replay_sketch_entry(se, history: History, before_index: int,
                         own: bool = False) -> tuple[bool, str]:
    """
    Update the plane cache and re-project all parametric ReferenceEntity
    instances on a SketchEntry.  Called during replay_from() for sketch ops.

    Never mutates se if resolution fails — the cached plane stays valid so
    the sketch doesn't visually corrupt to Z=0 while in an error state.
    Also fails if the body the sketch plane references is itself in error.

    own=True: this is the sketch's own entry replaying (not a consumer
    reprojecting it later).  An include that can't be re-found there is
    recorded in se.stale_includes and drops its cached occ edge, so the
    profile is built from the same UV points a reload would use.
    """
    from cad.sketch import ReferenceEntity
    from cad import resolve_ledger

    # Check that the body this sketch plane depends on is not in an error state.
    # FacePlaneSource depends on a specific body; if that body's chain is broken
    # the plane resolve might "succeed" but return stale/wrong geometry.
    plane_body_id = getattr(se.plane_source, 'body_id', None)
    if plane_body_id is not None:
        # Walk entries up to before_index looking for the last entry on that body.
        last_entry_for_body = None
        for e in history._entries[:before_index]:
            if e.body_id == plane_body_id:
                last_entry_for_body = e
        if last_entry_for_body is not None and last_entry_for_body.error:
            return False, f"Sketch plane body '{plane_body_id}' is in an error state"

    if getattr(se.plane_source, "origin", False) is None and se.plane_origin is not None:
        # An old pick with no provenance: the plane it last resolved to,
        # so FacePlaneSource can recognise that face and bake an origin.
        se.plane_source.last_plane = (tuple(map(float, se.plane_origin)),
                                      tuple(map(float, se.plane_normal)))
    if se.plane_origin is not None:
        # Saved before face planes had an anchor: bake it from the origin the
        # sketch's points were saved against (FacePlaneSource.plane_for).
        ps = se.plane_source
        while ps is not None and hasattr(ps, "parent"):
            ps = ps.parent
        if ps is not None and hasattr(ps, "anchor") and ps.anchor is None:
            ps.seed_origin = tuple(map(float, se.plane_origin))
    try:
        b3d_plane = se.plane_source.resolve(history, before_index)
    except Exception as ex:
        return False, f"Sketch plane resolve failed: {ex}"

    # Resolve all reference entities.  Failures here are non-fatal: a
    # reference whose source has been deleted (e.g. an edge consumed by a
    # later cut) keeps its previously-baked UV cache, which is at least
    # stable.  Without this fallback the entire sketch — including all the
    # user's drawn geometry — would fail to replay just because one Include
    # reference can no longer be re-found.
    ref_updates = []
    stale = []
    for i, ent in enumerate(se.entities):
        if not isinstance(ent, ReferenceEntity):
            continue
        if ent.source is None:
            # A body edge / vertex included without a source never follows
            # its geometry: say so rather than keep it silently.
            if getattr(ent, "source_type", None) in ("edge", "vertex"):
                resolve_ledger.note(resolve_ledger.STALE, "include",
                                    f"entity {i}: included {ent.source_type} has no "
                                    f"source, frozen where it was included")
            continue
        try:
            world_pts, occ_edges = ent.source.resolve(history, before_index)
        except Exception as ex:
            # Skip this reference — its cached UV remains in place.
            stale.append((i, str(ex)))
            resolve_ledger.note(resolve_ledger.STALE, "include",
                                f"entity {i}: {ex}")
            continue
        ref_updates.append((ent, world_pts, occ_edges))
    if own:
        se.stale_includes = stale
        for i, _ in stale:
            se.entities[i].occ_edges = None

    # Where drawn points sit on includes, measured before the includes move.
    from cad.sketch import (include_attachments, attachment_targets, _drawn_points,
                            remember_links)
    attachments = include_attachments(se, trust_links=True) if ref_updates else []
    if ref_updates:
        remember_links(se, attachments)
    sides = se.dimension_sides() if ref_updates else None
    before = {id(ent): [p.copy() for p in ent.points] for ent, _, _ in ref_updates}

    # All resolved successfully — now commit.
    v = lambda d: np.array([d.X, d.Y, d.Z], dtype=np.float64)
    frame = (v(b3d_plane.origin), v(b3d_plane.x_dir), v(b3d_plane.y_dir), v(b3d_plane.z_dir))
    if not own:
        frame = _sketch_frame_for_consumer(se, history, before_index, frame)
    plane_origin, plane_x_axis, plane_y_axis, plane_normal = frame
    se.plane_origin = plane_origin
    se.plane_x_axis = plane_x_axis
    se.plane_y_axis = plane_y_axis
    se.plane_normal = plane_normal

    def _to_uv(wp):
        delta = np.array(wp, dtype=np.float64) - plane_origin
        return np.array([np.dot(delta, plane_x_axis),
                         np.dot(delta, plane_y_axis)], dtype=np.float64)

    from cad.edge_ref import BodyEdgeSource
    from cad.sketch_tools.include import project_body_edge
    for ent, world_pts, occ_edges in ref_updates:
        if isinstance(ent.source, BodyEdgeSource):
            # Same projection the Include tool used, so the include keeps
            # its point count and the drawn geometry snapped to it.
            ent.points, ent.occ_edges = project_body_edge(
                world_pts, (occ_edges or [None])[0], _to_uv, se.plane_normal,
                collapse=len(ent.points) <= 2)
            continue
        ent.points = [_to_uv(wp) for wp in world_pts]
        if occ_edges is not None:
            ent.occ_edges = occ_edges
    # A re-found edge can come back with its parameter running the other
    # way.  Drawn ends are attached to the include's first/last point, so a
    # flipped include swaps them and the profile crosses itself (walker
    # seed 4: a quad became a bowtie).  Keep the previous orientation.
    for ent, _, _ in ref_updates:
        old, new = before[id(ent)], ent.points
        if len(old) < 2 or len(new) < 2:
            continue
        d_old, d_new = old[-1] - old[0], new[-1] - new[0]
        if np.linalg.norm(d_old) > 1e-6 and float(d_old @ d_new) < 0:
            ent.points = new[::-1]
    from cad.sketch import refresh_ref_circles
    refresh_ref_circles(se)

    # Re-resolving an unchanged include jitters by ~1e-7; only a real move
    # is worth a re-solve.
    def _moved(ent):
        old = before[id(ent)]
        return len(old) != len(ent.points) or any(
            np.linalg.norm(a - b) > 1e-6 for a, b in zip(old, ent.points))
    moved = {i for i, ent in enumerate(se.entities)
             if id(ent) in before and _moved(ent)}

    targets = attachment_targets(se, attachments) if attachments else {}
    current = dict(_drawn_points(se.entities))
    any_pin_moved = any(np.linalg.norm(uv - current[k]) > 1e-6
                        for k, uv in targets.items())
    # Constraints naming a moved include (equal / parallel to it) need a
    # re-solve even when no drawn point touches it.
    refs_moved = bool(moved) and any(set(c.indices) & moved for c in se.constraints)
    # Once re-solving, hold EVERY attached point at its target, moved or
    # not: one left only on a sliding constraint gets dragged along its
    # include by the snap implicit lengths of lines whose other end moved.
    pins = targets if (any_pin_moved or refs_moved) else {}
    se._held_sides = sides
    try:
        return _resolve_bound_dimensions(se, history, pins, attachments, refs_moved)
    finally:
        se._held_sides = None


def _resolve_bound_dimensions(se, history: History, pins: dict | None = None,
                              attachments: list | None = None,
                              refs_moved: bool = False) -> tuple[bool, str]:
    """Recompute parameter-driven dimension values and re-solve the sketch
    if any changed, or if included geometry moved under attached points
    (pins) or under constraints that reference it (refs_moved).  Untouched
    sketches are never re-solved, so files without parameters or moving
    includes replay exactly as before.

    attachments: include_attachments() measured BEFORE the includes were
    re-projected (afterwards the drawn points no longer sit on them)."""
    from cad.expression import parse_expr
    bound = [c for c in se.constraints if getattr(c, "expr", None)]
    follow = bool(pins) or refs_moved
    if not bound and not follow:
        return True, ""
    values = history.param_values()
    changed = False
    seed: set[int] = set()
    for c in bound:
        try:
            v = parse_expr(c.expr, c.expr_unit, values)
        except ValueError as ex:
            return False, f"Dimension {c.expr!r}: {ex}"
        if v <= 0:
            return False, f"Dimension {c.expr!r} evaluates to {v:g} — must be positive"
        if abs(v - c.value) > 1e-9:
            c.value = v
            changed = True
            # A gap moves its second line, like the dimension tool does.
            seed.update(c.indices[1:] if c.type == 'gap' else c.indices)
    # A parameter edit moves only what it has to (apply_constraints_local);
    # following moved includes stays a global solve.
    if follow:
        ok = se.apply_all_constraints(pins=pins, attachments=attachments)
    else:
        ok = not changed or se.apply_constraints_local(seed, attachments=attachments)
    if (changed or follow) and not ok:
        if follow and not changed:
            return False, ("Sketch can't follow its included geometry: "
                           "the constraint solver failed")
        return False, "Sketch constraints can't be solved with the new parameter values"
    return True, ""


def _sketch_frame_for_consumer(se, history: History, before_index: int, found):
    """The plane a consumer (a cut / extrude / sweep … reprojecting the
    sketch at its own position) puts the sketch on.

    A face plane's origin is the face's centroid (build123d Plane(face)), so
    an op between the sketch and its consumer that only changes the face's
    outline (a sibling cut of a no-target fan-out, a pocket from another
    loop, a boss merged onto it) moved the centroid and dragged the profile
    across the face with it: replay cut somewhere the commit didn't.

    The sketch's frame is the one at its own position.  Moved by any
    TransformOps on the face's body since (cad.body_motion), and while the
    face found here is still parallel (same sense) to it, the consumer keeps
    that frame, only moved along the normal onto the face (so a face pushed
    out by a later op still carries the sketch).  A face that turned uses
    the face plane, as before.  *found* = (origin, x, y, normal)."""
    import numpy as np
    idx = next((i for i, e in enumerate(history._entries)
                if e.params.get("sketch_entry") is se), None)
    if idx is None or idx >= before_index or se.plane_source is None:
        return found
    try:
        own = se.plane_source.resolve(history, idx)
    except Exception:
        return found
    v = lambda d: np.array([d.X, d.Y, d.Z], dtype=np.float64)
    o, x, y, n = v(own.origin), v(own.x_dir), v(own.y_dir), v(own.z_dir)
    body = getattr(se.plane_source, "body_id", None)
    if body:
        from cad import body_motion as BM
        _, T = BM.motion(history, body, before_index, idx)
        if T is not None:
            o, x, y, n = BM.pt(T, o), BM.vec(T, x), BM.vec(T, y), BM.vec(T, n)
    fo, _, _, fn = found
    if abs(float(n @ fn) - 1.0) > 1e-9:
        return found
    return o + float((fo - o) @ n) * n, x, y, n


def reproject_consumed_sketch(se, history: History, before_index: int,
                              sketch_entry_rec) -> tuple[bool, str]:
    """Reproject a sketch's plane for an op that *consumes* it (a cut/revolve
    cut/extrude driven by from_sketch_id / source_sketch_id).

    Reprojecting at the consuming op's position lets the profile follow a parent
    face that intervening ops moved — the intended behavior.  But when an
    intervening op *destroys* the anchor face (e.g. a self-cut splits the face
    the sketch sat on), reprojection at that later position can no longer locate
    it, even though the sketch's plane is perfectly well-defined at the position
    where it was authored.

    Rule:
      1. Try reprojection at before_index (follow the moving face).
      2. If that fails, retry at the sketch's *own* creation position.  Success
         there means the plane is genuinely well-defined and only the later
         anchor-face destruction broke step 1 — use that (now-baked) plane.
      3. If even the sketch's own position can't resolve the plane, it's truly
         unresolvable — fail.

    Retrying at the sketch's own index is self-validating: it doesn't trust a
    stale `error` flag (a never-replayed sketch reports error=False) but instead
    re-derives the plane from scratch, so an unresolvable plane stays fatal.

    Returns (ok, err).
    """
    ok, err = _replay_sketch_entry(se, history, before_index)
    if ok:
        return True, ""

    sketch_idx = None
    if sketch_entry_rec is not None:
        sketch_idx = history.id_to_index(sketch_entry_rec.entry_id)
    if sketch_idx is not None:
        ok2, _ = _replay_sketch_entry(se, history, sketch_idx)
        if ok2:
            # Plane is valid at authoring time; the consumer-position failure
            # was an intervening op destroying the anchor face. Use the plane
            # just baked at the sketch's own position.
            return True, ""

    return False, err


def _trim_to_retained_solids(shape, retained_dicts):
    """Return a Compound of just the solids in *shape* that match the stored
    SolidRefs — the parent body's own pieces after a split.

    Matches each retained ref to its best solid by geometric fingerprint (robust
    to OCCT solid-ordering churn across a rebuild) and never claims a solid
    twice.  If nothing matches (e.g. the op degenerated), the original shape is
    returned unchanged so the body doesn't silently vanish.
    """
    from cad.solid_ref import solid_refs_from_dicts
    from build123d import Compound
    from cad.op_base import _compound_of

    refs = solid_refs_from_dicts(retained_dicts)
    if not refs:
        return shape
    solids = list(shape.solids())
    if len(solids) <= len(refs):
        # Nothing split off (or fewer solids than expected) — keep as-is.
        return shape

    chosen, used = [], set()
    for ref in refs:
        idx, _ = ref.find_in(shape)
        if idx is None or idx in used:
            # Fall back to the first unused solid so we still trim something
            # sane rather than dropping the body.
            idx = next((j for j in range(len(solids)) if j not in used), None)
            if idx is not None and len(solids) - len(used) > 1:
                from cad import resolve_ledger
                resolve_ledger.note(resolve_ledger.FUZZY, "retained solid",
                                    f"positional pick {idx}")
        if idx is not None:
            used.add(idx)
            chosen.append(solids[idx])

    if not chosen:
        return shape
    return Compound(_compound_of(chosen))


def _seed_split_import_children(entries, parent_idx, full_shape,
                                current_shapes, shape_cache, mutated_bodies):
    """Give each split-away ImportOp child body its solid from the parent op's
    FULL (pre-trim) result.

    The `split_from`/`solid_index` child bodies bake their geometry at commit,
    but that shape isn't persisted across save/load (no STEP path), so on reload
    the child would have no shape once the parent trims itself.  Here we recover
    each child's solid from the full multi-solid result: by SolidRef fingerprint
    when the child stored one (robust to OCCT ordering), else by solid_index.
    """
    from build123d import Compound
    from cad.solid_ref import SolidRef

    solids = list(full_shape.solids())
    if not solids:
        return
    parent_id = entries[parent_idx].entry_id
    used = set()
    for e in entries[parent_idx + 1:]:
        if e.operation != "import":
            continue
        if e.params.get("source_entry_id") != parent_id:
            continue
        idx = None
        ref_dict = e.params.get("solid_ref")
        if ref_dict is not None:
            ref = SolidRef(volume=float(ref_dict["volume"]),
                           com=tuple(ref_dict["com"]),
                           extents=tuple(ref_dict["extents"]))
            j, _ = ref.find_in(full_shape)
            if j is not None and j not in used:
                idx = j
        if idx is None:
            si = int(e.params.get("solid_index", 0))
            idx = si if (0 <= si < len(solids) and si not in used) else \
                next((k for k in range(len(solids)) if k not in used), None)
            if idx is not None and len(solids) - len(used) > 1:
                from cad import resolve_ledger
                resolve_ledger.note(resolve_ledger.FUZZY, "split child solid",
                                    f"positional pick {idx}")
        if idx is None:
            continue
        used.add(idx)
        child_shape = as_compound(solids[idx])
        e.shape_after = child_shape
        cbid = e.body_id
        current_shapes[cbid] = child_shape
        shape_cache[cbid]    = (parent_idx, child_shape)
        mutated_bodies.add(cbid)


def _null_split_dependents(entries: list, parent_idx: int) -> None:
    """
    Null the shape_after of any split-body import entries that were produced
    by the entry at *parent_idx*.  Called whenever a parent op errors so the
    child bodies disappear from the viewport instead of showing stale geometry.
    """
    parent_id = entries[parent_idx].entry_id
    for e in entries[parent_idx + 1:]:
        if (e.operation == "import" and
                e.params.get("source_entry_id") == parent_id):
            e.shape_after = None


def _make_label(operation: str, params: dict) -> str:
    from cad.units import format_op_label
    return format_op_label(operation, params)
