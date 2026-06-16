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
    editing:      bool = False   # True while the ExtrudePanel is open editing this entry
    # Typed operation object — mirrors params, set by history.push()
    op:           Any  = field(default=None, repr=False)


class History:
    def __init__(self):
        self._entries:   list[HistoryEntry] = []
        self._cursor:    int = -1
        self._workspace: Any = None   # set by Workspace after construction

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def push(self, label, operation, params, body_id,
             face_ref, shape_before, shape_after,
             entry_id: str | None = None) -> HistoryEntry:
        from cad.op_types import Op
        insert_at = self._cursor + 1
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
        if insert_at < len(self._entries):
            self._entries.insert(insert_at, entry)
            for i in range(insert_at + 1, len(self._entries)):
                self._entries[i].diverged = True
            self._cursor = insert_at
            return entry
        self._entries.append(entry)
        self._cursor = insert_at
        return entry

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
            # For FaceExtrudeOp the sign is stored on the op distance field.
            # Rebuild op first, then patch sign.
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

    def undo(self) -> HistoryEntry | None:
        if self._cursor <= 0:
            return None
        entry = self._entries[self._cursor]
        self._cursor -= 1
        return entry

    def redo(self) -> HistoryEntry | None:
        if self._cursor >= len(self._entries) - 1:
            return None
        self._cursor += 1
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

        # Per-body state: current_shape and chain_broken flag.
        current_shapes: dict[str, Any]  = {bid: sv[1] for bid, sv in shape_cache.items()}
        chain_broken:   dict[str, bool] = {}

        first_error    = ""
        mutated_bodies: set[str] = set()

        self._replay_shape_cache = shape_cache
        try:
            for i in range(index, len(self._entries)):
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

                if isinstance(op, OffsetPlaneOp):
                    # Datum plane — no shape change.  Resolve to surface
                    # parametric errors (e.g. parent face vanished).  body_id
                    # is None for offset planes, so no chain-broken cascade.
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

        finally:
            self._replay_shape_cache = None

        return (not first_error), first_error, mutated_bodies

    # Legacy name — kept so existing viewer/serializer call sites keep working.
    # Both names invoke the same unified replay; pick replay_from() in new code.
    replay_all_from = replay_from

    def replay_sketch_dependents(self, sketch_entry_id: str) -> tuple[bool, str, set[str]]:
        """
        Replay every body chain that contains a SketchExtrudeOp or
        SketchRevolveOp referencing sketch_entry_id.  Called after a sketch is
        re-edited so that ops built from it are automatically recomputed.

        Returns (all_ok, first_error, mutated_body_ids).
        """
        from cad.op_types import SketchExtrudeOp, SketchRevolveOp

        earliest = None
        for i, entry in enumerate(self._entries):
            op = entry.op
            if not isinstance(op, (SketchExtrudeOp, SketchRevolveOp)):
                continue
            if op.from_sketch_id != sketch_entry_id:
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

def _replay_sketch_entry(se, history: History, before_index: int
                         ) -> tuple[bool, str]:
    """
    Update the plane cache and re-project all parametric ReferenceEntity
    instances on a SketchEntry.  Called during replay_from() for sketch ops.

    Never mutates se if resolution fails — the cached plane stays valid so
    the sketch doesn't visually corrupt to Z=0 while in an error state.
    Also fails if the body the sketch plane references is itself in error.
    """
    from cad.sketch import ReferenceEntity

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
    for ent in se.entities:
        if not isinstance(ent, ReferenceEntity) or ent.source is None:
            continue
        try:
            world_pts, occ_edges = ent.source.resolve(history, before_index)
        except Exception:
            # Skip this reference — its cached UV remains in place.
            continue
        ref_updates.append((ent, world_pts, occ_edges))

    # All resolved successfully — now commit.
    o = b3d_plane.origin
    x = b3d_plane.x_dir
    y = b3d_plane.y_dir
    z = b3d_plane.z_dir
    plane_origin = np.array([o.X, o.Y, o.Z], dtype=np.float64)
    plane_x_axis = np.array([x.X, x.Y, x.Z], dtype=np.float64)
    plane_y_axis = np.array([y.X, y.Y, y.Z], dtype=np.float64)
    se.plane_origin = plane_origin
    se.plane_x_axis = plane_x_axis
    se.plane_y_axis = plane_y_axis
    se.plane_normal = np.array([z.X, z.Y, z.Z], dtype=np.float64)

    for ent, world_pts, occ_edges in ref_updates:
        uv_pts = []
        for wp in world_pts:
            wp_arr = np.array(wp, dtype=np.float64)
            delta  = wp_arr - plane_origin
            u = float(np.dot(delta, plane_x_axis))
            v = float(np.dot(delta, plane_y_axis))
            uv_pts.append(np.array([u, v], dtype=np.float64))
        ent.points = uv_pts
        if occ_edges is not None:
            ent.occ_edges = occ_edges

    return True, ""


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
