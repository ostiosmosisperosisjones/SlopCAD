"""
viewer/vp_history.py

HistoryMixin — undo/redo/seek/delete/reorder/replay and mesh-rebuild helpers.

Expects self to have:
  history, workspace, _meshes, _body_visible, _hovered_vertex, _hovered_edge,
  _rebuild_body_mesh(), _rebuild_all_meshes(), _rebuild_sketch_faces(),
  selection, history_changed signal.
"""

from __future__ import annotations


class HistoryMixin:

    # ------------------------------------------------------------------
    # Mesh rebuild helpers
    # ------------------------------------------------------------------

    def _post_push_cascade(self, primary_body_id: str | None = None):
        """
        Called after every history.push().

        Mid-history insert: replay the affected body forward so diverged
        entries get correct shapes or error flags, then rebuild affected meshes.
        Tip-of-history append: just rebuild the primary body — cheaper.

        Also clears the structural command stack because entry indices are
        invalidated by any new push.
        """
        self._cmd_stack.clear()
        if self.history.is_mid_history:
            cursor = self.history.cursor
            ok, err, mutated = self.history.replay_from(cursor)
            if not ok:
                print(f"[Cascade] Replay after insert: {err}")
            # Advance cursor to tip so current_shape() sees bodies created after
            # the edit point (e.g. the result body of a boolean op).
            self.history.seek(len(self.history.entries) - 1)
            self._rebuild_bodies(mutated or ({primary_body_id} if primary_body_id else set()))
        else:
            if primary_body_id is not None:
                self._rebuild_body_mesh(primary_body_id)
            else:
                self._rebuild_all_meshes()

    def _rebuild_bodies(self, body_ids: set):
        """
        Rebuild meshes for a specific set of body IDs.
        Falls back to _rebuild_all_meshes if the set is empty.
        Tessellates bodies in parallel, then uploads to GL in one pass.
        """
        if not body_ids:
            self._rebuild_all_meshes()
            return
        if len(body_ids) == 1:
            self._rebuild_body_mesh(next(iter(body_ids)))
            return
        from viewer.viewport import _tessellate_parallel
        from OpenGL.GL import glDeleteBuffers
        bodies = [
            (bid, self.workspace.current_shape(bid))
            for bid in body_ids
            if self.workspace.current_shape(bid) is not None
        ]
        for bid in body_ids:
            self._body_visible.setdefault(bid, True)
        new_meshes = _tessellate_parallel(bodies)

        self.makeCurrent()
        for bid in body_ids:
            old = self._meshes.get(bid)
            if old:
                for buf in (old.vbo_verts, old.vbo_normals, old.vbo_edges, old.ebo):
                    if buf is not None:
                        glDeleteBuffers(1, [buf])
            if bid in new_meshes:
                new_meshes[bid].upload()
                self._meshes[bid] = new_meshes[bid]
            else:
                self._meshes.pop(bid, None)
            self.selection.clear_for_body(bid)
        self._hovered_vertex = (None, None)
        self._hovered_edge   = (None, None)
        self.doneCurrent()
        self._rebuild_sketch_faces()
        self.selection_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Undo / Redo / Seek / Replay
    # ------------------------------------------------------------------

    def _do_undo(self):
        entry = self.history.undo()
        if entry is None:
            print("[Undo] Nothing to undo."); return
        print(f"[Undo] Reverting: {entry.label}")
        if entry.operation == "sketch":
            self._rebuild_sketch_faces()
        # Rebuild all workspace bodies — undo may hide split children that were
        # created by the reverted entry, and _rebuild_all_meshes respects cursor.
        self._rebuild_all_meshes()
        self.history_changed.emit()

    def _do_redo(self):
        entry = self.history.redo()
        if entry is None:
            print("[Redo] Nothing to redo."); return
        print(f"[Redo] Restoring: {entry.label}")
        if entry.operation == "sketch":
            self._rebuild_sketch_faces()
        self._rebuild_all_meshes()
        self.history_changed.emit()

    def handle_undo(self):
        """
        Three-level context-aware undo:
          1. Inside sketch  → per-sketch entity undo
          2. Outside sketch, structural commands pending → undo reorder/delete
          3. Otherwise      → history cursor undo (CAD operation)
        """
        if self._sketch is not None:
            if self._sketch.undo_entity():
                print("[Sketch] Undo last entity")
                self.update()
            else:
                print("[Sketch] Nothing to undo in sketch")
        elif self._cmd_stack.can_undo:
            desc = self._cmd_stack.undo(self)
            print(f"[Cmd] Undo: {desc}")
            self.history_changed.emit()
        else:
            self._do_undo()

    def handle_redo(self):
        """
        Context-aware redo:
          1. Inside sketch  → per-sketch redo
          2. Structural commands available → redo reorder/delete
          3. Otherwise      → history cursor redo
        """
        if self._sketch is not None:
            if self._sketch.redo_entity():
                self.sketch_mode_changed.emit(True)
                self.update()
            return
        if self._cmd_stack.can_redo:
            desc = self._cmd_stack.redo(self)
            print(f"[Cmd] Redo: {desc}")
            self.history_changed.emit()
        else:
            self._do_redo()

    def seek_history(self, index: int):
        if self.history.seek(index):
            self._rebuild_all_meshes()
            self.history_changed.emit()

    def do_replay(self, from_index: int):
        print(f"[Replay] Replaying from entry {from_index}…")
        ok, err, mutated = self.history.replay_from(from_index)
        if not ok:
            print(f"[Replay] FAILED: {err}")
        self._rebuild_bodies(mutated)
        self.history_changed.emit()
        print("[Replay] Done.")

    # ------------------------------------------------------------------
    # Delete / Reorder  (internal implementations)
    # ------------------------------------------------------------------

    def _do_delete(self, index: int):
        """Raw delete — called by DeleteCommand.apply() and legacy callers."""
        entries = self.history.entries
        if index < 0 or index >= len(entries):
            return
        body_id = entries[index].body_id
        print(f"[History] Deleting entry {index}: '{entries[index].label}'")

        deleted_entry = entries[index]
        child_body_ids = list(deleted_entry.params.get("child_body_ids", []))
        entry_id = deleted_entry.entry_id
        split_body_ids = [
            e.body_id for e in entries
            if e.params.get("source_entry_id") == entry_id
        ]
        split_indices = sorted(
            [i for i, e in enumerate(entries)
             if e.params.get("source_entry_id") == entry_id],
            reverse=True
        )

        for i in split_indices:
            self.history.delete(i)
        for bid in split_body_ids + child_body_ids:
            self.workspace.remove_body(bid)
        self.history.delete(index)

        # If the primary body now has no history entries at all, it has no
        # source of geometry — remove it so it doesn't linger as a ghost.
        if not any(e.body_id == body_id for e in self.history.entries):
            self.workspace.remove_body(body_id)

        # Replay all surviving body chains in a single ordered pass so that
        # cross-body ops see up-to-date dependency shapes.
        self.history.replay_all_from(index)

        # Full rebuild — equivalent to clicking the last history entry — so any
        # ghost meshes for removed bodies are guaranteed to be flushed.
        self._rebuild_all_meshes()
        self.history_changed.emit()

    def _do_reorder(self, src: int, dst: int):
        """Raw reorder — called by ReorderCommand.apply()/revert() and direct callers."""
        entries = self.history.entries
        if src == dst or src < 0 or src >= len(entries):
            return
        print(f"[History] Reordering entry {src} → {dst}")
        self.history.reorder(src, dst)

        lo = min(src, dst)
        _, _, all_mutated = self.history.replay_all_from(lo)

        self._rebuild_bodies(all_mutated)
        self.history_changed.emit()

    # ------------------------------------------------------------------
    # Public command-stack entry points (used by history panel / keybinds)
    # ------------------------------------------------------------------

    def do_delete(self, index: int):
        """Delete with undo support via command stack."""
        import copy
        entries = self.history.entries
        if index < 0 or index >= len(entries):
            return

        # Snapshot the primary entry and any split/child entries before deletion.
        entry_id = entries[index].entry_id
        split_indices = sorted(
            [i for i, e in enumerate(entries)
             if e.params.get("source_entry_id") == entry_id]
        )
        all_indices = [index] + split_indices
        entries_snapshot = [(i, copy.deepcopy(entries[i])) for i in all_indices]

        # Snapshot bodies that will be removed.
        child_body_ids = list(entries[index].params.get("child_body_ids", []))
        split_body_ids = [entries[i].body_id for i in split_indices]
        primary_body_id = entries[index].body_id
        # If this is the only entry for the primary body, deleting it will also
        # remove the body itself — snapshot it so undo can restore it.
        primary_will_be_removed = not any(
            e.body_id == primary_body_id
            for i, e in enumerate(entries)
            if i != index and i not in split_indices
        )
        bodies_to_snapshot = split_body_ids + child_body_ids
        if primary_will_be_removed:
            bodies_to_snapshot = [primary_body_id] + bodies_to_snapshot
        removed_bodies = [
            (bid, copy.deepcopy(self.workspace.bodies[bid]))
            for bid in bodies_to_snapshot
            if bid in self.workspace.bodies
        ]

        from viewer.history_commands import DeleteCommand
        cmd = DeleteCommand(index, entries_snapshot, removed_bodies)
        # Apply directly (skip push so we don't double-apply).
        self._cmd_stack._do_push_no_apply(cmd)
        self._do_delete(index)

    def do_reorder(self, src: int, dst: int):
        """Reorder with undo support via command stack."""
        entries = self.history.entries
        if src == dst or src < 0 or src >= len(entries):
            return
        from viewer.history_commands import ReorderCommand
        cmd = ReorderCommand(src, dst)
        self._cmd_stack._do_push_no_apply(cmd)
        self._do_reorder(src, dst)
