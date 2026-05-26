"""
viewer/vp_boolean.py

BooleanMixin — panel lifecycle, body pick routing, and op dispatch for the
boolean union / subtract / intersect operation.

Union / Intersect : multi-body list (pick N bodies), fuse/intersect them all.
Subtract          : Target (single) + Tools (pick multiple).

Preview computes the boolean live and swaps the result in for the first body.
"""

from __future__ import annotations


class BooleanMixin:

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _try_boolean(self):
        """Open the boolean panel.  If a single face/body is selected, use it
        as the default target for subtract mode."""
        faces = self.selection.faces
        if faces:
            body_id = faces[0].body_id
            self._show_boolean_panel(
                target_body_id=body_id, keep_inputs=False)
        else:
            self._show_boolean_panel(keep_inputs=False)

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _show_boolean_panel(self, target_body_id: str | None = None,
                             keep_inputs: bool = False,
                             body_ids: list | None = None,
                             tool_ids: list | None = None,
                             operation: str = "union",
                             editing_entry=None):
        from gui.boolean_panel import BooleanPanel

        if getattr(self, '_boolean_panel', None) is not None:
            old = self._boolean_panel
            try:
                old.preview_changed.disconnect(self._update_boolean_preview)
                old.picking_changed.disconnect(self._on_boolean_pick_changed)
            except Exception:
                pass
            old.close()
            self._boolean_panel = None

        self._boolean_body_ids = list(body_ids or [])
        self._boolean_tool_ids = list(tool_ids or [])
        if target_body_id and not self._boolean_body_ids:
            self._boolean_body_ids = [target_body_id]
        self._boolean_operation = operation
        self._boolean_keep_inputs = keep_inputs
        self._boolean_preview_mesh = None
        self._editing_boolean_idx = (
            self.history.entries.index(editing_entry)
            if editing_entry is not None else None)

        panel = BooleanPanel(self.workspace, parent=self)
        panel.confirmed.connect(self._on_boolean_ok)
        panel.cancelled.connect(self._close_boolean_panel)
        panel.picking_changed.connect(self._on_boolean_pick_changed)
        panel.preview_changed.connect(self._update_boolean_preview)

        # Restore state
        panel.set_operation(operation)
        panel.set_keep_inputs(keep_inputs)

        # Populate bodies list
        if body_ids:
            for bid in body_ids:
                body = self.workspace.bodies.get(bid)
                if body:
                    panel.add_body_entry(bid, body.name)
        elif target_body_id and target_body_id in self.workspace.bodies:
            body = self.workspace.bodies[target_body_id]
            panel.add_body_entry(target_body_id, body.name)

        # Populate tools list (subtract mode)
        if tool_ids:
            for bid in tool_ids:
                body = self.workspace.bodies.get(bid)
                if body:
                    panel.add_tool_entry(bid, body.name)

        self._boolean_panel = panel
        self._position_boolean_panel()
        panel.show()
        panel.setFocus()

    def _position_boolean_panel(self):
        p = getattr(self, '_boolean_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_boolean_panel(self):
        panel = getattr(self, '_boolean_panel', None)
        if panel is not None:
            try:
                panel.preview_changed.disconnect(self._update_boolean_preview)
                panel.picking_changed.disconnect(self._on_boolean_pick_changed)
            except Exception:
                pass
            panel.end_pick()
            panel.end_pick_tool()
            panel.close()
            self._boolean_panel = None

        if getattr(self, '_editing_boolean_idx', None) is not None:
            self._cancel_boolean_edit()

        self._boolean_body_ids = []
        self._boolean_tool_ids = []
        self._boolean_operation = None
        self._boolean_keep_inputs = False
        self._boolean_preview_mesh = None
        self._boolean_pick_active = False
        self._boolean_pick_target = False
        self._boolean_pick_tool = False
        self.update()

    # ------------------------------------------------------------------
    # Pick routing
    # ------------------------------------------------------------------

    def _on_boolean_pick_changed(self, active: bool):
        """The panel doesn't distinguish target vs tool in the signal — we
        infer it from which picker button is active on the panel."""
        self._boolean_pick_active = active

    def route_body_pick_for_boolean(self, body_id: str) -> bool:
        """Route a body click to the boolean panel's current picker."""
        panel = getattr(self, '_boolean_panel', None)
        if panel is None or not getattr(self, '_boolean_pick_active', False):
            return False
        body = self.workspace.bodies.get(body_id)
        if body is None:
            return False

        op = panel._get_operation()
        if op == "subtract":
            if panel._pick_btn.isChecked():
                # Target — single, replaces existing
                panel._target_list.clear()
                self._boolean_body_ids = [body_id]
                panel.add_body_entry(body_id, body.name)
                panel.end_pick()
            elif panel._pick_tool_btn.isChecked():
                # Tools — toggle add/remove
                existing = panel._tool_list.keys
                if body_id in existing:
                    self._boolean_tool_ids = [b for b in self._boolean_tool_ids if b != body_id]
                    panel.remove_tool_entry(existing.index(body_id))
                else:
                    self._boolean_tool_ids.append(body_id)
                    panel.add_tool_entry(body_id, body.name)
        else:
            # Union / intersect — toggle add/remove in bodies list
            existing = panel._target_list.keys
            if body_id in existing:
                self._boolean_body_ids = [b for b in self._boolean_body_ids if b != body_id]
                panel.remove_body_entry(existing.index(body_id))
            else:
                self._boolean_body_ids.append(body_id)
                panel.add_body_entry(body_id, body.name)
        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_boolean_preview(self):
        """Compute a translucent boolean preview from current panel state."""
        panel = getattr(self, '_boolean_panel', None)
        if panel is None:
            self._boolean_preview_mesh = None
            self.update()
            return

        op = panel._get_operation()
        body_ids = panel._target_list.keys
        if op == "subtract":
            tool_ids = panel._tool_list.keys
            all_ids = body_ids + tool_ids
        else:
            all_ids = body_ids

        if len(all_ids) < 2:
            self._boolean_preview_mesh = None
            self.update()
            return

        shapes = []
        for bid in all_ids:
            s = self.workspace.current_shape(bid)
            if s is None:
                self._boolean_preview_mesh = None
                self.update()
                return
            shapes.append(s)

        try:
            from OCP.BRepAlgoAPI import (
                BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common,
            )
            from OCP.TopTools import TopTools_ListOfShape
            from build123d import Compound

            lst_a = TopTools_ListOfShape()
            lst_a.Append(shapes[0].wrapped)
            lst_b = TopTools_ListOfShape()
            for s in shapes[1:]:
                lst_b.Append(s.wrapped)

            if op == "union":
                bool_op = BRepAlgoAPI_Fuse()
            elif op == "subtract":
                bool_op = BRepAlgoAPI_Cut()
            elif op == "intersect":
                bool_op = BRepAlgoAPI_Common()
            else:
                self._boolean_preview_mesh = None
                self.update()
                return

            bool_op.SetArguments(lst_a)
            bool_op.SetTools(lst_b)
            bool_op.SetRunParallel(True)
            bool_op.Build()

            if not bool_op.IsDone():
                self._boolean_preview_mesh = None
                self.update()
                return

            result = Compound(bool_op.Shape())
            from viewer.mesh import Mesh
            preview_mesh = Mesh(result)
            self.makeCurrent()
            try:
                preview_mesh.upload()
            except Exception:
                preview_mesh = None
            self._boolean_preview_mesh = preview_mesh
        except Exception:
            self._boolean_preview_mesh = None

        self.update()

    def _draw_boolean_preview(self):
        """No-op — the preview solid is drawn via _visible_meshes swap
        in the main renderer.  This placeholder avoids a missing-method crash."""
        pass

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def _on_boolean_ok(self, body_ids: list, operation: str, keep_inputs: bool):
        editing_idx = getattr(self, '_editing_boolean_idx', None)
        if editing_idx is not None:
            self._editing_boolean_idx = None

        self._close_boolean_panel()

        if len(body_ids) < 2:
            print("[Boolean] Need at least 2 bodies.")
            return

        from cad.op_types import BooleanOp
        op = BooleanOp(
            body_ids=body_ids,
            operation=operation,
            keep_inputs=keep_inputs,
        )

        if editing_idx is not None:
            self._commit_boolean_edit(editing_idx, op)
            return

        op.commit_async(self)

    # ------------------------------------------------------------------
    # Edit / reopen
    # ------------------------------------------------------------------

    def reopen_boolean(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation not in ("union", "subtract", "intersect") or entry.op is None:
            return
        op = entry.op
        entry.editing = True
        self._editing_boolean_idx = history_idx

        # Seek to pre-op state — current_shape will reveal the input bodies
        # automatically because cursor < their consumed_at_entry_id.
        if history_idx > 0:
            self.history.seek(history_idx - 1)
        self._rebuild_all_meshes()
        self.history_changed.emit()

        # For subtract mode, split body_ids into target + tools
        if op.operation == "subtract":
            target_id = op.body_ids[0]
            tool_ids = op.body_ids[1:]
            self._show_boolean_panel(
                body_ids=[target_id],
                tool_ids=tool_ids,
                operation=op.operation,
                keep_inputs=getattr(op, 'keep_inputs', False),
                editing_entry=entry)
        else:
            self._show_boolean_panel(
                body_ids=list(op.body_ids),
                operation=op.operation,
                keep_inputs=getattr(op, 'keep_inputs', False),
                editing_entry=entry)

    def _commit_boolean_edit(self, editing_idx: int, new_op):
        """Delete the old boolean entry group, then commit the new op."""
        entries = self.history.entries
        if editing_idx >= len(entries):
            return
        entry = entries[editing_idx]
        entry.editing = False
        entry_id = entry.entry_id

        # Collect the entry group + any split-children.
        group_indices = [editing_idx]
        group_body_ids = {entry.body_id}
        for j in range(editing_idx + 1, len(entries)):
            if entries[j].params.get("source_entry_id") == entry_id:
                group_indices.append(j)
                group_body_ids.add(entries[j].body_id)

        self.history.seek(max(editing_idx - 1, 0))

        source_id = new_op.source_body_id
        if self.workspace.current_shape(source_id) is None:
            print(f"[Boolean edit] Source body '{source_id}' has no shape here.")
            self.history.seek(editing_idx)
            entries[editing_idx].editing = False
            self._rebuild_body_mesh(source_id)
            self.history_changed.emit()
            return

        # Delete old group. Clear consumed_at_entry_id on any body consumed by
        # an entry we're about to delete (the new commit will re-consume them).
        group_entry_ids = {entries[j].entry_id for j in group_indices}
        removable = set()
        for bid in group_body_ids:
            body = self.workspace.bodies.get(bid)
            if body is not None and body.created_at_entry_id in group_entry_ids:
                removable.add(bid)
        for body in self.workspace.bodies.values():
            if body.consumed_at_entry_id in group_entry_ids:
                body.consumed_at_entry_id = None

        # Preserve the result body id ONLY when the old op also created a new
        # body for its result (union/intersect). Subtract's "result body" is
        # its target (an existing input body) — reusing that id for a new
        # union would overwrite the input in workspace.bodies and break things.
        old_op_str = entry.operation
        if old_op_str in ("union", "intersect"):
            new_op.result_body_id = entry.body_id

        for j in reversed(group_indices):
            self.history.delete(j)
        for bid in removable:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

        new_op.commit_async(self)

    def _cancel_boolean_edit(self):
        idx = getattr(self, '_editing_boolean_idx', None)
        self._editing_boolean_idx = None
        if idx is None:
            return
        entries = self.history.entries
        if idx < len(entries):
            entries[idx].editing = False
        self.history.seek(idx)
        # current_shape will re-hide the consumed inputs automatically because
        # cursor is back at-or-after the boolean entry.
        self._rebuild_all_meshes()
        self.history_changed.emit()
