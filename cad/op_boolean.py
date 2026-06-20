"""
cad/op_boolean.py

BooleanOp — standalone boolean union / subtract / intersect between bodies.

Union / Intersect : result is pushed to a brand-new body; all input bodies
                    are consumed (hidden) by default.
                    With keep_inputs=True: inputs remain visible AND the new
                    result body is added alongside them.

Subtract          : result replaces the target body (body_ids[0]); tool bodies
                    are consumed (hidden) by default.
                    With keep_inputs=True: tool bodies remain visible.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cad.op_base import Op, _push_result

if TYPE_CHECKING:
    from cad.history import History


@dataclass
class BooleanOp(Op):
    """
    Boolean operation between one or more bodies.

    body_ids       : list of body ids participating in the operation.
                     For subtract: body_ids[0] is the target, the rest are tools.
                     For union/intersect: all bodies are operands.
    operation      : "union" | "subtract" | "intersect"
    keep_inputs    : when True, do NOT hide input bodies after the operation
    result_body_id : id of the new body created for union/intersect result;
                     None for subtract (result stays on body_ids[0])
    """
    body_ids:       list        # list[str]
    operation:      str  = "union"
    keep_inputs:    bool = False
    result_body_id: str | None = None
    body_names:     dict = None  # {body_id: name} captured at commit, used on reopen

    def __post_init__(self):
        if self.body_names is None:
            self.body_names = {}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def source_body_id(self) -> str:
        """First body — for subtract this is the target (result stays here)."""
        return self.body_ids[0]

    @property
    def tool_body_ids(self) -> list[str]:
        return self.body_ids[1:]

    def creates_body_from_nothing(self, history: "History", entry_index: int) -> bool:
        # The result body for union/intersect has no prior shape chain.
        return self.operation in ("union", "intersect")

    # ------------------------------------------------------------------
    # Shared OCC compute
    # ------------------------------------------------------------------

    def _run_bool(self, wrapped_shapes: list) -> Any:
        from build123d import Compound
        from OCP.BRepAlgoAPI import (
            BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common,
        )
        from OCP.TopTools import TopTools_ListOfShape

        if self.operation == "union":
            bool_op = BRepAlgoAPI_Fuse()
        elif self.operation == "subtract":
            bool_op = BRepAlgoAPI_Cut()
        elif self.operation == "intersect":
            bool_op = BRepAlgoAPI_Common()
        else:
            raise RuntimeError(f"BooleanOp: unknown operation '{self.operation}'")

        lst_a = TopTools_ListOfShape()
        lst_b = TopTools_ListOfShape()
        lst_a.Append(wrapped_shapes[0])
        for w in wrapped_shapes[1:]:
            lst_b.Append(w)

        bool_op.SetArguments(lst_a)
        bool_op.SetTools(lst_b)
        bool_op.SetRunParallel(True)
        bool_op.Build()

        if not bool_op.IsDone():
            raise RuntimeError(f"BooleanOp: {self.operation} failed")

        return Compound(bool_op.Shape())

    # ------------------------------------------------------------------
    # Execute (replay)
    # ------------------------------------------------------------------

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        wrapped = []
        for bid in self.body_ids:
            s = history._shape_for_body_at(bid, entry_index)
            if s is None:
                raise RuntimeError(f"BooleanOp: no shape for body '{bid}'")
            wrapped.append(s.wrapped)

        # Re-assert consumption on input bodies. Edit paths can recreate input
        # bodies (with preserved ids) which resets consumed_at_entry_id; replay
        # must restore that link so the parts panel/viewport stay consistent.
        if not self.keep_inputs:
            entry = history._entries[entry_index]
            consumed_ids = (self.body_ids if self.operation in ("union", "intersect")
                            else self.tool_body_ids)
            ws = getattr(history, '_workspace', None)
            if ws is not None:
                for bid in consumed_ids:
                    body = ws.bodies.get(bid)
                    if body is not None:
                        body.consumed_at_entry_id = entry.entry_id

        return self._run_bool(wrapped)

    # ------------------------------------------------------------------
    # Commit (first run)
    # ------------------------------------------------------------------

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        compute, finalize = self._split_commit(viewport, extra_params)
        try:
            shape_after = compute()
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            shape_after = None
            viewport._pending_op_error = str(ex)
        else:
            viewport._pending_op_error = None
        try:
            finalize(shape_after)
        finally:
            viewport._pending_op_error = None
        return shape_after

    def _split_commit(self, viewport: Any, extra_params: dict | None = None):
        if len(self.body_ids) < 2:
            raise RuntimeError("[Boolean] Need at least 2 bodies")

        shapes = []
        for bid in self.body_ids:
            s = viewport.workspace.current_shape(bid)
            if s is None:
                raise RuntimeError(f"[Boolean] No shape for body {bid}")
            shapes.append(s)

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)

        wrapped_shapes = [s.wrapped for s in shapes]
        operation = self.operation

        if operation in ("union", "intersect"):
            return self._split_commit_new_body(viewport, shapes, wrapped_shapes, op_params)
        else:
            return self._split_commit_subtract(viewport, shapes, wrapped_shapes, op_params)

    def _split_commit_new_body(self, viewport, shapes, wrapped_shapes, op_params):
        """Union / intersect — result goes to a brand-new body."""
        from build123d import Compound
        from cad.units import format_op_label as _lbl

        operation = self.operation

        # Capture names before any bodies are removed, store for reopen
        body_names = {bid: viewport.workspace.bodies[bid].name
                      for bid in self.body_ids
                      if bid in viewport.workspace.bodies}
        self.body_names = body_names
        op_params["body_names"] = body_names

        # Operand names are kept in op_params["body_names"] for reopen/labels;
        # the result body itself just gets the next sequential "Part N" name so
        # repeated merges don't compound into ever-growing concatenations.
        # On the edit path the result body already exists — keep its name (and
        # don't burn a counter number) rather than renaming it on re-commit.
        existing = (viewport.workspace.bodies.get(self.result_body_id)
                    if self.result_body_id else None)
        result_name = existing.name if existing else viewport.workspace.next_part_name()

        # Reuse result_body_id if set (edit path preserves it so downstream ops
        # keep their reference), otherwise generate a new body.
        result_body = viewport.workspace.add_body(
            result_name, None, body_id=self.result_body_id or None)
        self.result_body_id = result_body.id
        op_params["result_body_id"] = result_body.id

        keep = self.keep_inputs

        def compute():
            return self._run_bool(wrapped_shapes)

        def finalize(shape_after):
            from cad.units import format_op_label as _lbl
            label = _lbl(operation, op_params)

            if shape_after is None:
                err = getattr(viewport, '_pending_op_error', None) or f"{operation} failed"
                entry = viewport.history.push(
                    label=label, operation=operation, params=op_params,
                    body_id=result_body.id, face_ref=None,
                    shape_before=None, shape_after=None)
                entry.error = True
                entry.error_msg = err
                viewport._post_push_cascade(result_body.id)
                viewport.history_changed.emit()
                return

            result_body.source_shape = shape_after
            entry = viewport.history.push(
                label=label, operation=operation, params=op_params,
                body_id=result_body.id, face_ref=None,
                shape_before=None, shape_after=shape_after)
            result_body.created_at_entry_id = entry.entry_id

            # Mark inputs as consumed at this entry (parametric: stepping back
            # before this entry makes them visible again).
            if not keep:
                for bid in self.body_ids:
                    body = viewport.workspace.bodies.get(bid)
                    if body is not None:
                        body.consumed_at_entry_id = entry.entry_id
                    viewport._meshes.pop(bid, None)

            viewport._post_push_cascade(result_body.id)
            viewport.history_changed.emit()
            viewport.update()

        return compute, finalize

    def _split_commit_subtract(self, viewport, shapes, wrapped_shapes, op_params):
        """Subtract — result replaces the target body (body_ids[0])."""
        source_shape = shapes[0]
        original_solid_count = len(list(source_shape.solids()))
        keep = self.keep_inputs
        operation = self.operation

        def compute():
            return self._run_bool(wrapped_shapes)

        def finalize(shape_after):
            _push_result(viewport, operation, op_params, self.source_body_id,
                         None, source_shape, shape_after, original_solid_count)
            if not keep and shape_after is not None:
                # Find the subtract entry we just pushed (look it up by op
                # rather than relying on cursor — _push_result may have pushed
                # additional import entries for splits, moving the cursor).
                consume_eid = None
                for e in reversed(viewport.history.entries):
                    if e.operation == operation and e.body_id == self.source_body_id:
                        consume_eid = e.entry_id
                        break
                for bid in self.tool_body_ids:
                    body = viewport.workspace.bodies.get(bid)
                    if body is not None and consume_eid is not None:
                        body.consumed_at_entry_id = consume_eid
                    viewport._meshes.pop(bid, None)
                # Re-emit history_changed so the parts panel re-filters with
                # the freshly-set consumed_at_entry_id.
                viewport.history_changed.emit()
                viewport.update()

        return compute, finalize

    # ------------------------------------------------------------------
    # Reopen (edit)
    # ------------------------------------------------------------------

    def reopen(self, viewport: Any, history_idx: int) -> None:
        viewport.reopen_boolean(history_idx)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_params(self) -> dict:
        p = {
            "body_ids":    list(self.body_ids),
            "operation":   self.operation,
            "keep_inputs": self.keep_inputs,
        }
        if self.result_body_id is not None:
            p["result_body_id"] = self.result_body_id
        if self.body_names:
            p["body_names"] = dict(self.body_names)
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "BooleanOp":
        return cls(
            body_ids       = list(params.get("body_ids", [])),
            operation      = params.get("operation", "union"),
            keep_inputs    = bool(params.get("keep_inputs", False)),
            result_body_id = params.get("result_body_id"),
            body_names     = dict(params.get("body_names", {})),
        )
