"""
viewer/history_commands.py

Reversible command objects for structural history mutations (reorder, delete).
These sit on a separate command stack — distinct from the History cursor which
handles parametric undo of individual CAD operations.

Usage (on the viewport mixin):
    self._cmd_stack: list[HistoryCommand]  — committed commands
    self._cmd_cursor: int                  — points at last applied command
                                             (-1 = nothing applied)

Ctrl+Z outside sketch mode pops and reverses the top command.
Ctrl+Y re-applies the next command.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass   # avoid circular import; viewport passed at call time


class HistoryCommand:
    def apply(self, vp) -> None:   raise NotImplementedError
    def revert(self, vp) -> None:  raise NotImplementedError
    def describe(self) -> str:     return ""


# ---------------------------------------------------------------------------
# ReorderCommand
# ---------------------------------------------------------------------------

class ReorderCommand(HistoryCommand):
    def __init__(self, src: int, dst: int):
        self.src = src
        self.dst = dst

    def apply(self, vp) -> None:
        vp._do_reorder(self.src, self.dst)

    def revert(self, vp) -> None:
        # After apply, the entry that was at src is now at dst.
        # Reorder maps differently depending on direction; compute the
        # inverse positions after the original move.
        src, dst = self.src, self.dst
        if src < dst:
            # Entry moved from src to dst; it's now at dst, others shifted left.
            # To undo: move from dst back to src.
            vp._do_reorder(dst, src)
        else:
            # Entry moved from src to dst (dst < src); it's now at dst.
            # To undo: move from dst back to src.
            vp._do_reorder(dst, src)

    def describe(self) -> str:
        return f"Reorder {self.src} → {self.dst}"


# ---------------------------------------------------------------------------
# DeleteCommand
# ---------------------------------------------------------------------------

@dataclass
class _DeletedEntry:
    """Everything needed to re-insert a deleted history entry."""
    index:          int
    entry:          object          # HistoryEntry (deep-copied)
    body_snapshot:  object | None   # Body (deep-copied), if it was removed


class DeleteCommand(HistoryCommand):
    def __init__(self, index: int, entries_snapshot: list,
                 removed_bodies: list, cursor: int | None = None):
        """
        index             : position of the primary entry being deleted
        entries_snapshot  : deep-copy of the relevant entries (primary +
                            split/child entries) before deletion
        removed_bodies    : list of (body_id, Body) tuples that were removed
        cursor            : history cursor before the delete, restored on
                            revert (else the restored entries sit in the future)
        """
        self.index = index
        self.entries_snapshot = entries_snapshot   # [(original_index, entry), ...]
        self.removed_bodies = removed_bodies       # [(body_id, Body), ...]
        self.cursor = cursor

    def apply(self, vp) -> None:
        vp._do_delete(self.index)

    def revert(self, vp) -> None:
        # Re-insert removed bodies first so history entries can reference them.
        for body_id, body in self.removed_bodies:
            if body_id not in vp.workspace.bodies:
                vp.workspace.bodies[body_id] = body

        # Re-insert entries at their original positions (sorted ascending so
        # earlier inserts don't shift later indices).
        for orig_idx, entry in sorted(self.entries_snapshot, key=lambda x: x[0]):
            entries = vp.history._entries
            insert_at = min(orig_idx, len(entries))
            entries.insert(insert_at, entry)

        if self.cursor is not None:
            vp.history.seek(self.cursor)

        # Replay from the earliest restored position to recompute shapes.
        if self.entries_snapshot:
            lo = min(idx for idx, _ in self.entries_snapshot)
            _, _, all_mutated = vp.history.replay_all_from(lo)
            vp._rebuild_bodies(all_mutated)
        else:
            vp._rebuild_all_meshes()

        vp.history_changed.emit()

    def describe(self) -> str:
        if self.entries_snapshot:
            return f"Delete '{self.entries_snapshot[0][1].label}'"
        return "Delete"


# ---------------------------------------------------------------------------
# ParamCommand  — add / edit / rename / remove a global parameter
# ---------------------------------------------------------------------------

class ParamCommand(HistoryCommand):
    """Snapshot of the parameter table before/after an edit.  A rename also
    rewrote expressions in history entries, so it's re-applied/undone via
    History.rename_param before the table swap."""

    def __init__(self, before: list, after: list,
                 rename: tuple[str, str] | None = None, what: str = ""):
        self.before = before
        self.after  = after
        self.rename = rename
        self.what   = what

    def _swap(self, vp, items, rename) -> None:
        if rename:
            vp.history.rename_param(*rename)
        _, _, mutated = vp.history.replace_params(items)
        vp._rebuild_bodies(mutated)
        vp.history_changed.emit()

    def apply(self, vp) -> None:
        self._swap(vp, self.after, self.rename)

    def revert(self, vp) -> None:
        self._swap(vp, self.before,
                   (self.rename[1], self.rename[0]) if self.rename else None)

    def describe(self) -> str:
        return self.what or "Edit parameter"


# ---------------------------------------------------------------------------
# CommandStack  — thin wrapper used by the viewport mixin
# ---------------------------------------------------------------------------

class CommandStack:
    def __init__(self):
        self._stack: list[HistoryCommand] = []
        self._cursor: int = -1   # index of last applied command

    @property
    def can_undo(self) -> bool:
        return self._cursor >= 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._stack) - 1

    def push(self, cmd: HistoryCommand, vp) -> None:
        """Apply cmd and push onto the stack, discarding any redo branch."""
        del self._stack[self._cursor + 1:]
        cmd.apply(vp)
        self._stack.append(cmd)
        self._cursor = len(self._stack) - 1

    def _do_push_no_apply(self, cmd: HistoryCommand) -> None:
        """Record cmd as already applied — caller does the work itself."""
        del self._stack[self._cursor + 1:]
        self._stack.append(cmd)
        self._cursor = len(self._stack) - 1

    def undo(self, vp) -> str | None:
        """Revert top command. Returns description or None if nothing to undo."""
        if not self.can_undo:
            return None
        cmd = self._stack[self._cursor]
        cmd.revert(vp)
        self._cursor -= 1
        return cmd.describe()

    def redo(self, vp) -> str | None:
        """Re-apply next command. Returns description or None if nothing to redo."""
        if not self.can_redo:
            return None
        self._cursor += 1
        cmd = self._stack[self._cursor]
        cmd.apply(vp)
        return cmd.describe()

    def clear(self) -> None:
        self._stack.clear()
        self._cursor = -1
