"""
cad/workspace.py

Workspace — the top-level container for a CAD session.

Structure
---------
Workspace
  ├── bodies: dict[str, Body]   — all bodies, keyed by stable uuid
  └── history: History          — single flat timeline, entries tagged body_id

Body
  ├── id: str                   — stable uuid (survives rename)
  ├── name: str                 — user-visible, renameable
  ├── visible: bool
  └── source_shape              — the original imported solid (never mutated)

The current shape of a body is derived by replaying all history entries
tagged with that body_id up to the cursor. Workspace.current_shape(body_id)
does this efficiently by returning the shape_after of the last entry for
that body at or before the cursor.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Body:
    id:              str
    name:            str
    visible:         bool = True
    source_shape:    Any  = None   # original solid, set on import
    created_at_entry_id:  str | None = None  # entry_id of the op that created this body; None = always existed
    consumed_at_entry_id: str | None = None  # entry_id of the op that consumed this body (e.g. boolean union);
                                              # body is invisible when cursor >= this entry's index


class Workspace:
    def __init__(self):
        self.bodies:  dict[str, Body] = {}
        self.history: "History | None" = None   # set after History is constructed
        # The active body — operations target this one
        self._active_body_id: str | None = None
        # World-plane visibility — toggled from the parts panel
        self.world_plane_visible: dict[str, bool] = {
            "XY": True,
            "XZ": False,
            "YZ": False,
        }

    # ------------------------------------------------------------------
    # Body management
    # ------------------------------------------------------------------

    def add_body(self, name: str, source_shape, body_id: str | None = None) -> Body:
        """
        Create a Body and register it in the workspace. If body_id is given,
        the body adopts that exact uuid — used by the edit path to preserve
        identity across re-commits so downstream ops keep their source refs.
        """
        body = Body(
            id           = body_id if body_id else uuid.uuid4().hex,
            name         = name,
            source_shape = source_shape,
        )
        self.bodies[body.id] = body
        if self._active_body_id is None:
            self._active_body_id = body.id
        return body

    def remove_body(self, body_id: str):
        self.bodies.pop(body_id, None)
        if self._active_body_id == body_id:
            self._active_body_id = next(iter(self.bodies), None)

    @property
    def active_body(self) -> Body | None:
        return self.bodies.get(self._active_body_id)

    @property
    def active_body_id(self) -> str | None:
        return self._active_body_id

    def set_active_body(self, body_id: str):
        if body_id in self.bodies:
            self._active_body_id = body_id

    # ------------------------------------------------------------------
    # Shape queries
    # ------------------------------------------------------------------

    def current_shape(self, body_id: str) -> Any | None:
        """
        Return the current shape of a body, honoring temporal/parametric
        validity strictly:

          - If the body's creation entry is at or after the cursor, the body
            doesn't exist yet → return None.
          - If the body's creation entry is errored, the body has no valid
            origin → return None.
          - Otherwise look up the last history entry for this body at or
            before the cursor.  If it errored or has no shape_after, return
            None (the body is logically invisible until the error is fixed).
          - If the body has no entries of its own (e.g. STEP-imported bodies
            with only an import entry that doesn't push a non-None
            shape_after), fall back to source_shape.

        Mesh rebuild + viewport rendering use this single function as the
        source of truth, so an errored op anywhere in the relevant chain
        causes the body to disappear from the viewport entirely.
        """
        if body_id not in self.bodies:
            return None
        body = self.bodies[body_id]

        if self.history is None:
            return body.source_shape

        entries = self.history.entries
        cursor  = self.history.cursor

        # Creation entry must be visible (at-or-before cursor) AND not errored.
        if body.created_at_entry_id is not None:
            created_idx = self.history.id_to_index(body.created_at_entry_id)
            if created_idx is None or created_idx > cursor:
                return None
            if entries[created_idx].error:
                return None

        # If consumed by a later op (e.g. boolean union), body becomes invisible
        # once cursor reaches the consuming op. Stepping back makes it reappear.
        if body.consumed_at_entry_id is not None:
            consumed_idx = self.history.id_to_index(body.consumed_at_entry_id)
            if consumed_idx is not None and cursor >= consumed_idx:
                # Consumption only takes effect if the consuming op itself succeeded
                if not entries[consumed_idx].error:
                    return None

        # Find the last entry for this body at-or-before the cursor.
        last_entry = None
        for i, entry in enumerate(entries):
            if i > cursor:
                break
            if entry.body_id != body_id:
                continue
            last_entry = entry

        if last_entry is not None:
            if last_entry.error:
                return None
            return last_entry.shape_after   # may legitimately be a Compound

        # No history entries for this body — fall back to source_shape.
        return body.source_shape

    def all_current_shapes(self) -> list[tuple[str, Any]]:
        """Return [(body_id, current_shape)] for all visible bodies."""
        result = []
        for body_id, body in self.bodies.items():
            if not body.visible:
                continue
            shape = self.current_shape(body_id)
            if shape is not None:
                result.append((body_id, shape))
        return result
