"""
viewer/vp_loft.py

LoftMixin — panel lifecycle, sketch-face / body pick routing, and op
dispatch for the loft / loft-cut operation.

Expects self to have: history, workspace, _meshes, _sketch_faces,
selection, _rebuild_body_mesh(), _rebuild_bodies(), _post_push_cascade(),
history_changed signal.
"""

from __future__ import annotations


class LoftMixin:

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _show_loft_panel(self):
        from gui.loft_panel import LoftPanel
        if getattr(self, '_loft_panel', None) is not None:
            return
        panel = LoftPanel(self.workspace, parent=self)
        panel.loft_requested.connect(self._on_loft_panel_ok)
        panel.cancelled.connect(self._close_loft_panel)
        panel.picking_profile_changed.connect(self._on_loft_pick_profile)
        panel.picking_body_changed.connect(self._on_loft_pick_body)
        panel.preview_changed.connect(self._on_loft_preview_live)
        self._loft_panel               = panel
        self._loft_profile_pick_active = False
        self._loft_body_pick_active    = False
        self._loft_preview_mesh        = None   # Solid | None
        self._loft_preview_is_cut      = False
        self._position_loft_panel()
        panel.show()
        panel.setFocus()

    def _position_loft_panel(self):
        p = getattr(self, '_loft_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_loft_panel(self):
        p = getattr(self, '_loft_panel', None)
        if p is not None:
            p.close()
        self._loft_panel               = None
        self._loft_profile_pick_active = False
        self._loft_body_pick_active    = False
        self._loft_preview_mesh        = None
        self._loft_preview_is_cut      = False
        # If the panel was closed without an OK (e.g. Cancel/Esc) while editing,
        # restore the original entry so the user doesn't lose their loft.
        idx = getattr(self, '_editing_loft_idx', None)
        if idx is not None:
            self._editing_loft_idx = None
            entries = self.history.entries
            if 0 <= idx < len(entries):
                entries[idx].editing = False
                self.history.seek(idx)
                self._rebuild_all_meshes()
            self.history_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Reopen / edit
    # ------------------------------------------------------------------

    def reopen_loft(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation not in ("loft", "loft_cut") or entry.op is None:
            return
        entry.editing = True
        self._editing_loft_idx = history_idx
        op = entry.op

        # Seek to just before the op so the panel shows pre-op state.
        if history_idx > 0:
            self.history.seek(history_idx - 1)
            self._rebuild_all_meshes()
        self.history_changed.emit()

        self._show_loft_panel()
        panel = self._loft_panel
        if panel is None:
            return

        # Populate profile list
        for sid, face_idx in op.from_profiles:
            sidx = self.history.id_to_index(sid)
            if sidx is None:
                continue
            label = (f"Sketch {sidx}" if face_idx is None
                     else f"Sketch {sidx}  ·  face {face_idx}")
            panel.add_profile(sid, label, face_idx=face_idx)

        # Restore mode + target body
        if op.cut_body_id is not None:
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)
            if op.cut_body_id in self.workspace.bodies:
                panel.set_picked_body(op.cut_body_id,
                                       self.workspace.bodies[op.cut_body_id].name)
        elif entry.operation == "loft_cut":
            # Edit of a cut-thru-all: no specific target to restore.
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)
        elif op.merge_body_id is not None and not op.force_new_body:
            panel._radio_merge.setChecked(True)
            panel._on_op_changed(1)
            if op.merge_body_id in self.workspace.bodies:
                panel.set_picked_body(op.merge_body_id,
                                       self.workspace.bodies[op.merge_body_id].name)

        # Restore surface params
        panel.set_ruled(op.ruled)
        panel.set_continuity(op.continuity)

    def _tear_down_loft_entry_group(self, idx: int):
        """Delete the loft entry at idx plus its split-body children, and
        remove any bodies the original op created."""
        entries = self.history.entries
        if idx >= len(entries):
            return
        entry = entries[idx]
        entry.editing = False

        group_indices = [idx]
        entry_id = entry.entry_id
        for j in range(idx + 1, len(entries)):
            e = entries[j]
            if e.params.get("source_entry_id") == entry_id:
                group_indices.append(j)

        child_body_ids = entry.params.get("child_body_ids", []) or []
        group_entry_ids = {entries[j].entry_id for j in group_indices}
        removable_bodies = set()
        candidate_body_ids = set(child_body_ids)
        if entry.body_id:
            candidate_body_ids.add(entry.body_id)
        for bid in candidate_body_ids:
            body = self.workspace.bodies.get(bid)
            if body is not None and body.created_at_entry_id in group_entry_ids:
                removable_bodies.add(bid)

        # Collect preserved ids (in order) so the new commit reuses the same
        # body uuids — keeps downstream boolean ops pointing at the right bodies
        # even after parametric edits. Include absent bodies (consumed downstream).
        all_produced = ([entry.body_id] if entry.body_id else []) + list(child_body_ids)
        preserved_body_ids = []
        for bid in all_produced:
            if bid and bid not in preserved_body_ids:
                body = self.workspace.bodies.get(bid)
                if body is None or body.created_at_entry_id in group_entry_ids:
                    preserved_body_ids.append(bid)

        # Seek to idx-1 so the rebuild after delete is consistent.
        self.history.seek(max(idx - 1, 0))
        for j in reversed(group_indices):
            self.history.delete(j)
        for bid in removable_bodies:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

        return preserved_body_ids

    def _commit_loft_edit(self, editing_idx: int, new_op):
        """Delete the old loft entry group, then commit new_op fresh and replay
        downstream entries."""
        preserved_body_ids = self._tear_down_loft_entry_group(editing_idx)
        new_op.commit(self, {"_preserved_body_ids": preserved_body_ids} if preserved_body_ids else None)
        new_idx = self.history.cursor
        ok, err, _ = self.history.replay_from(new_idx + 1)
        if not ok:
            print(f"[Edit] Downstream replay failed: {err}")
        # Advance cursor to tip so bodies created downstream are visible.
        self.history.seek(len(self.history.entries) - 1)
        self._rebuild_all_meshes()
        self.history_changed.emit()

    # ------------------------------------------------------------------
    # Pick routing
    # ------------------------------------------------------------------

    def _on_loft_pick_profile(self, active: bool):
        self._loft_profile_pick_active = active

    def _on_loft_pick_body(self, active: bool):
        self._loft_body_pick_active = active

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _draw_loft_preview(self):
        """Render the loft preview solid as a translucent surface with edge
        outlines.  Color: prefs.op_preview_color for add/merge, red for cut."""
        solid = getattr(self, '_loft_preview_mesh', None)
        if solid is None:
            return

        from OpenGL.GL import (
            glDisable, glEnable, glColor4f, glLineWidth, glBegin, glEnd,
            glVertex3f, glBlendFunc,
            GL_LIGHTING, GL_BLEND, GL_DEPTH_TEST, GL_CULL_FACE,
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            GL_TRIANGLES, GL_LINE_STRIP,
        )
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCP.TopoDS import TopoDS
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GCPnts import GCPnts_UniformAbscissa
        from cad.prefs import prefs as _prefs

        is_cut = getattr(self, '_loft_preview_is_cut', False)
        op = _prefs.op_preview_opacity
        if is_cut:
            fill_color = (0.75, 0.18, 0.18, op)
            edge_color = (1.00, 0.35, 0.35, min(op + 0.35, 1.0))
        else:
            r, g, b = _prefs.op_preview_color
            fill_color = (r, g, b, op)
            edge_color = (min(r + 0.23, 1.0), min(g + 0.30, 1.0),
                          min(b + 0.15, 1.0), min(op + 0.35, 1.0))

        glDisable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        try:
            wrapped = solid.wrapped
            BRepMesh_IncrementalMesh(wrapped, 0.15)

            glColor4f(*fill_color)
            exp = TopExp_Explorer(wrapped, TopAbs_FACE)
            while exp.More():
                face = TopoDS.Face_s(exp.Current())
                loc  = face.Location()
                tri  = BRep_Tool.Triangulation_s(face, loc)
                if tri is not None:
                    has_trsf = not loc.IsIdentity()
                    trsf = loc.Transformation() if has_trsf else None
                    glBegin(GL_TRIANGLES)
                    for i in range(1, tri.NbTriangles() + 1):
                        n1, n2, n3 = tri.Triangle(i).Get()
                        for ni in (n1, n2, n3):
                            p = tri.Node(ni)
                            if trsf is not None:
                                p = p.Transformed(trsf)
                            glVertex3f(p.X(), p.Y(), p.Z())
                    glEnd()
                exp.Next()

            glColor4f(*edge_color)
            glLineWidth(1.4)
            exp2 = TopExp_Explorer(wrapped, TopAbs_EDGE)
            while exp2.More():
                edge = exp2.Current()
                try:
                    adaptor = BRepAdaptor_Curve(edge)
                    disc    = GCPnts_UniformAbscissa()
                    disc.Initialize(adaptor, 24)
                    if disc.IsDone() and disc.NbPoints() >= 2:
                        glBegin(GL_LINE_STRIP)
                        for pi in range(1, disc.NbPoints() + 1):
                            p = adaptor.Value(disc.Parameter(pi))
                            glVertex3f(p.X(), p.Y(), p.Z())
                        glEnd()
                except Exception:
                    pass
                exp2.Next()
            glLineWidth(1.0)
        except Exception as ex:
            print(f"[Loft preview] draw error: {ex}")

        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)

    def _on_loft_preview_live(self):
        """Recompute the loft preview solid from current panel state.

        Silent on failure (panel may be in a transient invalid state, e.g.
        only one profile picked); just clears the preview so nothing draws.
        """
        from cad.op_types import SketchLoftOp
        panel = getattr(self, '_loft_panel', None)
        if panel is None:
            return
        profiles = panel.profiles()
        if len(profiles) < 2:
            if self._loft_preview_mesh is not None:
                self._loft_preview_mesh = None
                self.update()
            return
        probe = SketchLoftOp(
            from_profiles = list(profiles),
            ruled         = panel.ruled(),
            continuity    = panel.continuity(),
        )
        try:
            faces = probe._resolve_profile_faces(
                self.history, self.history.cursor + 1)
            solid = probe._build_loft(faces)
        except Exception:
            self._loft_preview_mesh = None
            self.update()
            return
        self._loft_preview_mesh   = solid
        self._loft_preview_is_cut = (panel._mode_str() == "cut")
        self.update()

    def route_sketch_face_pick_for_loft(self, sketch_idx: int,
                                         face_idx: int) -> bool:
        """Add (or toggle off) a sketch face as a loft profile.

        Multi-loop sketches: each face is its own profile key (sid, face_idx).
        Single-loop sketches: also keyed by face_idx for consistency — the
        face_idx is virtually always 0 and there's no ambiguity.
        """
        if not getattr(self, '_loft_profile_pick_active', False):
            return False
        panel = getattr(self, '_loft_panel', None)
        if panel is None:
            return False
        entries = self.history.entries
        if sketch_idx < 0 or sketch_idx >= len(entries):
            return False
        entry = entries[sketch_idx]
        if entry.operation != "sketch":
            return False
        sid = entry.entry_id
        face_count = len(self._sketch_faces.get(sketch_idx, []))
        # Single-loop sketches: store face_idx=None so saves remain compact.
        key_face_idx = None if face_count <= 1 else face_idx
        label = (f"Sketch {sketch_idx}" if key_face_idx is None
                 else f"Sketch {sketch_idx}  ·  face {face_idx}")
        existing = panel.profiles()
        if (sid, key_face_idx) in existing:
            panel.remove_profile(sid, key_face_idx)
        else:
            panel.add_profile(sid, label, face_idx=key_face_idx)
        return True

    def route_body_pick_for_loft(self, body_id: str) -> bool:
        if not getattr(self, '_loft_body_pick_active', False):
            return False
        body = self.workspace.bodies.get(body_id)
        if body is None:
            return False
        panel = getattr(self, '_loft_panel', None)
        if panel is None:
            return False
        panel.set_picked_body(body_id, body.name)
        return True

    # ------------------------------------------------------------------
    # OK handler
    # ------------------------------------------------------------------

    def _do_loft_cut_thru_all(self, profiles: list,
                              ruled: bool = False, continuity: str = "C1"):
        """Build the loft solid once, then cut it from every body whose bbox
        overlaps. Uses cad.cut_all.fan_out_cut so behavior matches extrude's
        no-target cut."""
        from cad.op_types import SketchLoftOp
        from cad.cut_all import fan_out_cut
        from build123d import Compound

        # Build the tool solid once, using the same resolver + builder the
        # op uses so any error message looks the same.
        probe_op = SketchLoftOp(from_profiles=list(profiles),
                                ruled=ruled, continuity=continuity)
        try:
            faces = probe_op._resolve_profile_faces(
                self.history, self.history.cursor + 1)
            tool_solid = Compound(probe_op._build_loft(faces).wrapped)
        except Exception as ex:
            print(f"[Loft cut] Tool construction failed: {ex}")
            return

        def build_op(bid: str):
            return SketchLoftOp(
                from_profiles = list(profiles),
                cut_body_id   = bid,
                ruled         = ruled,
                continuity    = continuity,
            )

        fan_out_cut(self, tool_solid, build_op, None, op_label="Loft cut")

    def _commit_loft_edit_cut_thru_all(self, editing_idx: int, profiles: list,
                                        ruled: bool = False,
                                        continuity: str = "C1"):
        """Edit path for a no-target cut: tear down the original entry group
        then run the cut-thru-all fresh.  Mirrors _commit_loft_edit."""
        self._tear_down_loft_entry_group(editing_idx)
        self._do_loft_cut_thru_all(profiles, ruled=ruled, continuity=continuity)
        # Replay any downstream entries past the new tip.
        new_idx = self.history.cursor
        ok, err, _ = self.history.replay_from(new_idx + 1)
        if not ok:
            print(f"[Edit] Downstream replay failed: {err}")
        # Advance cursor to tip so bodies created downstream are visible.
        self.history.seek(len(self.history.entries) - 1)
        self._rebuild_all_meshes()
        self.history_changed.emit()

    def _on_loft_panel_ok(self, profiles: list, mode: str, target_body_id,
                          ruled: bool, continuity: str):
        from cad.op_types import SketchLoftOp
        if len(profiles) < 2:
            return

        editing_idx = getattr(self, '_editing_loft_idx', None)
        if editing_idx is not None:
            self._editing_loft_idx = None  # clear before close so cancel doesn't fire

        self._close_loft_panel()

        # Cut with no target → fan out across every intersecting body.
        if mode == "cut" and target_body_id is None:
            if editing_idx is not None:
                self._commit_loft_edit_cut_thru_all(
                    editing_idx, profiles, ruled=ruled, continuity=continuity)
                return
            self._do_loft_cut_thru_all(profiles, ruled=ruled, continuity=continuity)
            return

        op = SketchLoftOp(
            from_profiles  = list(profiles),
            merge_body_id  = target_body_id if mode == "merge" else None,
            cut_body_id    = target_body_id if mode == "cut"   else None,
            force_new_body = (mode == "new"),
            ruled          = ruled,
            continuity     = continuity,
        )

        if editing_idx is not None:
            self._commit_loft_edit(editing_idx, op)
            return

        op.commit_async(self, None)
