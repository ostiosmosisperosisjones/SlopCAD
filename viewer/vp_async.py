"""
viewer/vp_async.py

AsyncOpMixin — thread heavy OCC compute off the main thread.

Usage in any op handler:

    def _on_foo_ok(self, ...):
        panel_state = ...          # capture everything before closing panel
        self._close_foo_panel()

        def compute():             # runs in worker thread — pure OCC, no Qt
            return some_op(...)    # return shape_after, or raise on failure

        def finalize(result):      # runs on main thread — push history, rebuild
            _push_result(...)
            self._rebuild_body_mesh(...)
            self.history_changed.emit()

        self.run_op_async("OpName", compute, finalize)

While the op is running:
  - A "Computing…" overlay label is shown in the viewport.
  - User input is blocked (setEnabled(False)).
  - On error: the error is printed and the UI is re-enabled, no history entry.
  - If compute() raises, finalize() is never called.

The worker thread runs compute() then immediately tessellates the result into a
Mesh object (also pure CPU).  finalize() receives the shape_after; _rebuild_body_mesh
is monkey-patched for the duration of the call so it skips re-tessellating and
just does the GL upload with the pre-built mesh.
"""

from __future__ import annotations
import threading
from PyQt6.QtCore import QMetaObject, Qt, Q_ARG, pyqtSlot
from PyQt6.QtWidgets import QLabel, QApplication


class AsyncOpMixin:

    def run_op_async(self, name: str, compute, finalize):
        """
        Run compute() + tessellation in a background thread, then call
        finalize(shape_after) on the main thread with GL-upload-only rebuild.
        """
        self._async_set_busy(True, name)

        pc = getattr(self, 'progress', None)

        def _worker():
            import time
            t0 = time.time()
            try:
                if pc is not None:
                    pc.phase("computing…")
                shape_after = compute()
                if pc is not None:
                    pc.phase("tessellating…")
                mesh = _tessellate(shape_after)
                err  = None
            except Exception as ex:
                shape_after = None
                mesh        = None
                err         = ex
            finally:
                elapsed = time.time() - t0
                print(f"[{name}] {'done' if err is None else 'failed'} in {elapsed:.2f}s", flush=True)
            QMetaObject.invokeMethod(
                self, "_async_done",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(object, finalize),
                Q_ARG(object, shape_after),
                Q_ARG(object, mesh),
                Q_ARG(object, err),
            )

        threading.Thread(target=_worker, daemon=True).start()

    @pyqtSlot(object, object, object, object)
    def _async_done(self, finalize, shape_after, mesh, err):
        if err is not None:
            print(f"[Op] FAILED: {err}")
        # Always call finalize — on failure shape_after is None and _push_result
        # will push an error (red) history entry instead of silently dropping it.
        # Store the error string so _push_result can attach it to the entry.
        self._pending_op_error = str(err) if err is not None else None
        _orig = self._rebuild_body_mesh
        _used = {}

        def _fast_rebuild(body_id: str):
            if body_id in _used:
                _orig(body_id)
                return
            _used[body_id] = True
            if mesh is not None:
                _upload_mesh(self, body_id, mesh)
            else:
                _orig(body_id)

        self._rebuild_body_mesh = _fast_rebuild
        try:
            finalize(shape_after)
        finally:
            self._rebuild_body_mesh = _orig
            self._pending_op_error  = None
            self._async_set_busy(False)

    def _async_set_busy(self, busy: bool, label: str = ""):
        # Prefer the program-wide progress controller (status bar + overlay +
        # cancel). Fall back to the legacy local overlay if it isn't wired yet.
        pc = getattr(self, 'progress', None)
        if pc is not None:
            if busy:
                pc.begin(label or "Working", modal=True, cancelable=False)
                self.setEnabled(False)
            else:
                pc.end()
                self.setEnabled(True)
            return

        overlay = getattr(self, '_async_overlay', None)

        if busy:
            if overlay is None:
                overlay = QLabel(self)
                overlay.setObjectName("AsyncOverlay")
                overlay.setStyleSheet(
                    "background: rgba(0,0,0,160); color: #fff;"
                    "font-size: 14px; border-radius: 6px; padding: 10px 18px;"
                )
                overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._async_overlay = overlay
            overlay.setText(f"Computing {label}…")
            overlay.adjustSize()
            overlay.move(
                (self.width()  - overlay.width())  // 2,
                (self.height() - overlay.height()) // 2,
            )
            overlay.show()
            overlay.raise_()
            QApplication.processEvents()  # flush paint so overlay is visible before thread starts
            self.setEnabled(False)
        else:
            if overlay is not None:
                overlay.hide()
            self.setEnabled(True)


# ---------------------------------------------------------------------------
# Helpers (module-level, no Qt, safe to call from any thread)
# ---------------------------------------------------------------------------

def _tessellate(shape_after):
    """Build a Mesh from shape_after in the worker thread (no GL calls)."""
    if shape_after is None:
        return None
    from viewer.mesh import Mesh
    try:
        return Mesh(shape_after)
    except Exception as ex:
        print(f"[Mesh] Tessellation failed: {ex}")
        return None


def _upload_mesh(viewport, body_id: str, mesh):
    """GL-upload a pre-tessellated Mesh and swap it into the viewport."""
    from OpenGL.GL import glDeleteBuffers
    viewport._body_visible.setdefault(body_id, True)
    viewport.makeCurrent()
    old = viewport._meshes.get(body_id)
    if old:
        for buf in (old.vbo_verts, old.vbo_normals, old.vbo_edges, old.ebo):
            if buf is not None:
                glDeleteBuffers(1, [buf])
    if mesh is not None:
        mesh.upload()
        viewport._meshes[body_id] = mesh
    else:
        viewport._meshes.pop(body_id, None)
    viewport.selection.clear_for_body(body_id)
    if viewport._hovered_vertex[0] == body_id:
        viewport._hovered_vertex = (None, None)
    if viewport._hovered_edge[0] == body_id:
        viewport._hovered_edge = (None, None)
    viewport.doneCurrent()
    viewport._rebuild_sketch_faces()
    viewport.selection_changed.emit()
    viewport.update()
