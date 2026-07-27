"""
viewer/gpu_pick.py

GPU colour-picking for mesh faces.

CPU ray/triangle picking (cad.picker) is O(triangles): on a ~2M-triangle
imported assembly a single pick took ~180ms, and hover ran it on every mouse
move. This renders every face in a unique flat colour to an off-screen buffer
once per view; a pick is then one glReadPixels of a single pixel — O(1) in the
triangle count.

Encoding
--------
Faces get a stable global id (`gid`) assigned by walking visible bodies in a
fixed order and numbering their faces consecutively. The colour written is
`gid + 1` packed little-endian into RGB (so 0/black stays "nothing here").
A parallel list maps gid -> (body_id, face_idx) for decode.

The colour buffer is single-sample: MSAA would blend face colours along
silhouettes and decode to bogus ids. Depth is kept too, so occlusion queries
(is this world point the front-most surface under the cursor?) can read it.

Everything uses the same fixed-function pipeline and current GL matrices as the
main viewport draw, so the id image lines up pixel-for-pixel with what's on
screen — no separate projection math to keep in sync.
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import *
import ctypes


class GpuPicker:
    def __init__(self):
        self._fbo = None
        self._color_rb = None
        self._depth_rb = None
        self._w = 0
        self._h = 0
        self._gid_map: list[tuple[str, int]] = []   # gid -> (body_id, face_idx)
        self._sig = None                            # view/scene signature
        self._valid = False
        # Set once if the GL context can't support the offscreen FBO (old GL, no
        # FBO extension, software rasterizer, etc.). When disabled, render() is a
        # permanent no-op and callers fall back to CPU picking silently — no
        # per-frame retry or error spam. Keeps the shipped binary "just works"
        # across GPUs/drivers on Linux and Windows.
        self._disabled = False

    # ------------------------------------------------------------------
    # FBO lifecycle
    # ------------------------------------------------------------------

    def _ensure_fbo(self, w: int, h: int):
        """(Re)allocate the off-screen buffer to match the framebuffer size."""
        if self._fbo is not None and (w, h) == (self._w, self._h):
            return
        # Whatever the caller had bound — for a QOpenGLWidget this is its own
        # default FBO, NOT 0 — must be restored, else later rendering targets a
        # nonexistent buffer ("No fbo, cannot render") and the driver aborts.
        prev_fbo = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))

        self._release()
        self._w, self._h = w, h

        # glGen* return numpy arrays in PyOpenGL; coerce to plain ints so the
        # matching glDelete* / glBind* calls don't get malformed array args
        # (a native-crash hazard).
        self._fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # Single-sample RGBA8 colour renderbuffer (no MSAA — see module docstring).
        self._color_rb = int(glGenRenderbuffers(1))
        glBindRenderbuffer(GL_RENDERBUFFER, self._color_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_RENDERBUFFER, self._color_rb)

        self._depth_rb = int(glGenRenderbuffers(1))
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, self._depth_rb)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)   # restore, never hard-0
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            self._release()
            raise RuntimeError(f"GPU pick FBO incomplete: {status}")

    def _release(self):
        if self._fbo is not None:
            try:
                glDeleteFramebuffers(1, [int(self._fbo)])
                glDeleteRenderbuffers(1, [int(self._color_rb)])
                glDeleteRenderbuffers(1, [int(self._depth_rb)])
            except Exception:
                pass
        self._fbo = self._color_rb = self._depth_rb = None
        self._valid = False

    # ------------------------------------------------------------------
    # Colour encode / decode
    # ------------------------------------------------------------------

    @staticmethod
    def _gid_to_rgb(gid: int) -> tuple[int, int, int]:
        v = gid + 1  # reserve 0 for background
        return (v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF)

    @staticmethod
    def _rgb_to_gid(r: int, g: int, b: int) -> int:
        return (r | (g << 8) | (b << 16)) - 1

    # ------------------------------------------------------------------
    # Render the id image
    # ------------------------------------------------------------------

    def render(self, meshes: dict, workspace, viewport, sig):
        """Render every visible face in its id-colour into the off-screen FBO.

        `viewport` is the GL_VIEWPORT (x, y, w, h) in framebuffer pixels.
        `sig` is an opaque signature; if it matches the last render this is a
        no-op so repeated picks between camera moves are free.
        """
        if self._disabled:
            return
        w, h = int(viewport[2]), int(viewport[3])
        if w <= 0 or h <= 0:
            self._valid = False
            return
        if self._valid and sig == self._sig and (w, h) == (self._w, self._h):
            return

        # First-ever render on an unsupported GL context: disable permanently so
        # we don't retry (and spam) every frame — callers fall back to CPU.
        try:
            self._ensure_fbo(w, h)
        except Exception as ex:
            self._disabled = True
            self._valid = False
            print(f"[gpu_pick] disabled (offscreen FBO unavailable): {ex}; "
                  f"using CPU picking.")
            return

        # Remember the caller's framebuffer. In a QOpenGLWidget the on-screen
        # target is its own default FBO (NOT 0), and there may be drawing after
        # this call, so we must restore exactly what was bound — never hard-code 0.
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        prev_viewport = glGetIntegerv(GL_VIEWPORT)

        self._gid_map = []

        # Save the specific state we mutate (glPushAttrib is unreliable when
        # mixed with FBO binding — it can raise GL_INVALID_OPERATION on pop).
        # MODELVIEW/PROJECTION are left as-is so the id image matches the
        # on-screen projection exactly.
        was_lighting = glIsEnabled(GL_LIGHTING)
        was_blend    = glIsEnabled(GL_BLEND)
        was_ms       = glIsEnabled(GL_MULTISAMPLE)
        prev_clear   = glGetFloatv(GL_COLOR_CLEAR_VALUE)

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, w, h)
        glDisable(GL_LIGHTING)
        glDisable(GL_BLEND)
        glDisable(GL_MULTISAMPLE)
        glShadeModel(GL_FLAT)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnableClientState(GL_VERTEX_ARRAY)
        for body_id, mesh in meshes.items():
            body = workspace.bodies.get(body_id)
            if body is not None and not body.visible:
                continue
            if mesh.vbo_verts is None or mesh.ebo is None:
                continue
            base = len(self._gid_map)
            nfaces = len(mesh.triangles_per_face)
            # Extend the reverse map for this body's faces.
            self._gid_map.extend((body_id, fi) for fi in range(nfaces))

            glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo_verts)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo)

            # One draw per face over its triangle sub-range, coloured by gid.
            tpf = mesh.triangles_per_face
            starts = np.concatenate([[0], np.cumsum(tpf)]).astype(np.int64)
            for fi in range(nfaces):
                count = int(tpf[fi])
                if count == 0:
                    continue
                r, g, b = self._gid_to_rgb(base + fi)
                glColor3ub(r, g, b)
                glDrawElements(GL_TRIANGLES, count * 3, GL_UNSIGNED_INT,
                               ctypes.c_void_p(int(starts[fi]) * 3 * 4))

        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, int(prev_fbo))
        glViewport(int(prev_viewport[0]), int(prev_viewport[1]),
                   int(prev_viewport[2]), int(prev_viewport[3]))

        # Restore the state we changed.
        glShadeModel(GL_SMOOTH)
        if was_lighting: glEnable(GL_LIGHTING)
        if was_blend:    glEnable(GL_BLEND)
        if was_ms:       glEnable(GL_MULTISAMPLE)
        glClearColor(float(prev_clear[0]), float(prev_clear[1]),
                     float(prev_clear[2]), float(prev_clear[3]))

        self._sig = sig
        self._valid = True

    # ------------------------------------------------------------------
    # Queries (coordinates are framebuffer pixels, origin bottom-left)
    # ------------------------------------------------------------------

    def pick(self, x_px: int, y_px: int) -> tuple[str | None, int | None]:
        """Return (body_id, face_idx) under the pixel, or (None, None)."""
        if not self._valid:
            return None, None
        if not (0 <= x_px < self._w and 0 <= y_px < self._h):
            return None, None
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        data = glReadPixels(int(x_px), int(y_px), 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, int(prev_fbo))
        px = np.frombuffer(data, dtype=np.uint8)
        r, g, b = int(px[0]), int(px[1]), int(px[2])
        gid = self._rgb_to_gid(r, g, b)
        if gid < 0 or gid >= len(self._gid_map):
            return None, None
        return self._gid_map[gid]

    def depth_at(self, x_px: int, y_px: int) -> float | None:
        """Window-space depth [0,1] at the pixel, or None if nothing was drawn."""
        if not self._valid:
            return None
        if not (0 <= x_px < self._w and 0 <= y_px < self._h):
            return None
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        d = glReadPixels(int(x_px), int(y_px), 1, 1,
                         GL_DEPTH_COMPONENT, GL_FLOAT)
        glBindFramebuffer(GL_FRAMEBUFFER, int(prev_fbo))
        val = float(np.frombuffer(d, dtype=np.float32)[0])
        return None if val >= 1.0 else val

    def invalidate(self):
        self._valid = False
