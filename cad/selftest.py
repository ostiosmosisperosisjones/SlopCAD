"""
cad/selftest.py

`slopcad --selftest`: headless check that a (frozen) build can do real kernel
work — every heavy op below runs in a kernel_pool worker process, which is the
part a PyInstaller build breaks first (spawn re-entering the frozen exe,
missing hidden imports in the child, BREP pickling).  No GL, no window.

Prints one JSON line per check and a summary; exit code 0 only if all pass.
"""
from __future__ import annotations

import json
import os
import sys
import time


def _sleep(s: float):
    """Pool target for the timeout check (must be module-level: workers
    resolve the function by module + qualname)."""
    time.sleep(s)
    return s


def _pid():
    return os.getpid()


def _checks():
    import numpy as np
    from build123d import Box, Cylinder, Pos
    from cad.kernel_pool import pool, KernelTimeout
    from cad.operations.boolean import bool_op
    from cad.operations.fillet_proc import fillet_edges_isolated
    from cad.operations.chamfer import chamfer_edges
    from cad.operations.thicken import thicken_face
    from cad.operations.revolve import revolve_face_direct

    box = Box(20, 20, 10)
    state = {}

    def worker_pid():
        pid = pool().call(_pid)
        assert pid != os.getpid(), "call ran in-process, not in a worker"
        return f"worker pid {pid}"

    def boolean():
        r = bool_op("cut", [box.wrapped], [Pos(0, 0, 0) * Cylinder(4, 30)])
        want = 4000 - np.pi * 16 * 10
        assert abs(r.volume - want) < 1e-3, (r.volume, want)
        state["holed"] = r
        return f"volume {r.volume:.3f}"

    def fillet():
        b = state["holed"]
        top = max(b.faces(), key=lambda f: f.center().Z)
        edges = [e for e in top.edges() if e.geom_type.name == "LINE"]
        r = fillet_edges_isolated(b, [], [e.wrapped for e in edges], 1.0)
        assert r.is_valid and r.volume < b.volume
        state["filleted"] = r
        return f"{len(edges)} edges, volume {r.volume:.3f}"

    def chamfer():
        bot = min(box.faces(), key=lambda f: f.center().Z)
        r = chamfer_edges(box, [e.wrapped for e in bot.edges()], 1.0, 45.0)
        assert r.is_valid and 3950 < r.volume < 4000 - 1, r.volume
        return f"volume {r.volume:.3f}"

    def thicken():
        top = max(box.faces(), key=lambda f: f.center().Z)
        r = thicken_face(box.wrapped, top.wrapped, -2.0)
        assert abs(r.volume - 3200) < 1e-3, r.volume
        return f"volume {r.volume:.3f}"

    def revolve():
        # top face (x -10..10) about an in-plane line x=30: radii 20..40
        top = max(box.faces(), key=lambda f: f.center().Z)
        r = revolve_face_direct(None, top, np.array([30.0, 0, 5.0]),
                                np.array([0, 1.0, 0]), 90.0)
        want = np.pi / 4 * (40 ** 2 - 20 ** 2) * 20
        assert r.is_valid and abs(abs(r.volume) - want) < 1e-2, (r.volume, want)
        return f"volume {r.volume:.3f}"

    def mesh():
        from viewer.mesh import Mesh
        m = Mesh(state["filleted"])
        n = len(m.occt_faces)
        assert len(m.tris) > 0, "empty mesh"
        assert len(m.triangles_per_face) == n, "tessellator lost faces"
        return f"{n} faces, {len(m.tris)} tris"

    def timeout_respawn():
        try:
            pool().call(_sleep, 30, timeout=1.0)
        except KernelTimeout:
            pass
        else:
            raise AssertionError("timeout did not fire")
        assert pool().call(_sleep, 0) == 0, "no worker after kill"
        return "killed hung worker, replacement answered"

    return [("worker_pid", worker_pid), ("boolean", boolean),
            ("fillet", fillet), ("chamfer", chamfer), ("thicken", thicken),
            ("revolve", revolve), ("mesh", mesh),
            ("timeout_respawn", timeout_respawn)]


def run() -> int:
    from cad.kernel_pool import pool
    t_all = time.time()
    failed = 0
    try:
        checks = _checks()
    except Exception as ex:                     # import failure in the bundle
        print(json.dumps({"check": "import", "ok": False,
                          "error": f"{type(ex).__name__}: {ex}"}), flush=True)
        return 1
    for name, fn in checks:
        t0 = time.time()
        try:
            info = fn()
            rec = {"check": name, "ok": True, "info": info}
        except Exception as ex:
            failed += 1
            rec = {"check": name, "ok": False,
                   "error": f"{type(ex).__name__}: {ex}"}
        rec["s"] = round(time.time() - t0, 2)
        print(json.dumps(rec), flush=True)
    pool().shutdown()
    print(json.dumps({"summary": f"{len(checks) - failed}/{len(checks)} passed",
                      "frozen": bool(getattr(sys, "frozen", False)),
                      "s": round(time.time() - t_all, 2)}), flush=True)
    return 1 if failed else 0


def gui_check(app, window, out_path: str | None) -> None:
    """`slopcad --smoke-gui [OUT]` (run under xvfb-run): the real window came
    up with a working GL context, paints, and a kernel op still runs with the
    full app (warm pool, ui-idle hook, nav-only filter) installed.  Writes one
    JSON object to OUT (default stdout) and exits."""
    rec = {"check": "gui", "ok": False}
    try:
        from build123d import Box, Cylinder
        from viewer.viewport import Viewport
        from cad.operations.boolean import bool_op
        vps = window.findChildren(Viewport)
        assert vps, "no Viewport widget"
        vp = vps[0]
        ctx = vp.context()
        assert ctx is not None and ctx.isValid(), "no valid GL context"
        img = vp.grabFramebuffer()
        assert not img.isNull(), "framebuffer grab failed"
        from OpenGL.GL import glGetString, GL_RENDERER, GL_VERSION
        vp.makeCurrent()
        rec["gl"] = f"{glGetString(GL_RENDERER).decode()} / {glGetString(GL_VERSION).decode()}"
        vp.doneCurrent()
        r = bool_op("cut", [Box(10, 10, 10).wrapped], [Cylinder(2, 20)])
        assert abs(r.volume - (1000 - 3.141592653589793 * 4 * 10)) < 1e-3, r.volume
        rec["ok"] = True
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"
    text = json.dumps(rec)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text, flush=True)
    app.exit(0 if rec["ok"] else 1)
