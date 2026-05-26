"""
cad/debug.py

Verbose-logging and BREP-dumping helpers, gated by CADAPP_DEBUG=1.

When the env var is unset (the normal case) both log() and dump_brep()
return immediately, so callers can sprinkle them at op boundaries
without worrying about cost or noise.

Quick use:
    from cad.debug import log, dump_brep, shape_stats
    log("Thicken", f"face_idx={idx} thickness={thk}")
    log("Thicken", f"input body: {shape_stats(body_occ)}")
    dump_brep("thicken_input_body", body_occ)
    dump_brep("thicken_input_face", face_occ)
    # ...run op...
    dump_brep("thicken_result", result.wrapped)

Dumps land in /tmp/cadapp_debug/<timestamp>__<tag>.brep — reload in a
standalone script with BRepTools.Read_s() to poke at them in isolation.
"""

from __future__ import annotations
import os
import time
from pathlib import Path

_DEBUG = os.environ.get("CADAPP_DEBUG") == "1"
_DUMP_DIR = Path("/tmp/cadapp_debug")


def is_enabled() -> bool:
    return _DEBUG


def log(tag: str, msg: str) -> None:
    if not _DEBUG:
        return
    print(f"[{tag}] {msg}", flush=True)


def shape_stats(shape) -> str:
    """One-line summary of a TopoDS_Shape / Compound for log lines."""
    if shape is None:
        return "None"
    try:
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_SHELL, TopAbs_EDGE
        s = shape.wrapped if hasattr(shape, "wrapped") else shape
        if s.IsNull():
            return "IsNull=True"

        def _count(t):
            n = 0
            e = TopExp_Explorer(s, t)
            while e.More():
                n += 1
                e.Next()
            return n

        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        bb = Bnd_Box()
        try:
            BRepBndLib.Add_s(s, bb)
            if not bb.IsVoid():
                xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
                bbox = (f"bbox=[{xmin:.3f},{ymin:.3f},{zmin:.3f}]→"
                        f"[{xmax:.3f},{ymax:.3f},{zmax:.3f}]")
            else:
                bbox = "bbox=void"
        except Exception:
            bbox = "bbox=?"
        return (f"solids={_count(TopAbs_SOLID)} shells={_count(TopAbs_SHELL)} "
                f"faces={_count(TopAbs_FACE)} edges={_count(TopAbs_EDGE)} {bbox}")
    except Exception as ex:
        return f"stats-error: {ex}"


def dump_brep(tag: str, shape) -> str | None:
    """
    Write a TopoDS_Shape to /tmp/cadapp_debug/<ts>__<tag>.brep.
    Returns the path written, or None if disabled / dump failed.
    """
    if not _DEBUG or shape is None:
        return None
    try:
        from OCP.BRepTools import BRepTools
        s = shape.wrapped if hasattr(shape, "wrapped") else shape
        if s.IsNull():
            log("Debug", f"dump_brep skipped — {tag} is null")
            return None
        _DUMP_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        # nanosecond suffix so calls in the same second don't collide
        suffix = f"{time.time_ns() % 1_000_000:06d}"
        path = _DUMP_DIR / f"{ts}_{suffix}__{tag}.brep"
        BRepTools.Write_s(s, str(path))
        log("Debug", f"dumped {tag} → {path}")
        return str(path)
    except Exception as ex:
        log("Debug", f"dump_brep({tag}) failed: {ex}")
        return None
