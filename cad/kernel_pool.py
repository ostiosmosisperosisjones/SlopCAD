"""
cad/kernel_pool.py

Warm worker processes for heavy OCCT calls.

OCP holds the GIL for the whole of every kernel call, so a "background" thread
running a fillet or a boolean still freezes the UI, and threads never run two
kernel calls at once.  Child processes fix both.  fillet_proc used to spawn a
fresh child per call, paying ~2 s of imports every time (preview, commit,
replay, every fit-largest probe); these workers start once and stay warm.

    from cad.kernel_pool import offload

    @offload()                      # runs in a worker when called from the app
    def chamfer_edges(shape, edge_occs, d): ...

    pool().call(fn, *args, timeout=25, should_cancel=cb, **kwargs)

Arguments and results cross the process boundary pickled; shapes go as binary
BREP.  A shape that is a sub-shape of an earlier argument (edges or faces of
the body passed first) travels as an index into that body's
TopExp.MapShapes, so the worker gets real sub-shapes of its copy of the body —
the BREP round trip preserves topology order, so indices are stable.

A call that times out or is canceled kills its worker (the only way to stop a
spinning OCCT call) and raises; the pool starts a replacement on demand.

CAD_KERNEL_POOL=0 runs everything in-process (debugging, profiling).
"""

from __future__ import annotations

import atexit
import functools
import importlib
import io
import multiprocessing as _mp
import os
import pickle
import threading
import time

_IN_WORKER = False


def enabled() -> bool:
    return not _IN_WORKER and os.environ.get("CAD_KERNEL_POOL", "1") != "0"


# Called repeatedly while the MAIN thread waits on a worker (main.py installs
# a Qt processEvents with navigation-only input), so a replay or re-mesh that
# runs on the main thread still repaints and orbits instead of freezing.  Never
# re-entered: a pool call made from inside the hook just waits.
_ui_idle_hook = None
_in_ui_idle = False
_activity_hook = None


def set_activity_hook(fn) -> None:
    """fn(delta, qualname) is called (from any thread) as each worker call
    starts (+1) and ends (-1) — drives the status-bar spinner."""
    global _activity_hook
    _activity_hook = fn


def _activity(delta: int, name: str) -> None:
    if _activity_hook is not None:
        try:
            _activity_hook(delta, name)
        except Exception:
            pass


def set_ui_idle_hook(fn) -> None:
    global _ui_idle_hook
    _ui_idle_hook = fn


def ui_idle() -> None:
    """Run the UI idle hook if we're on the main thread and not already in it."""
    global _in_ui_idle
    if (_ui_idle_hook is None or _in_ui_idle
            or threading.current_thread() is not threading.main_thread()):
        return
    _in_ui_idle = True
    try:
        _ui_idle_hook()
    except Exception as ex:          # a broken hook must not kill the op
        print(f"[kernel_pool] ui idle hook: {ex}")
    finally:
        _in_ui_idle = False


class KernelTimeout(RuntimeError):
    pass


class KernelCanceled(RuntimeError):
    pass


class KernelWorkerDied(RuntimeError):
    pass


# ── shape packing ──────────────────────────────────────────────────────────

class _Root:
    __slots__ = ("data", "cls")

    def __init__(self, data, cls):
        self.data, self.cls = data, cls


class _Sub:
    __slots__ = ("root", "typ", "index", "orient", "cls")

    def __init__(self, root, typ, index, orient, cls):
        self.root, self.typ, self.index, self.orient, self.cls = root, typ, index, orient, cls


def _occ_of(obj):
    from OCP.TopoDS import TopoDS_Shape
    if isinstance(obj, TopoDS_Shape):
        return obj, None
    w = getattr(obj, "wrapped", None)
    if isinstance(w, TopoDS_Shape):
        # Rewrap as the nearest topology class: a Box or a Part can't be
        # rebuilt from a TopoDS shape, their base Solid/Compound can.
        t = next((c for c in type(obj).__mro__
                  if c.__module__.startswith("build123d.topology")), type(obj))
        return w, (t.__module__, t.__qualname__)
    return None, None


class _Packer:
    def __init__(self):
        self.roots = []         # TopoDS_Shape per root slot
        self._maps = {}         # (root slot, shape type) → IndexedMap

    def _map(self, slot, typ):
        m = self._maps.get((slot, typ))
        if m is None:
            from OCP.TopExp import TopExp
            from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap
            m = IMap()
            TopExp.MapShapes_s(self.roots[slot], typ, m)
            self._maps[(slot, typ)] = m
        return m

    def pack(self, obj):
        occ, cls = _occ_of(obj)
        if occ is not None:
            if occ.IsNull():
                return _Root(b"", cls)
            typ = occ.ShapeType()
            for slot in range(len(self.roots)):
                k = self._map(slot, typ).FindIndex(occ)
                if k:
                    return _Sub(slot, int(typ), k, int(occ.Orientation()), cls)
            from OCP.BinTools import BinTools
            buf = io.BytesIO()
            BinTools.Write_s(occ, buf)
            self.roots.append(occ)
            return _Root(buf.getvalue(), cls)
        if isinstance(obj, list):
            return [self.pack(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self.pack(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self.pack(v) for k, v in obj.items()}
        return obj


def _wrap(occ, cls):
    from build123d.topology import downcast
    if occ.IsNull():
        return occ if cls is None else getattr(importlib.import_module(cls[0]), cls[1])()
    occ = downcast(occ)
    if cls is None:
        return occ
    return getattr(importlib.import_module(cls[0]), cls[1])(occ)


class _Unpacker:
    def __init__(self):
        self.roots = []
        self._maps = {}

    def unpack(self, obj):
        if isinstance(obj, _Root):
            from OCP.TopoDS import TopoDS_Shape
            occ = TopoDS_Shape()
            if obj.data:
                from OCP.BinTools import BinTools
                BinTools.Read_s(occ, io.BytesIO(obj.data))
            self.roots.append(occ)
            return _wrap(occ, obj.cls)
        if isinstance(obj, _Sub):
            from OCP.TopAbs import TopAbs_ShapeEnum, TopAbs_Orientation
            m = self._maps.get((obj.root, obj.typ))
            if m is None:
                from OCP.TopExp import TopExp
                from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap
                m = IMap()
                TopExp.MapShapes_s(self.roots[obj.root], TopAbs_ShapeEnum(obj.typ), m)
                self._maps[(obj.root, obj.typ)] = m
            return _wrap(m.FindKey(obj.index).Oriented(TopAbs_Orientation(obj.orient)), obj.cls)
        if isinstance(obj, list):
            return [self.unpack(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self.unpack(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self.unpack(v) for k, v in obj.items()}
        return obj


def _dumps(obj) -> bytes:
    return pickle.dumps(_Packer().pack(obj), protocol=pickle.HIGHEST_PROTOCOL)


def _loads(data: bytes):
    return _Unpacker().unpack(pickle.loads(data))


def _picklable_exc(ex: BaseException) -> BaseException:
    try:
        pickle.loads(pickle.dumps(ex))
        return ex
    except Exception:
        return RuntimeError(f"{type(ex).__name__}: {ex}")


# ── worker side ─────────────────────────────────────────────────────────────

_WARM_MODULES = ("build123d", "cad.operations.fillet", "cad.operations.chamfer",
                 "cad.operations.thicken", "cad.operations.revolve",
                 "cad.operations.boolean", "viewer.mesh", "viewer.preview_job")


def _resolve(mod: str, qual: str):
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return getattr(obj, "_local", obj)


def _serve(conn):
    global _IN_WORKER
    _IN_WORKER = True
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    for m in _WARM_MODULES:
        try:
            importlib.import_module(m)
        except Exception:
            pass
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        mod, qual, payload = msg
        try:
            args, kwargs = _loads(payload)
            out = ("ok", _dumps(_resolve(mod, qual)(*args, **kwargs)))
        except BaseException as ex:      # noqa: BLE001 — report everything
            out = ("err", _picklable_exc(ex))
        try:
            conn.send(out)
        except (EOFError, OSError):
            return


# ── parent side ─────────────────────────────────────────────────────────────

class _Worker:
    def __init__(self, ctx):
        parent, child = ctx.Pipe()
        self.conn = parent
        self.proc = ctx.Process(target=_serve, args=(child,), daemon=True,
                                name="cad-kernel")
        self.proc.start()
        child.close()

    def kill(self):
        try:
            self.proc.kill()
            self.proc.join(1.0)
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass


class KernelPool:
    def __init__(self, size: int | None = None):
        self.size = size or max(2, min(6, (os.cpu_count() or 4) // 2))
        self._ctx = _mp.get_context("spawn")
        self._idle: list[_Worker] = []
        self._count = 0
        self._cv = threading.Condition()
        # Set at interpreter exit: no new workers.  A background call still
        # in flight then sees its worker killed by multiprocessing's exit
        # hook; a replacement spawned after that hook is never reaped and
        # keeps the resource tracker's pipe open, hanging the exit.
        self._closing = False

    def warm(self, n: int | None = None):
        """Start up to n workers now so the first real call doesn't pay the
        spawn + import cost.  Non-blocking."""
        with self._cv:
            want = min(self.size, n or self.size) - self._count
            for _ in range(max(0, want)):
                self._count += 1
                try:
                    self._idle.append(_Worker(self._ctx))
                except Exception:
                    self._count -= 1

    def _acquire(self) -> _Worker:
        with self._cv:
            while True:
                while self._idle:
                    w = self._idle.pop()
                    if w.proc.is_alive():
                        return w
                    w.kill()
                    self._count -= 1
                if self._closing:
                    raise KernelWorkerDied("kernel pool is shut down")
                if self._count < self.size:
                    self._count += 1
                    break
                self._cv.wait()
        try:
            return _Worker(self._ctx)
        except Exception:
            with self._cv:
                self._count -= 1
                self._cv.notify()
            raise

    def _release(self, w: _Worker, healthy: bool):
        with self._cv:
            if self._closing:
                w.kill()
                self._count -= 1
            elif healthy:
                self._idle.append(w)
            else:
                # Killed (timeout/cancel) or died: start a replacement now so
                # the next call finds it warm.
                w.kill()
                try:
                    self._idle.append(_Worker(self._ctx))
                except Exception:
                    self._count -= 1
            self._cv.notify()

    def call(self, fn, *args, timeout: float | None = None,
             should_cancel=None, **kwargs):
        """Run fn(*args, **kwargs) in a worker; return its result.  fn must be
        a module-level function (or an @offload wrapper of one)."""
        fn = getattr(fn, "_local", fn)
        payload = _dumps((args, kwargs))
        _activity(+1, fn.__qualname__)
        try:
            return self._call(fn, payload, timeout, should_cancel)
        finally:
            _activity(-1, fn.__qualname__)

    def _call(self, fn, payload, timeout, should_cancel):
        w = self._acquire()
        healthy = False
        try:
            w.conn.send((fn.__module__, fn.__qualname__, payload))
            deadline = None if timeout is None else time.monotonic() + timeout
            on_main = threading.current_thread() is threading.main_thread()
            while not w.conn.poll(0.01 if on_main else 0.05):
                ui_idle()
                if not w.proc.is_alive():
                    raise KernelWorkerDied(
                        f"kernel worker died (exit {w.proc.exitcode}) in {fn.__qualname__}")
                if should_cancel is not None and should_cancel():
                    raise KernelCanceled(f"{fn.__qualname__} canceled")
                if deadline is not None and time.monotonic() > deadline:
                    raise KernelTimeout(f"{fn.__qualname__} timed out after {timeout:g}s")
            try:
                status, body = w.conn.recv()
            except (EOFError, OSError):
                raise KernelWorkerDied(f"kernel worker died in {fn.__qualname__}")
            healthy = True
            if status == "err":
                raise body
            return _loads(body)
        finally:
            self._release(w, healthy)

    def shutdown(self):
        with self._cv:
            self._closing = True
            for w in self._idle:
                try:
                    w.conn.send(None)
                except Exception:
                    pass
                w.kill()
            self._idle.clear()
            self._count = 0
            self._cv.notify_all()


_pool: KernelPool | None = None
_pool_lock = threading.Lock()


def pool() -> KernelPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = KernelPool()
        return _pool


@atexit.register
def _shutdown_at_exit():
    # Registered after multiprocessing's own exit hook, so runs before it.
    if _pool is not None:
        _pool.shutdown()


def offload(timeout: float | None = None):
    """Decorator: run the function in a kernel worker when called from the app
    process (in-process inside a worker, or with CAD_KERNEL_POOL=0)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            return pool().call(fn, *args, timeout=timeout, **kwargs)
        wrapper._local = fn
        return wrapper
    return deco
