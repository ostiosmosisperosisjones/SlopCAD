"""
viewer/preview_job.py

PreviewJob — the one background-compute pattern every op preview needs:

  - at most ONE worker in flight per job, for the job's whole lifetime
    (keep one job per op on the viewport, don't recreate it per panel open,
    or a reopened panel can start a second worker beside a stale one —
    concurrent OCCT on the same shape)
  - latest request wins: submits while busy replace the pending one
  - cancel() drops the pending request and discards the in-flight result
  - the result is delivered on the Qt main thread
  - on_idle (optional) fires whenever nothing is in flight or pending,
    including after a discarded result — for busy indicators

Usage:
    job = PreviewJob(on_done)          # on_done(result, error) — main thread
    job.submit(lambda: heavy(shape))   # runs in a worker thread
    job.cancel()                       # panel closed

Also `preview_mesh_data()`: triangles + edge polylines of a shape for the
immediate-mode previews (revolve, thicken).
"""

from __future__ import annotations

import threading
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal


class PreviewJob(QObject):
    _finished = pyqtSignal(int, object, object)     # gen, result, error

    def __init__(self, on_done: Callable, name: str = "preview",
                 on_idle: Callable | None = None):
        super().__init__()
        self._on_done  = on_done
        self._on_idle  = on_idle
        self._name     = name
        self._gen      = 0
        self._inflight = False
        self._pending: Callable | None = None
        # Emitted from the worker; this QObject lives on the main thread, so
        # the default (auto) connection queues the slot onto it.
        self._finished.connect(self._deliver)

    @property
    def busy(self) -> bool:
        return self._inflight

    def submit(self, compute: Callable) -> None:
        self._gen += 1
        self._pending = compute
        if not self._inflight:
            self._start()

    def cancel(self) -> None:
        self._gen += 1
        self._pending = None

    def _start(self) -> None:
        compute, self._pending = self._pending, None
        self._inflight = True
        threading.Thread(target=self._run, args=(self._gen, compute),
                         daemon=True).start()

    def _run(self, gen: int, compute: Callable) -> None:
        try:
            result, error = compute(), None
        except Exception as ex:
            result, error = None, ex
            print(f"[{self._name}] {ex}")
        self._finished.emit(gen, result, error)

    def _deliver(self, gen: int, result, error) -> None:
        self._inflight = False
        if gen == self._gen:
            self._on_done(result, error)
        if self._pending is not None:
            self._start()
        elif self._on_idle is not None:
            self._on_idle()


def job_for(owner, attr: str, on_done: Callable, name: str,
            on_idle: Callable | None = None) -> PreviewJob:
    """The PreviewJob stored on owner.attr, created on first use.  One per
    op for the viewport's lifetime (see module docstring)."""
    job = getattr(owner, attr, None)
    if job is None:
        job = PreviewJob(on_done, name, on_idle)
        setattr(owner, attr, job)
    return job


def preview_mesh_data(wrapped, deflection: float, edge_samples: int = 16):
    """(tris, edges) for a TopoDS_Shape: tris is a flat list of xyz triples
    (3 per triangle), edges a list of sampled polylines."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_UniformAbscissa

    BRepMesh_IncrementalMesh(wrapped, deflection)
    identity = TopLoc_Location()
    tris = []
    exp = TopExp_Explorer(wrapped, TopAbs_FACE)
    while exp.More():
        tri = BRep_Tool.Triangulation_s(TopoDS.Face_s(exp.Current()), identity)
        if tri is not None:
            for j in range(1, tri.NbTriangles() + 1):
                for ni in tri.Triangle(j).Get():
                    p = tri.Node(ni)
                    tris.append((p.X(), p.Y(), p.Z()))
        exp.Next()

    edges = []
    exp = TopExp_Explorer(wrapped, TopAbs_EDGE)
    while exp.More():
        try:
            adp  = BRepAdaptor_Curve(TopoDS.Edge_s(exp.Current()))
            disc = GCPnts_UniformAbscissa()
            disc.Initialize(adp, edge_samples)
            if disc.IsDone() and disc.NbPoints() >= 2:
                pts = []
                for pi in range(1, disc.NbPoints() + 1):
                    p = adp.Value(disc.Parameter(pi))
                    pts.append((p.X(), p.Y(), p.Z()))
                edges.append(pts)
        except Exception:
            pass
        exp.Next()
    return tris, edges
