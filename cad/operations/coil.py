"""
cad/operations/coil.py

Coil: a circle or square section swept along a helix, optionally cut from /
fused into a target.  Runs in a kernel worker.
"""

from __future__ import annotations

from cad.kernel_pool import offload


def helix_edge(center, normal, x_dir, radius: float, height: float,
               pitch: float, left_hand: bool = False):
    """Helix (build123d Edge) around the axis through *center* along
    *normal*, *height* long, starting on the *x_dir* side."""
    import math
    from build123d import Helix, Vector, Axis
    if radius <= 0 or height <= 0 or pitch <= 0:
        raise RuntimeError("Helix needs a positive diameter, height and pitch")
    c, n, x = Vector(*center), Vector(*normal).normalized(), Vector(*x_dir).normalized()
    helix = Helix(pitch, height, radius, center=c, direction=n, lefthand=left_hand)
    # Turn it so it starts on the x_dir side (Helix starts on the plane's
    # own x axis).
    start = Vector(helix.position_at(0)) - c
    ang = math.degrees(math.atan2(n.dot(start.cross(x)), start.dot(x)))
    if abs(ang) > 1e-9:
        helix = helix.rotate(Axis(c, n), ang)
    return helix


def coil_solid(center, normal, x_dir, radius: float, height: float,
               turns: float, section: str = "circle", size: float = 1.0,
               left_hand: bool = False):
    """The coil around the axis through *center* along *normal*, starting on
    the *x_dir* side; *size* is the section's diameter / side.  A build123d
    Solid.  Raises when the turns would run into each other."""
    from OCP.gp import gp_Dir
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
    from OCP.BRepCheck import BRepCheck_Analyzer
    from build123d import Plane, Circle, Rectangle, Vector, Compound, Wire

    if radius <= 0 or height <= 0 or turns <= 0 or size <= 0:
        raise RuntimeError("Coil needs a positive diameter, height, turns and "
                           "section size")
    pitch = height / turns
    if size >= pitch - 1e-6:
        raise RuntimeError(f"Coil turns overlap: section {size:g} mm is not "
                           f"less than the pitch {pitch:g} mm (height / turns)")
    if size / 2 >= radius - 1e-6:
        raise RuntimeError("Coil section is too big for its diameter")
    helix = helix_edge(center, normal, x_dir, radius, height, pitch, left_hand)
    c, n = Vector(*center), Vector(*normal).normalized()
    p0 = Vector(helix.position_at(0))
    # Section in the plane through the axis and the start point.
    radial = (p0 - c - n * n.dot(p0 - c)).normalized()
    sec_plane = Plane(origin=p0, x_dir=radial, z_dir=radial.cross(n))
    shape2d = Circle(size / 2) if section == "circle" else Rectangle(size, size)
    profile = shape2d.moved(sec_plane.location).faces()[0].outer_wire()

    mk = BRepOffsetAPI_MakePipeShell(Wire([helix]).wrapped)
    mk.SetMode(gp_Dir(n.X, n.Y, n.Z))              # binormal = axis: no roll
    mk.Add(profile.wrapped)
    mk.Build()
    if not mk.IsDone() or not mk.MakeSolid():
        raise RuntimeError("Coil failed — OCCT couldn't sweep the section")
    shape = mk.Shape()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise RuntimeError("Coil made an invalid solid")
    solids = Compound(shape).solids()
    if not solids or solids[0].volume < 1e-9:
        raise RuntimeError("Coil made no solid")
    return solids[0]


@offload()
def coil_task(center, normal, x_dir, radius, height, turns, section, size,
              left_hand=False, target=None, mode=None):
    """The coil, or (mode "cut" / "fuse") target with it cut / fused."""
    from cad.solid_ref import as_compound
    from cad.operations.boolean import bool_op
    part = coil_solid(center, normal, x_dir, radius, height, turns, section,
                      size, left_hand)
    if mode is None:
        return part
    out = bool_op(mode, [target], [part])
    if out is None:
        raise RuntimeError(f"Boolean {mode} failed")
    return as_compound(out)
