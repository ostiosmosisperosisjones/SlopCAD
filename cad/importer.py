from build123d import import_step, Compound
from OCP.ShapeFix import ShapeFix_Shape
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.BRepLib import BRepLib


def _heal(occ_shape):
    """Run the standard healing pipeline on a raw TopoDS_Shape."""
    # 1. General shape fix (bad topology, missing seams, etc.)
    fixer = ShapeFix_Shape(occ_shape)
    fixer.Perform()
    fixed = fixer.Shape()

    # 2. Unify same-domain faces/edges and concatenate B-splines to remove
    #    artificial C0 seams introduced by STEP tessellation or export quirks.
    unifier = ShapeUpgrade_UnifySameDomain(fixed, True, True, True)
    unifier.Build()
    unified = unifier.Shape()

    # 3. Encode geometric continuity so offset/thicken ops can exploit it.
    BRepLib.EncodeRegularity_s(unified)

    return unified


def load_step(path):
    """Load a STEP file, apply geometry healing, and return the shape."""
    raw = import_step(path)
    healed_occ = _heal(raw.wrapped)
    return Compound(healed_occ)


def embed_step(history, path: str) -> str | None:
    """Copy a STEP file into the project so it survives the original moving
    or the .vc being sent elsewhere.  Returns the embed name for
    params["embed"], or None if the file can't be read."""
    import hashlib
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    name = f"imports/{hashlib.sha1(data).hexdigest()[:16]}.step"
    history.embedded[name] = data
    return name


def load_import_source(history, params: dict):
    """The healed compound for an import entry: embedded copy first, then
    the original path.  Raises if neither is available."""
    data = history.embedded.get(params.get("embed") or "")
    if data is None:
        return load_step(params["path"])
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".step")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return load_step(tmp)
    finally:
        os.unlink(tmp)


def pick_import_solid(compound, params: dict):
    """The solid an import entry refers to: by SolidRef when recorded
    (robust to solid order changing across OCCT versions), else by index."""
    solids = list(compound.solids()) or [compound]
    ref = params.get("solid_ref")
    if ref:
        from cad.solid_ref import SolidRef
        _, solid = SolidRef(ref["volume"], tuple(ref["com"]),
                            tuple(ref["extents"])).find_in(compound)
        if solid is not None:
            return solid
    i = int(params.get("solid_index", 0))
    return solids[i] if i < len(solids) else solids[0]


def push_step_import(workspace, history, compound, path: str) -> list:
    """Add one body + 'import' history entry per solid of a loaded STEP.
    The file is embedded in the project and each solid is recorded by
    SolidRef as well as index.  Returns [(body, solid)]."""
    import os
    from dataclasses import asdict
    from cad.solid_ref import SolidRef

    solids = list(compound.solids()) or [compound]
    basename = os.path.splitext(os.path.basename(path))[0]
    embed = embed_step(history, path)
    out = []
    for i, solid in enumerate(solids):
        name = basename if len(solids) == 1 else f"{basename}  [{i+1}]"
        body = workspace.add_body(name, solid)
        params = {"path": path, "solid_index": i,
                  "solid_ref": asdict(SolidRef.from_b3d_solid(solid))}
        if embed:
            params["embed"] = embed
        history.push(label=f"Import  {name}", operation="import", params=params,
                     body_id=body.id, face_ref=None,
                     shape_before=None, shape_after=solid)
        out.append((body, solid))
    return out
