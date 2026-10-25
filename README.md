# SlopCAD

A parametric 3D CAD program. Keyboard-driven, constraint-backed sketching;
solid modeling with a feature history; STEP and STL export. Written in Python on
top of OpenCASCADE.

It's more capable than the name suggests and rougher than a finished product.

## What it does

- **Sketching.** The sketch workflow is keyboard-driven: you call tools and snap
  points from the keyboard rather than dragging, in the style of OMAX Layout.
  A SolveSpace constraint solver is available so sketches can be made parametric
  (coincidence, tangency, dimensions).
- **Feature history.** Extrude, cut, revolve, loft, boolean
  union/subtract/intersect, fillet, chamfer, thicken/shell, offset datum
  planes. Cuts that split a body produce separate bodies. Editing an earlier
  feature replays the tree; sketches, faces and edges picked downstream follow
  the geometry they came from, so moderate upstream edits don't break them.
- **Named parameters.** Define document parameters (`width = 40mm`,
  `wall = width/20`) and type expressions into any dimension or feature size.
  Changing a parameter replays the whole model.
- **Python scripting.** `cad.api` builds and edits models from a script and
  writes a normal `.vc` you can open in the GUI:

  ```python
  from cad.api import Doc
  doc = Doc.new()
  doc.param("w", "40mm")
  box = doc.extrude(doc.sketch("XY").rect("w", 20), "w/4")
  doc.extrude(doc.sketch(box, ">Z").circle((0, 0), 6), 3, cut=True)
  doc.save("box.vc")
  ```

  Run headless with `python -m cad.api script.py out.vc`.
- **Solid modeling on OpenCASCADE**, the same geometry kernel used by FreeCAD.
- **Image tracing.** Load a photo or line-art image and trace it into sketch
  geometry (lines and tangent arcs). Two front ends: silhouette/fill and
  stroke-centerline. Similar in purpose to OMAX Layout's image tracing.
- **Export** to STEP and STL. Projects save to `.vc` files with full history.

## Running it

Needs Python 3.12 and a working OpenGL setup.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Prebuilt Windows and Linux (AppImage) binaries are on the
[Releases](../../releases) page.

## Known limitations

This is a one-person project still under development. Expect the following:

- **Fillets and chamfers can fail or run slowly** on difficult edges (long
  splines, awkward geometry). Fillets run in a separate process with a timeout
  so a bad one cannot hang the application, but some edges will not round.
  If an upstream edit removes an edge a fillet or chamfer used, the feature
  reports it and you reselect.
- **Image tracing requires a clean input.** It uses classical thresholding,
  contour/skeleton tracing, and curve fitting, with manual sliders — no machine
  learning. Photos with shadows, gradients, or busy backgrounds are difficult.
- **Missing features.** No assemblies, sketch patterns/arrays, or drawings; 3MF
  export is not implemented.
- **Linux-first.** Developed on Artix Linux under X11. A Windows build exists,
  but Linux is the primary target.

## Built with

[build123d](https://github.com/gumyr/build123d) / OpenCASCADE (geometry),
[python-solvespace](https://github.com/KmolYuan/solvespace) (sketch solver),
PyQt6 and PyOpenGL (interface and viewport), NumPy / SciPy / scikit-image
(image tracing).

## License

Copyright (C) 2026 ostiosmosisperosisjones

GPLv3. See [LICENSE](LICENSE).

