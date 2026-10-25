# Bug: Included-geometry → "placed lines" produces FACETED extrudes

**Status:** REVERTED 2026-07-23. The included-geometry mirror/offset feature (bug #1
from that session) was reverted because it introduces faceting. To attack fresh.

## Symptom chain (how it "screws up everything")

1. You include a curved edge into a sketch, then **mirror or offset** it (the reverted
   feature turned the included `ReferenceEntity` into "real" drawn geometry so the
   transform tools could grab it).
2. For a curve **without a clean line/circle `occ_edge`** (BSpline edges, mesh-derived
   edges, offset results), the conversion falls back to a **polyline: N separate
   `LineEntity` segments** chained along `ref.points` (or a 33-sample curve).
3. Extruding that faceted profile produces a wall made of **N flat faces** instead of
   1–2 smooth faces.
4. **Extra facets = extra faces.** Every downstream op that identifies a face by
   **positional index** (`face_indices` on sketch cuts/extrudes, `edge_refs` on
   fillets/chamfers) now resolves against a shifted/renumbered face set → cuts the
   wrong region, fillets a missing edge, and the whole history cascades into errors.

This is what cooked `_tests/headphones.vc`: entry-50 sketch cut resolved
`face_indices=[1]` (a big wrap-around region) instead of the intended small face,
and entry-59 fillet's `edge_refs` pointed at edges that no longer existed.

## Root-cause quantification (measured)

`cad/sketch_tools/reference_convert.py::reference_to_uv_entities` (the reverted file):
- Included **circle with occ_edge** → `1 ArcEntity` (smooth). ✓ fine.
- Included **curve without usable occ_edge** → `_polyline_to_lines(ref.points)`:
  measured **19 `LineEntity` segments** for a 20-point quarter-circle include.
  → faceted. ✗

Two faceting sources in the reverted work:
1. **Transform-time** (`reference_convert.py`): mirror/offset of any non-line/circle
   include emits a line-segment chain.
2. **Region-build fallback** (`cad/sketch.py::build_faces`, off-plane projection of
   the curve-sample polyline branch): **THIS WAS ALSO REVERTED** — see below.

### CONFIRMED CULPRIT #2: the sketch.py off-plane region "fix" broke existing files

Measured on `headphones.vc` entry-50 sketch, same file, only difference is the code:
- **Pristine `build_faces`:** off-plane included edges are DISCARDED (their wire floats
  off the sketch plane, the OCC splitter ignores it) → **2 regions**. `face_indices=[1]`
  cuts the correct region. Only entries 77/80/81 fail (a minor pre-existing issue).
- **With the "fix":** the polyline fallback projected those included edges onto the
  plane so they participate → **5 regions**. Now `face_indices=[1]` points at a
  DIFFERENT region (a big wrap-around) → wrong cut → cascade from entry 50 onward,
  ~28–39 errored entries.

So making included edges bound regions (technically "more correct") **renumbers the
region set and silently breaks every saved file that relied on the old count.** This is
the same face-index-instability problem as culprit #1, triggered from the region side.
Reverted `cad/sketch.py` + deleted `_tests/test_included_edge_region.py`.

**Lesson for the fresh attempt:** ANY change that alters `build_faces` region count/order
is a breaking change for existing files unless paired with index-stable face selection
(direction #2 below). Do that FIRST.

## Why positional face/edge indices are the deeper fragility

The faceting is the trigger, but the reason it "screws up everything" is that
downstream ops store **positional indices** into geometry that isn't stable:
- Sketch cut/extrude: `face_indices=[i]` into `build_faces()` output order (by area).
- Fillet/chamfer: `edge_refs` (midpoint/length/face-sig fingerprints) — more robust,
  but still fail to relocate when the edge genuinely changes/vanishes.
Any face-count change upstream renumbers these. (Same class as the split-body
positional-index bug fixed earlier via `SolidRef`.)

## Directions for the fresh attempt

1. **Preserve smooth curves through transforms.** Mirror/offset of an included curve
   should produce a genuine reflected/offset **curve** (BSpline/arc), not a polyline.
   - Reflect: transform the `occ_edge`'s control points / build a mirrored OCC curve.
   - Offset: OCC 2D curve offset (`Geom2dAPI` / `BRepOffsetAPI`), not per-segment.
   - Keep `ReferenceEntity` as a first-class curve entity end-to-end rather than
     down-converting to line segments.
2. **Make face selection index-stable.** Bake a UV centroid (or a small fingerprint)
   alongside `face_indices` on sketch cuts/extrudes and re-resolve the face by
   nearest-centroid on replay — immune to region renumbering. (Precedent:
   `_match_sketch_face_by_target` in `op_base.py`, and `SolidRef` for bodies.)
3. **Audit the kept `build_faces` polyline fallback** (33-sample) for curved includes —
   confirm it isn't independently faceting extrude/cut tool geometry.

## What was kept from that session (unrelated, safe)
- **Edit / Re-enter** on history right-click (`gui/history_panel.py`) — the keeper that
  let the model get stitched back together.
- **Split-parent trim** (`cad/op_base.py`, `cad/history.py`) — a cut/subtract that
  disconnects a body now yields two separate bodies; survives save/load via `SolidRef`.
  Test: `_tests/test_split_parent_trim.py`.
- **Off-plane included-edge region fix** (`cad/sketch.py`) — build_faces projects the
  curve-sample fallback onto the sketch plane. Test: `_tests/test_included_edge_region.py`.
  (NOTE: this is the 33-sample fallback flagged in direction #3 — re-examine.)
