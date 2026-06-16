# Replay Reliability — Root Cause & Generalized Plan

## 1. The specific failure (curzie.vc)

```
[warn] Replay error: Sketch plane resolve failed:
       FacePlaneSource: could not relocate face in body 'fd4174f2…'
```

### Chain of facts (verified, not theorized)

- Entry 4 is a `SketchExtrudeOp` with `force_new_body=True` that **splits one
  sketch into 8 separate bodies** (`child_body_ids` = 8 ids, segments of a
  revolved ring). `fd4174f2` is `child_body_ids[0]`.
- Entry 5 is a sketch whose plane is a `FacePlaneSource` on body `fd4174f2`,
  referencing a 45° face at world centroid `(-34.99, 34.99, 7.87)`.
- Entries 7 (thicken) and 8 (revolve_cut) build on `fd4174f2`; entries 9–16
  depend transitively. **The whole right half of the tree hangs off entry 5.**

### Two bugs, one symptom

**Bug A — missing replay branch (a true asymmetry).**
`SketchExtrudeOp.commit` has a `force_new_body` path that creates N child
bodies from `shape_after.solids()`. `SketchExtrudeOp.execute` (the replay path)
has **no** matching branch — it returns only `solids[0]` and never repopulates
the child bodies' `source_shape`. So on reload, 7 of the 8 child bodies get no
geometry at all. (`FaceExtrudeOp` and `FaceRevolveOp` *do* have this branch;
`SketchExtrudeOp` and `Loft` were the ones missing/most fragile.)

**Bug B — the child→segment mapping is never persisted (the deeper one).**
Even with Bug A fixed, the mapping is unrecoverable for this file. At commit,
`child_body_ids[i]` is assigned `list(shape_after.solids())[i]` — by **OCCT
iteration index**. Nothing geometric ties body `fd4174f2` to the physical
segment it represents. Verified:
- `fd4174f2` (child 0) was authored on the `(-34.99, 34.99)` segment.
- A fresh rebuild's `.solids()` puts `(-34.99, 34.99)` at **index 3**, and
  `(0, -49.48)` at index 0.
- So replay hands `fd4174f2` the wrong segment, the 45° face isn't there, and
  the sketch plane can't resolve.

`.solids()` order is deterministic within a build but is **not** a function of
input face order — it follows OCCT's internal ordering, which can differ between
the commit-time shape and a replay-time rebuild (and across edits/reorders/
OCCT versions). **Index-based identity is the root disease; this file is one
symptom.**

## 2. The general problem the user named

> "Fillet the Edge of this Part to this Vertex in the normal direction of this
> Line…" — millions of interactions that must all replay.

Every parametric op references upstream entities by **identity**. The codebase
already has good *geometric* identity for some entity kinds, and fragile
*positional* identity for others. The reliability of save/load is exactly the
reliability of the weakest identity in the graph.

### Reference-type inventory (current state)

| Reference            | Identifies        | Match strategy                    | Robust? |
|----------------------|-------------------|-----------------------------------|---------|
| `FaceRef`            | a planar face     | normal + area + perp-centroid     | ✅ good |
| `AnyFaceRef`         | any face          | centroid + area + category        | ✅ good |
| `EdgeRef`            | an edge           | midpoint + length + tangent + face sigs | ✅ good |
| `FacePlaneSource`    | a sketch plane    | wraps a `FaceRef`                 | ✅ good |
| `WorldPlaneSource`   | XY/XZ/YZ          | constant                          | ✅ trivial |
| `OffsetPlaneSource`  | offset plane      | wraps parent source               | ✅ good |
| `BodyEdgeSource`     | edge on a body    | wraps `EdgeRef` + body_id         | ✅ good |
| `SketchEdgeSource`   | sketch entity     | entry_id (UUID) + entity_idx      | ✅ good |
| history entry        | an operation      | UUID (`entry_id`)                 | ✅ good |
| **child body of a split** | **a solid produced by an N-way split** | **`child_body_ids[i]` ↔ `.solids()[i]` (POSITIONAL)** | ❌ **fragile** |
| sketch face pick     | a loop in a sketch| UV centroid (`face_centroids`)    | ✅ good |

**The one positional identity left in the graph is the split-body mapping**, and
it is load-bearing: anything built on a child body (sketches, cuts, fillets,
unions) inherits its fragility. That is precisely what broke here.

## Status

- ✅ **Phase 3.1 (round-trip gate)** — `_tests/test_roundtrip_split_bodies.py`
  (synthetic split + downstream-op cases) and `test_roundtrip_fixture_files.py`
  (replays any `_tests/fixtures/*.vc` or `$VC_FIXTURES`). Harness gained
  `roundtrip()`, `shape_fingerprint()`, `assert_roundtrip_preserves_bodies()`,
  `push_split_extrude()`.
- ✅ **Phase 1 (geometric body identity)** — `cad/solid_ref.py` (`SolidRef` +
  `assign_solids_to_children` + ref (de)serialization). All four split ops now
  persist `child_solid_refs` at commit and re-locate children by ref on replay,
  with index fallback for legacy files. Added the missing `force_new_body`
  replay branches to `SketchExtrudeOp` / `SketchRevolveOp` / `SketchLoftOp`, and
  `History.replay_from` now seeds non-primary child shapes into the replay
  cache. Full suite green; synthetic split round-trip green.
- ✅ **Phase 2 (legacy recovery)** — `recover_child_solid_refs` in
  `cad/solid_ref.py` re-derives the child→segment mapping for files lacking
  `child_solid_refs` by matching each child to the segment containing its first
  downstream anchor (sketch FacePlaneSource face_ref, an entry's face_ref, or a
  thicken's face_refs). Hooked into `assign_solids_to_children` via optional
  `history`/`entry_index`. A *second* legacy bug surfaced behind it: consumed
  sketches reproject at the consuming op's position, which fails when an
  intervening self-cut destroys the anchor face — fixed by
  `reproject_consumed_sketch` (history.py), which falls back to the sketch's
  authored plane *only* when it re-validates at the sketch's own position
  (so unresolvable planes stay fatal). curzie.vc now replays clean: 0 errors,
  8 distinct child segments, union = 1 solid. Added as a permanent fixture in
  `_tests/fixtures/curzie.vc`.
- ✅ **Phase 3.4 (fail loud, fail local)** — `FaceRef.explain_no_match()`
  reports what was sought + the nearest candidate ranked by how far it got
  through the filter chain (normal → area → perp), wired into
  `FacePlaneSource.resolve`. The opaque "could not relocate face in body X" is
  now e.g. "…nearest candidate face 4: area 100.0 vs 999.0, drift 899 > cap
  499.5".
- ✅ **Phase 3.3 (commit/replay symmetry)** — `_tests/test_split_body_identity.py`
  asserts every split op's `execute()` references `child_body_ids` (the
  structural lock against Bug A — commit splits, replay doesn't), plus
  order-independent `SolidRef` assignment and legacy downstream-anchor recovery.
- ✅ **Phase 3.2 (no bare-index identity)** — same test asserts every split op's
  `_split_commit()` persists `child_solid_refs`, so a commit can't silently
  regress to the fragile positional mapping.

All phases complete. Full suite green (24 files); curzie.vc gated as a fixture.

## 3. Plan

### Phase 1 — Make body identity geometric (kills Bug B class-wide)

Give every split-produced body the same kind of durable fingerprint faces and
edges already have.

1. **New `SolidRef`** (`cad/solid_ref.py`), mirroring `FaceRef`/`EdgeRef`:
   fingerprint = volume + center-of-mass + bbox extents (rounded). `find_in
   (compound) → solid` picks the closest solid above a confidence cutoff;
   returns `None` when ambiguous instead of guessing.
2. **Persist one `SolidRef` per child** at commit, stored positionally aligned
   with `child_body_ids` (new param `child_solid_refs`). Write it in the
   `finalize()` of every split op:
   - `SketchExtrudeOp` (force_new_body) — `op_extrude.py:1217`
   - `FaceExtrudeOp` (force_new_body) — `op_extrude.py:370`
   - `FaceRevolveOp` — `op_revolve_thicken.py:319`
   - `LoftOp` — `op_loft.py:331`
3. **Replay maps by `SolidRef`, not index.** In each op's `execute`, after
   producing the result compound, assign `child_body_ids[i]` = the solid whose
   `child_solid_refs[i]` matches — falling back to index only when refs are
   absent (legacy) or a match is ambiguous.
4. **Add the missing `force_new_body` branch to `SketchExtrudeOp.execute`**
   (Bug A) and audit `LoftOp.execute` for the same gap.

### Phase 2 — Best-effort recovery for already-saved files (incl. curzie.vc)

Files saved before Phase 1 have no `child_solid_refs`. Recover the mapping from
information that *is* in the file: **every child body is referenced downstream,
and those references carry geometry.**

- For a split entry with no `child_solid_refs`, scan downstream entries for the
  first geometric reference into each child body — most directly a
  `FacePlaneSource`/`FaceRef` on that body (entry 5 → `fd4174f2`). The ref's
  world centroid tells us which physical segment that body must be.
- Match each child body to the segment containing its referenced face; assign
  accordingly. For `fd4174f2` this recovers the `(-34.99, 34.99)` segment and
  the chain replays.
- This is a one-time migration on load (bump save `version` to 3, write
  `child_solid_refs` back so the heuristic runs once). Where no downstream
  reference exists for a child, fall back to index order (no worse than today).

### Phase 3 — Invariants & guardrails (stop the *next* one)

The user's instinct is right: the space of interactions is combinatorial, so we
defend with invariants rather than case-by-case patches.

1. **Save/load round-trip test as a gate.** A test that, for a corpus of
   `.vc` files (curzie.vc + synthesized split/fillet/loft/cross-body cases),
   asserts: replay-from-scratch produces the *same* per-body shapes
   (volume + bbox within tol) as the in-memory session that saved them. This
   is the single most valuable artifact — it turns "did we break replay?" into
   a CI signal.
2. **One identity rule, enforced.** Lint/assert that no cross-entry or
   cross-body reference is stored as a bare index without a geometric ref
   beside it. Positional indices may exist only as a *fallback tiebreaker*,
   never as the sole identity.
3. **Replay symmetry checklist** in `op_base` docs: every op whose `commit`
   creates bodies or tags geometry MUST have an `execute` that reproduces the
   same bodies/tags. Add a dev-mode assertion that compares commit-produced
   body set vs. a immediate replay of that single entry.
4. **Fail loud, fail local.** Current cascade behavior (one broken plane nukes
   half the tree) is correct for safety but obscures cause. Add the failing
   ref's fingerprint + nearest candidate to the error so diagnosis is one line,
   not a debugging session.

## 4. Sequencing

1. Phase 3.1 first (round-trip test) — establish the safety net; curzie.vc is
   the first fixture and currently red.
2. Phase 1 (`SolidRef` + persist + replay-by-ref + Bug A branch) — turns the
   test green for *new* saves.
3. Phase 2 (recovery migration) — turns curzie.vc and other legacy files green.
4. Phase 3.2–3.4 — guardrails so regressions are caught at author time.

Net: the immediate file is fixed by Phase 2, but the actual deliverable is
Phase 1's geometric body identity + Phase 3's round-trip gate, which close the
*class* of "works live, breaks on reload" bugs rather than this one instance.
