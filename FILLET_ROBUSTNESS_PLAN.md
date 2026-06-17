# Fillet: hang root cause + path to a more capable, hang-proof fillet

## The reported hang (filletbug.vc)

Picked edge `Body 1 [3] + Body 1 [2]` = a **partial circular edge, R≈23.175mm**
(arc ~128°) where a **surface-of-revolution** (the revolve body) meets a
**plane** (the extrude body) in the union.

Behavior (authoritative, via the real `_run_fillet`/`fillet_edges` path on mesh
edge 2):

| radius | result            |
|--------|-------------------|
| 4      | builds (0.06s)    |
| 4.5–6.5| fails fast, clean ("1 faulty contour") |
| **7**  | **hangs forever** |
| 7.5    | fails fast, clean |
| 8      | hangs (via heal-retry) |

Key facts:
- The hang is **inside OCCT's `BRepFilletAPI_MakeFillet.Build()`** (the `ChFi3d`
  contour walker). It is C++ holding the GIL — Python `faulthandler` can't even
  fire. **No Python-side timeout can interrupt it.**
- It's **not monotonic in radius** (7 hangs, 7.5 doesn't), so a radius cap can't
  reliably avoid it.
- It's **not the fillet shape mode**: `ChFi3d_QuasiAngular` hangs at r=7 too.
- `fillet_edges` makes it worse: on a failing radius it runs `_heal()` + retry,
  so a near-miss radius pays the cost twice and can hang in either call.

This is a known class of OCCT pathology: rolling-ball fillets on
revolution/spline surfaces at radii where the ball's contact curve becomes hard
to track and the solver iterates without converging or terminating.

## What "more advanced" CAD actually does (and what transfers here)

FreeCAD, Onshape, SolidWorks don't have a magic non-hanging kernel — they wrap a
fillet kernel (OCCT for FreeCAD; Parasolid for Onshape/SW) in **robustness
strategies** and **richer fillet types**. The ones that matter for our hangs:

1. **Bounded, killable execution.** The non-negotiable backstop: run the kernel
   call where it can be *aborted*. OCCT can't be interrupted in-thread, so the
   call must run in a **child process** with a hard timeout; on overrun, kill it
   and report "couldn't fillet at this radius." This is what prevents the app
   from ever freezing, independent of any cleverness below.

2. **Edge subdivision / segmented contours.** Splitting a problem edge into
   shorter sub-edges (and filleting them as a chain) often lets the contour
   walker converge where the full edge stalls. Cheap, high yield on
   arc-of-revolution edges like this one.

3. **Radius search / largest-feasible-fit.** Instead of pass-or-fail at the
   typed radius, probe a few radii (the user's value, then a small bracket
   around it) and apply the largest that *builds within the timeout*. Real CAD
   surfaces "max radius here is ~6.5mm" instead of hanging. Our data shows 6.5
   and 7.5 are fine — a search would have sailed past the r=7 hole.

4. **Setback / variable-radius / face-blend fillets.** The genuine
   capability expansion — vertex setbacks at awkward junctions, variable radius
   along an edge, and face-face blends. These both add features users expect
   *and* dodge the exact corner cases that make constant-radius fillets stall.

5. **Boolean rolling-ball fallback.** When `BRepFilletAPI` refuses, sweep the
   blend surface (pipe/torus along the edge) and boolean it in — the same family
   as this repo's existing chamfer boolean fallback. Heavier and lower-fidelity,
   but rescues cases the standard API can't touch.

The user's intuition is right *with one caveat*: a more capable fillet path will
make **many** currently-failing cases succeed (and thereby stop hanging on
them). But it cannot *guarantee* no hang — OCCT can still spin on an input the
strategies don't rescue. So #1 (killable execution) stays mandatory as the floor;
#2–#5 raise the success rate above it.

## Status

- ✅ **Phase A (stop the freeze)** — `cad/operations/fillet_proc.py`:
  `fillet_edges_isolated()` runs the kernel call in a spawned child process with
  a hard timeout, killing it on overrun and raising `FilletTimeout`. Wired into
  replay (`op_fillet.execute`), commit (`_split_commit.compute`), and preview
  (`vp_fillet3d._launch_fillet3d_thread`, 12s budget). The filletbug.vc r=7 hang
  now returns a bounded timeout instead of freezing.
- ✅ **Phase B (find the buildable radius) — "Fit largest".**
  `fillet_edges_find_max()` binary-searches for the largest radius that builds
  **and tessellates** (`validate_mesh`), anchored to a geometry-derived ceiling
  (idempotent — repeated clicks converge), validated across N independent
  processes. Clicking **Fit largest sets the radius field** to the result (in the
  user's unit) and re-previews — so it's always recoverable by clicking again
  after changing your mind. Commit still **honors the exact typed radius** and
  fails loud (red entry) if it can't build — no silent substitution; borderline
  geometry that passed the search but fails commit surfaces as a red entry rather
  than a swap. (Earlier this was advisory-only/hint; reverted to write-the-field
  per UX feedback.) Preview is ONE bounded attempt at the typed radius (the
  per-keystroke search spawned ~10 procs/keystroke and leaked semaphores →
  lockup; fixed by status-file IPC + single-attempt preview). Edge subdivision
  was prototyped but free-standing sub-edges aren't accepted by `MakeFillet`.
- ✅ **Program-wide progress indicator** (`gui/progress.py` `ProgressController`):
  one status-bar spinner + phase line for heavy OCCT ops & checks, with a Cancel
  button shown for cancelable ops. Fit-largest reports each probed radius
  (`trying radius X`, in the user's unit) and is **cancelable** — Cancel kills the
  in-flight child process (`fillet_edges_isolated` polls `should_cancel` instead
  of one blocking join; raises `FilletCanceled`). `run_op_async`
  (extrude/revolve/loft/boolean) reports `computing…`→`tessellating…`. No
  on-screen overlay (it duplicated the status line).
- ◑ **Phase C (capability expansion)**
  - ✅ **Per-edge radius (Onshape-style)** — each edge row in the fillet panel
    has its own radius spinbox (`SelectionList(per_row_value=True)`); a blank row
    falls back to the Default Radius. `FaceFilletOp.edge_radii` (aligned with
    edge_indices/edge_refs) is plumbed through preview, commit, replay, and
    serialization; the subprocess worker maps each radius to its edge **by
    midpoint** so it survives the BREP round-trip.
  - ✅ **Variable radius `(r1,r2)` taper** — engine-level (`_run_fillet` accepts a
    tuple via OCCT `Add(R1,R2,E)`) and persists as `[r1,r2]` per edge; UI for
    entering a taper per row is the remaining bit (a row holds one spinbox
    today, so tapers round-trip through params but aren't yet editable in-panel).
  - ⏳ Remaining: in-panel taper entry, vertex setbacks, face-blend selection
    mode, boolean rolling-ball fallback.

**Borderline-geometry caveat:** the filletbug edge is unstable for both the
kernel and the tessellator — the same radius non-deterministically builds+meshes
in one process but fails in another near its limit. Subprocess pre-validation
can't perfectly predict the live result; that's why "Fit largest" is advisory.

Gate: `_tests/test_fillet_hang_guard.py` (engine + isolation + fit-largest
determinism + per-edge cases; fixture `_tests/filletbug.vc`). Full suite green.

## Proposed plan

### Phase A — Stop the freeze (backstop, mandatory)
- Run `fillet_edges` in a child process with a hard timeout (default ~20s,
  configurable). On timeout, terminate and raise a clean
  `FilletTimeout("couldn't fillet edge(s) at r=… — try a different radius")`.
- Wire it into both preview and commit. Preview already threads; the process
  wrapper replaces the in-thread kernel call so a hung build is killed, the
  spinner clears, and the panel shows the message instead of locking.
- Reuse the existing preview→commit result cache so we don't pay twice.
- *Gate:* a test that fillets filletbug.vc's edge at r=7 returns a timeout error
  in bounded time instead of hanging.

### Phase B — Make more fillets actually succeed
- **Edge subdivision**: when a single-edge fillet fails/times out, split the
  edge into N arcs and fillet the chain; accept if it builds.
- **Radius search**: on failure at the requested radius, binary-search down to
  the largest radius that builds within the timeout; report what was applied.
  (Opt-in or "auto-fit" toggle so it doesn't silently change the user's number.)
- Both run inside the Phase-A process wrapper, so neither can hang the app.

### Phase C — Capability expansion (the FreeCAD/Onshape parity work)
- Variable-radius fillet (radius per edge endpoint / along edge).
- Vertex setbacks at high-valence junctions (directly targets the faulty-vertex
  failures we see at r=3–8 here).
- Face-blend fillet (select two faces, blend between them).
- Boolean rolling-ball fallback for edges `BRepFilletAPI` rejects outright.

### Sequencing
A first (it's the safety floor and unblocks the reported bug), then B (biggest
success-rate gain for least work), then C as feature investment. A+B together
would turn this exact bug from "app hangs" into "fillet applied at 6.5mm
(7.0 wasn't buildable here)".
```
