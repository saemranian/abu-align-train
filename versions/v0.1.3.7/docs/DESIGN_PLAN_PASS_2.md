# Second-pass implementation plan

## Phase 1 — Explicit execution intent
Introduce a run-mode selector and one primary launch action. Synchronize existing
auto-training profile values; disable mode changes while the engine owns a run.
Retain internal legacy button references for established lifecycle callbacks.
Acceptance: each mode dispatches only its correct callback, profiles remain
compatible, disabled run controls prevent changes, and all layout tests pass.
Status: complete — 11 desktop checks passed (22.739s).

## Phase 2 — Exit log inspection clearly
Add Resume live to log search. Closing search alone retains the user's paused
follow state; Resume live explicitly restores it and scrolls to the newest line.
Acceptance: query and raw output preserved, follow enabled only by the explicit
action, and reopening search restores the current highlighted match.
Status: complete — 12 desktop checks passed (30.369s).

## Phase 3 — Direct camera selection
Add a numbered selector to the Orbit inspector. Synchronize it after render and
selection changes, rebuild choices when the rig changes, and prevent stale indices.
Acceptance: direct selection, mouse selection, and rig shrink remain consistent;
export geometry stays identical and the minimum-size inspector remains usable.
Status: complete — 9 rig checks passed (24.196s).

## Phase 4 — Isolated release
Bump the odometer, rebuild launchers, snapshot only the intended commit baseline,
run the suite with imports isolated from working changes, compile release modules,
update this record, and commit. Preserve unrelated edits in the working tree.
Status: complete — 25 isolated-snapshot tests passed (95.664s); 18 modules compiled.

Deferred: draggable rig split, engine progress-event redesign, full browser model
validation, user research, and cross-monitor DPI certification.
