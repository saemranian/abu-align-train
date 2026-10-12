# Abu Align & Train v0.0.5.0 — Design audit and phased UX corrections

## Rationale and changes
The candid audit in docs/DESIGN_CRITIQUE.md identifies excess persistent chrome,
unsearchable logs, training-field overflow, and hidden viewport interactions.
docs/DESIGN_IMPLEMENTATION_PLAN.md records the phased scope and verification.

Phase 1: profile editing is disclosed on demand with inline guidance. Shorter
headings make room for settings. Training fields wrap as complete labeled units.
Phase 2: Activity gains Ctrl+F / Find, case-insensitive next/previous search,
wraparound, match counts, empty and not-found states, and Escape to close.
Searching pauses Follow log; the field tooltip explains how to resume it.
The Activity toolbar wraps to keep its destinations accessible at narrow widths.
Phase 3: the rig Controls guide reveals pan/orbit/zoom/reset/selection gestures
without permanently covering the viewport or changing exported camera geometry.

## Verification and limitations
Per-phase tests passed: 9 desktop checks for phase 1, 10 for phase 2, 8 rig
checks for phase 3. Commands: python -m unittest discover -s tests -p
test_desktop_ui.py -v; python -m unittest discover -s tests -p test_rig_designer.py -v.
Final isolated release-snapshot verification: 22 tests passed in 41.399s; all
18 Python modules compiled and matched the staged baseline. The final runner
removed the working root/versions parent from sys.path after importing the snapshot
GUI, then asserted all GUI/renderer module origins remained inside the snapshot.
Active-file whitespace checks passed; historical whitespace is preserved in copies.
Audit evidence comes from prior desktop screenshots and current source inspection;
no fresh screenshot or browser-model rendering validation is claimed this pass.
No alignment/training jobs were started; no VRAM, RSS, or render-speed benchmarks
were measured. Search operates only on the retained text buffer and adds no timer.
Execution-mode redesign, job-event contracts, rig pane resizing, and full browser
model-load validation are explicitly deferred, not represented as completed.

## Release integrity
Pre-existing uncommitted engine, GUI, and rig renderer edits are excluded. Their
working files are preserved. Shared GUI/CLI/training version changes are staged
from the committed baseline; the snapshot reproduces the intended commit.
The base-10 odometer advances from v0.0.4.9 to v0.0.5.0. Launchers rebuilt with
csc.exe via version_manager.py --bump. Consolidated backup: versions/v0.0.5.0.

## Modified files
- desktop_ui.py; rig_designer_ui.py.
- tests/test_desktop_ui.py; tests/test_rig_designer.py.
- docs/DESIGN_CRITIQUE.md; docs/DESIGN_IMPLEMENTATION_PLAN.md.
- walkthrough.md; artifacts/walkthrough.md; docs/WALKTHROUGH_v0.0.5.0.md.
- VERSION.txt; version-only updates to abu_align_gui.py, align_cli.py, train_engine.py.
- Abu_Align.bat/.exe; Abu_Align_and_Train.bat/.exe; historical v0.0.5.0 executables.
- versions/v0.0.5.0/ scripts, launchers, version, and audit/release documents.
