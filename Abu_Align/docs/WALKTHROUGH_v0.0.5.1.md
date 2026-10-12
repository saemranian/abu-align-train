# Abu Align & Train v0.0.5.1 — Explicit execution and inspection controls

## Audit and implementation
Second-pass critique: docs/DESIGN_CRITIQUE_PASS_2.md.
Phase plan and results: docs/DESIGN_PLAN_PASS_2.md. The earlier audit is preserved.

Phase 1 replaces three competing visible launch buttons and the separate automatic
training checkbox with an explicit run-mode selector and one primary action.
Align & train dispatches alignment with force_auto_train=True; Align only uses
skip_auto_train=True; Train only calls the existing training entry point.
The workflow summary reflects the mode. Existing button lifecycle updates drive
the mode-selector lock without a polling timer. Internal legacy button handles
remain for compatibility with engine callbacks. Alignment modes synchronize the
existing auto-training profile field. Train only is a transient explicit mode;
it does not overwrite that stored field or add a new preset schema field.

Phase 2 adds Resume live to log search. Close/Escape keeps follow paused; Resume
live closes search, enables following, and scrolls to the end. Query and raw log
content are preserved. Reopening search restores the current match highlight.

Phase 3 adds a numbered camera selector beside Previous/Next in the Orbit
inspector. It tracks renderer selection and rebuilds choices on rig changes;
selection clamps when the rig shrinks. Exported geometry is unchanged.

## Verification
Per-phase: 11 desktop tests (22.739s), 12 desktop tests (30.369s), 9 rig tests
(24.196s). Commands: python -m unittest discover -s tests -p test_desktop_ui.py -v;
python -m unittest discover -s tests -p test_rig_designer.py -v.
The primary execution area was visually inspected in an isolated desktop preview.
Final result: 25 isolated-snapshot tests passed in 95.664s; all 18 release modules
compiled and matched the staged baseline. GUI/renderer module origins were verified
to exclude working-tree imports. Active-file whitespace checks passed.
The first final run passed 24/25: its layout test sampled a card corner that could
be clipped after focus reveal. The test now samples the visible viewport/card
intersection and uses an on-screen window origin; the complete rerun passed. Engine commands
were mocked; no training/alignment workload or GPU benchmark was started.
No VRAM, RSS, or render-speed improvements are claimed. No new polling timer was
introduced. Full browser model-load and cross-monitor DPI validation remain open.

## Release integrity
Pre-existing edits in seven engine/GUI/renderer files remain separate. Shared
version-only changes are staged from HEAD, and the snapshot is copied from the
intended index rather than capturing unrelated working edits.
Base-10 bump: v0.0.5.0 to v0.0.5.1; launchers rebuilt with csc.exe.
The consolidated-root backup is versions/v0.0.5.1 (no retired duplicate tree).

## Modified files
- desktop_ui.py; rig_designer_ui.py.
- tests/test_desktop_ui.py; tests/test_rig_designer.py.
- docs/DESIGN_CRITIQUE_PASS_2.md; docs/DESIGN_PLAN_PASS_2.md.
- walkthrough.md; artifacts/walkthrough.md; docs/WALKTHROUGH_v0.0.5.1.md.
- VERSION.txt; version-only changes in abu_align_gui.py, align_cli.py, train_engine.py.
- Abu_Align.bat/.exe; Abu_Align_and_Train.bat/.exe; archived v0.0.5.1 launchers.
- versions/v0.0.5.1/ release scripts, launchers, version, audit, and walkthrough.
