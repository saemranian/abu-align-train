# v0.0.7.9 — UI interaction and viewport polish

## Changes and rationale
- Rig toolbar responds to changing button labels, wraps with consistent spacing,
  and avoids clipped or overlapping expanded controls.
- Source-path fields yield width to Folder, Video, and Browse actions so these
  buttons remain fully visible at minimum size.
- Controls help floats over the viewport instead of shrinking the 3D scene.
  Escape dismisses help before restoring settings from focus view.
- Log search refreshes counts after appending, trimming, and clearing output.
  Updates are coalesced over 150 ms and preserve reading position and follow state.
- Settings pages respect Windows wheel magnitude and fractional deltas; zero
  movement is ignored and scrolling does not change option selections.
- Layout coverage now includes VGG-T3 alongside the other five alignment engines.

See docs/DESIGN_REVIEW_v0.0.7.9.md for critique, phased implementation, acceptance
criteria, and remaining limitations.

## Verification
- Desktop, rig, and preset suite: 29 tests passed in 96.550 seconds.
- Isolated release snapshot: the same 29 checks passed in 112.951 seconds.
- After the final source-row correction: all 11 affected rig checks passed in
  48.825 seconds (30 distinct UI/preset checks covered across these runs).
- All 20 snapshot Python modules compiled; source and launcher parity verified.
- Tested desktop sizes: 960x720, 1240x900, 1600x1000; rig: 1120x760.
- Real-window inspection identified viewport compression with both help and
  display open. Regression checks verify help no longer reduces scene height.
- Window hit-testing initially failed because another application obscured the
  preview; the isolated visibility test now keeps its own dialog on top.
- Commands: python -m unittest discover -s tests -p test_desktop_ui.py -v;
  repeat for test_rig_designer.py and test_gui_presets.py.
- No alignment or training jobs were launched. No VRAM or inference performance
  benchmark was performed. The existing CPU rig renderer is retained; the log
  buffer remains bounded to approximately 5,000 lines.

## Release and collaboration
Base-10 release: v0.0.7.8 -> v0.0.7.9. C# launchers rebuilt by version_manager.py.
Canonical source stays in the consolidated project root; backup: versions/v0.0.7.9/.
Antigravity commit 8957ca8 collected shared UI edits along with its engine fixes
while verification was running. This release preserves that history and records
UI verification separately. Earlier v0.0.7.8 engine notes remain in its snapshot.

## Files
Functional UI: abu_align_gui.py, desktop_ui.py, rig_designer_ui.py.
Regression coverage: tests/test_desktop_ui.py, tests/test_rig_designer.py.
Release metadata: VERSION.txt, both BAT launchers, version headers
in GUI/CLI/training files, rebuilt EXEs, release documentation, and snapshot.
