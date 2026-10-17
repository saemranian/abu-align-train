# Abu Align & Train v0.0.4.9 — Consistent, minimal workspace and tools

## Design and behavior
The previous UI mixed dense legacy tool controls with the redesigned workspace.
This release reduces visual repetition and applies the same graphite/mint treatment
across the workspace, tool dialogs, monitor, help reader, and browser inspector.

- Main workspace: Prepare input menu replaces two competing profile-row actions;
  Dataset/Profile use sentence case, spacing is tighter, descriptive engine cards
  wrap at narrow widths, detailed engine explanations move to accessible tooltips,
  settings groups have subtle boundaries, and notebook border states are unified.
- Fields: consistent text sizing, selection, focus, and disabled treatments.
  Tooltip timers cancel on control destruction and covered controls do not show
  delayed hover tips. Video input receives keyboard focus when opened.
- Video frames: a genuine scrolling settings panel with pinned action buttons,
  stacked label/input pairs, shorter labels, readable sampling choices, and
  contextual FPS/stride/JPEG/blur fields. Escape and window close honor an active
  extraction. Deferred errors preserve their message after the exception scope.
- Rig designer: Display reveals optional overlays; Orbit-specific controls are
  disabled in Polar/Map and the camera inspector hides there. The toolbar wraps
  at narrow widths. Settings are true children of their scroll pages.
- Hardware monitor: six balanced metric cells replace an unstructured text list.
- Help: unified reading/selection surfaces, first-match reset on query changes,
  Enter/Shift+Enter navigation, and highlight removal when the query is cleared.
- Browser scene inspector: shared mint branding, concise labels, reduced branding
  clutter, and camera frustums hidden initially (available through Display).
- Dataset feedback: shorter actionable text and transforms-only datasets correctly
  described as existing reconstruction input.

## Root causes and verification
Final result: all 19 tests passed in 54.026 seconds. All 18 snapshot Python modules
compiled and matched the active sources byte-for-byte. Whitespace checks passed for active-file changes. The byte-identical snapshot
retains pre-existing engine whitespace; it is excluded from that style check.

Visual inspection found settings drawn over pinned actions. Tk geometry placement
alone does not change widget ancestry; the older sibling-hosted controls could
escape the intended clipping boundary. Both tools now construct controls inside
their actual scrolling container. The video focus regression checks an off-screen
field is revealed inside the canvas after receiving focus.

The test cleanup also canceled child-owned timers through the root, causing double
Tcl callback deletion. It now cancels timer events while each widget retains
responsibility for deleting its own registered commands.

Commands: `python -m unittest discover -s tests -v`; compile all snapshot Python
modules with `py_compile`; `git diff --check`; compare snapshot files with active
release sources. Tests cover all engine pages, three workspace sizes, minimum-size
rig layout, video field states and keyboard scrolling, view-mode controls, camera
geometry/selection, focus mode, presets, logs, and mocked pipeline commands.
Desktop screenshots inspected: workspace, input menu, video dialog, and rig designer.
Browser inspector changes were source-reviewed and compiled; no browser model-load
or GPU rendering benchmark was performed.

## Resources and release
No alignment/training jobs or model loads were launched. Rig preview remains on
CPU. No VRAM/RSS/render-speed measurements or memory savings are claimed in this
pass. Existing arrays, model execution, and exported rig geometry are unchanged.
The snapshot includes fast_viewer.py, which the prior snapshot list omitted.
Launchers rebuilt with csc.exe through version_manager.py --bump.
Following the consolidated handover, active files remain at the canonical root;
backup is versions/v0.0.4.9, without recreating the retired Abu_Align duplicate.

## Modified files
- UI: abu_align_gui.py, desktop_ui.py, rig_designer_ui.py, fast_viewer.py.
- Release support: version_manager.py, VERSION.txt; version headers only in
  align_cli.py and train_engine.py.
- Tests: tests/test_desktop_ui.py, tests/test_gui_presets.py, tests/test_rig_designer.py.
- Documentation: walkthrough.md, artifacts/walkthrough.md,
  docs/WALKTHROUGH_v0.0.4.9.md, docs/ABU_ALIGN_USER_GUIDE.md.
- Launchers: Abu_Align.bat/.exe, Abu_Align_and_Train.bat/.exe, and the two
  v0.0.4.9 executables in versions/historical_binaries/.
- Complete release backup: versions/v0.0.4.9/ (scripts, launchers, version, guides).
