# v0.0.9.3 — Focused 3D inspection workflow

## Design critique and implementation
The viewers exposed too many competing visual elements and actions. Native photo
selection interrupted navigation, inspector actions could overflow horizontally,
and panel buttons did not communicate whether their panels were open. Browser
photo planes obscured geometry, and changing point color reset comparison framing.

Phase 1 — predictable comparison: implemented Switch A / B in both viewers without
moving the viewpoint. Native viewport shortcut B provides the same action; Tab
retains keyboard focus navigation. Corrected reversed native A/B selection callbacks:
A selection loads the reference; B selection activates and fits comparison.

Phase 2 — quieter presentation: photo planes are opt-in in the browser; native
photo popups are opt-in, consistent with browser previews. Photo availability is
concise and the browser View photo action is disabled when no photo is available.
Navigation action labels use plain text. Native point sizes explicitly show px.

Phase 3 — responsive controls: native camera selection gets its own row and actions
wrap below it. Compare/Inspect buttons report their open state with Hide labels.
Browser point-color changes preserve the viewpoint and chosen A/B mode; distance
coloring activates the overlay without resetting navigation.

## Verification
40 tests passed in 55.571 seconds across test_viewer_workspace,
test_native_inspection and test_desktop_ui. New regression cases verify fixed-view
A/B switching, quieter defaults, photo availability, and native inspector bounds.
Commands: python -m unittest discover -s tests -p test_viewer_workspace.py
(repeat for test_native_inspection.py and test_desktop_ui.py).
Browser synthetic preview confirmed Compare shows 70,000 combined points and
Switch A / B returns to 35,000 reference points with the same framing. The control
panel was visually inspected in the narrow in-app browser. No production dataset
or calibrated visual accuracy benchmark was used.

## Memory and release
No training/alignment or GPU benchmark was run; VRAM/FPS metrics are unmeasured.
Disabling photo planes by default avoids background thumbnail work until requested.
Source geometry is unchanged. Existing preview budgets and two-run caches remain.
Modified: native_inspection.py, viewer_workspace.py, their regression tests,
version headers, VERSION.txt, launchers and release notes.
Consolidated backup: versions/v0.0.9.3/.

Concurrent engine edit: vggttt_engine.py changed during packaging. It is excluded
from this UI commit; the backup uses its committed baseline. All other Python
modules and release assets were verified against the workspace.
