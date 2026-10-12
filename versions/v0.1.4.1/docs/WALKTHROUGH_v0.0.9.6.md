# v0.0.9.6 — Shared visual language and minimal icons

## Review and changes
Added locally rendered, cached monochrome line icons in ui_icons.py. Text labels
remain visible. Main tabs, section headings, shared buttons and tool dialogs now
use consistent icons; browser viewer actions use built-in Viser/Tabler icons.
Shared button icons follow disabled state and dynamic labels. Disabled buttons
use an arrow cursor. Entry/checkbox focus colors and disabled text follow the
shared dark palette. Reduced button/tab padding accommodates icons.
Visual inspection found clipped engine descriptions. Cards now use wider minimum
columns and wrap text within their actual bounds. The app heading retains its A
brand mark without a redundant section icon.

## Verification
Initial desktop/native/preset suite: 32 tests passed in 58.243 seconds.
Final desktop suite: 17 passed in 58.908 seconds; rig suite: 11 passed in 49.023
seconds; browser viewer suite: 15 passed in 1.031 seconds.
Commands: python -m unittest discover -s tests -p test_desktop_ui.py
(repeat for test_rig_designer.py and test_viewer_workspace.py).
Tests cover responsive layouts, dialog construction, icon reuse, disabled states,
text retention and tab icons. Isolated desktop preview visually inspected using
the computer-use skill; post-fix card wrapping verified by layout tests.
Viser produced a non-failing unclosed event-loop warning during test cleanup.

## Memory and release
Icons are cached per Tk root/type/color/size. No new dependencies or downloads.
No training/alignment, production FPS or VRAM benchmark was run.
Changed sources: desktop_ui.py, ui_icons.py, viewer_workspace.py,
version_manager.py, tests/test_desktop_ui.py. Release headers, launchers,
VERSION.txt and documentation synchronized. No engine behavior changed.
Backup: versions/v0.0.9.6/ with ui_icons.py included.
