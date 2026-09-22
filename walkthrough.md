# Abu Align & Train v0.0.3.6 — Modern desktop workspace

## Design and implementation
The previous UI placed all engine selectors, configuration, actions, hardware
badges, and a large console on one screen. Dense horizontal packing clipped
controls on smaller windows; native light dropdowns and many accent colors
competed with the primary action.

The redesigned workspace uses a graphite palette, a restrained mint accent,
consistent typography, dark dropdowns, clear focus states, and three tabs:
Alignment, Training, and Activity. Dataset and preset controls stay at the top;
progress, primary workflow, Align only, Train only, Pause, and Stop stay pinned
at the bottom. Default window size respects screen dimensions, up to 1440x1050.

Engine selectors use compact segmented controls and alignment descriptions.
All five alignment and four training engines retain their existing variables,
callbacks, tooltips, and detailed controls. Parameter groups reflow by available
width; settings pages support local mouse-wheel scrolling and keyboard focus
reveal. Preset schema remains compatible with v0.0.3.5 and keeps 65 settings.
An empty dataset prompt replaces the hard-coded client folder. Load is disabled
until a preset is selected. Keyboard shortcuts: Ctrl+O, Ctrl+Shift+S, Alt+1/2/3,
and Ctrl+Tab. Buttons support Enter/Space; tooltips support focus and Escape.

Activity opens when a run starts. It groups Results, Models, Logs, the viewer,
stages, and hardware readings. Compact windows prioritize a readable console;
a separate live Monitor remains available. Larger windows show stages and
hardware inline. Follow log can be disabled for reading older messages. The
console is read-only and bounded to about 5,000 lines; full logs stay on disk.
The video extractor, splitter, and documentation dialogs share surface, field,
button, and focus styling while preserving their existing tools and previews.

A semantic bug was corrected: Align only previously respected auto-training.
It now explicitly omits --train. The primary workflow honors the Training tab's
automatic training setting and changes its label accordingly. Existing forced
one-click entry points remain supported.

Presentation lives in desktop_ui.py; engine and process behavior remains in
abu_align_gui.py. version_manager.py now synchronizes and snapshots the new
presentation module. No photogrammetry or training kernel was changed.

## Verification
Commands (vggt conda Python):

```powershell
python -m unittest discover -s tests -v
python -c "import py_compile; [py_compile.compile(f,doraise=True) for f in ['abu_align_gui.py','desktop_ui.py','align_cli.py','train_engine.py','opensfm_engine.py','360dvo_engine.py','version_manager.py']]"
python version_manager.py --bump
git diff --check
```

Ten real-Tk tests passed in 23.299 seconds. Coverage includes all 20 preset
engine combinations, every alignment/training panel at 960x720, 1240x900, and
1600x1000, readable compact Activity logs, profile errors and atomic saves,
workflow command generation without launching GPU jobs, read-only/bounded logs,
and opening/closing the video and splitter dialogs without callback errors.
The splitter test required running outside the sandbox because NumPy's native
DLL could not load inside it. This was an execution restriction, not a package
change. The final dialog-style adjustment was rechecked with the six UI tests.

Seven release modules compile. Both C# launchers rebuilt via csc.exe. Desktop
visual review inspected alignment, training, and activity using the native
Windows computer-use tools. No reconstruction or training benchmark was run.

## Memory and VRAM hygiene
The new UI does not initialize GPU models or allocate GPU tensors. Telemetry
continues to use the existing monitor. Bounding the display log avoids unlimited
Tk text growth; complete diagnostic logging is unchanged. Peak RAM and VRAM
were not benchmarked, and no numerical memory reduction is claimed.

## Release artifacts
Version: v0.0.3.6. Canonical code remains in the consolidated project root.
Snapshot: versions/v0.0.3.6/ (active scripts, engines, launchers, tests, docs).
Historical binary: versions/historical_binaries/Abu_Align_and_Train_v0.0.3.6.exe.
Walkthrough copies: walkthrough.md, artifacts/walkthrough.md, and
 docs/WALKTHROUGH_v0.0.3.6.md. The user guide includes the new navigation and
workflow behavior. The retired Abu_Align mirror has not been recreated.
