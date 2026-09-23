# Abu Align & Train v0.0.4.7 — Workspace clarity and focused rig inspection

## Changes and rationale
Engine choices now explain their role directly on the selection cards. A live
workflow summary makes the alignment/training combination visible before launch.
It collapses below 820px window height so the activity log remains usable.
Buttons gain a restrained hover outline; the header uses less vertical space.

The rig designer opens Source first when no input exists. Focus view hides the
settings column and expands the preview; Show settings or Escape restores it.
Previous/Next cycles camera selection with wraparound and switches to Orbit.
A compact inspector shows selected camera index, yaw, pitch, and field of view.
The preview avoids duplicate optics values and keeps coverage interpretation clear.
Details now explicitly changes to Hide details when expanded. Export geometry,
solver variables, and engine execution are unchanged by this UI work.

## Verification
Final verification against the exact release snapshot: all 17 tests passed in
29.576 seconds; all 17 release Python modules compiled successfully. Command:
`python -m unittest discover -s ../../tests -v` from versions/v0.0.4.7.

Earlier diagnostic runs:
- `python -m unittest discover -s tests -v`: 16 of 17 passed; a window hit-test
  failed during overlapping preview instances.
- `python -m unittest discover -s tests -p test_rig_designer.py -v`: all 6 passed
  in isolation (12.263 seconds), including minimum-size visibility and the new
  focus/camera-navigation regression. Focus restores controls and preserves rig data.
- The desktop suite covered 960x720, 1240x900, and 1600x1000 layout, workflow
  callbacks, read-only logs, presets, and tool dialogs.
- Live screenshot inspection was unavailable: computer-use app approval timed out.
- No GPU jobs or model loads were started. Preview remains CPU-only, with one
  current Pillow image and no polling introduced. VRAM/RSS and render-time
  benchmarks were not measured in this pass; no memory reduction is claimed.

## Release coordination
Antigravity committed these UI changes with its v0.0.4.6 engine release in
42ead79 during verification. This follow-up records the UI release and snapshot.
Concurrent uncommitted ACE0/CLI work is intentionally excluded from this commit
and snapshot. Those working files are preserved. Only their version headers are
staged from the committed baseline.

The canonical root is the single source of truth; the consolidated handover
supersedes the retired duplicate Abu_Align directory. Snapshot: versions/v0.0.4.7.
Launchers are rebuilt using version_manager.py and csc.exe.
