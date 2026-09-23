# Abu Align & Train v0.0.4.0 — Minimal rig designer and 3D preview

## Design changes and rationale
The old splitter exposed all source, geometry, optics, and sampling controls at
once. Its viewport drew every large frustum, ray, and label with stippled fills,
which obscured the rig and made selection impossible.

The new 360° Rig Designer separates Source, Rig, and Export into scrollable tabs.
It keeps export actions pinned, uses readable layout/solver names, and shares
the graphite/mint workspace theme. Parameters retain their existing underlying
variables and solver callbacks. Rig summary details collapse by default.
Labels and inputs wrap without sharing grid-column widths across unrelated rows.
Controls remain above the notebook in Tk's stacking order, avoiding hidden cards.

The CPU-only Pillow renderer uses supersampling and smooth translucent coverage.
Camera markers show viewing directions from a shared origin; they do not imply
physical camera baselines. One selected camera is emphasized by default. Click a
marker to select it; independently toggle All coverage, Labels, and Grid. Fit,
Top, Front, and Rotate provide repeatable views. Orbit/pan/wheel interactions
remain; arrows, +/-, F, and bracket keys support keyboard navigation. Polar and
Map overviews remain available. Coverage is an ideal pinhole illustration, not
a lens-distortion or occlusion simulation. Export geometry is unchanged.

Preview basis calculations apply roll simultaneously to right/up axes and are
tested at poles and arbitrary orientations. Selection clamps when camera count
shrinks. Partial invalid numeric edits show a prompt rather than a Tk traceback.
Closing is blocked during export; normal close and Use dataset cancel rotation.
Deferred export failures retain their exception text for the error dialog.

Implementation: rig_designer_ui.py manages presentation and progressive
disclosure; rig_viewport.py provides geometry/rendering; abu_align_gui.py retains
workflow callbacks. version_manager.py now includes both modules in snapshots.

## Verification and performance
In the vggt environment:

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p test_rig_designer.py -v
python -c "import py_compile; [py_compile.compile(f,doraise=True) for f in ['abu_align_gui.py','desktop_ui.py','rig_designer_ui.py','rig_viewport.py','align_cli.py','train_engine.py','opensfm_engine.py','360dvo_engine.py','version_manager.py']]"
python version_manager.py --bump
git diff --check
```

All 16 tests passed in 54.228 seconds. Tests cover the existing workspace,
presets, workflow controls, and telemetry, plus all four rig layouts and three
preview modes, camera picking/clamping, FOV/aspect math, orthonormal rolled
bases, invalid numeric edits, and visible minimum-size controls. The five rig
tests were rerun after the final wrapping adjustment. Native NumPy/Pillow tests
ran outside the sandbox due DLL-loading restrictions. Nine modules compiled.
Desktop visual inspection caught and corrected a clipped adjustments label.
No panorama export, reconstruction, or training job was launched for these tests.

At 890x559 viewport pixels, 30 CPU-only renders after warmup measured:
- Median render: 38.32 ms; maximum: 42.32 ms.
- Retained process RSS: 81.21 MB before, 81.28 MB after (+0.07 MB).
These are local smoke measurements, not a universal frame-rate guarantee or a
peak-memory benchmark. The renderer holds one Tk image; intermediate raster
buffers are released. No GPU models/tensors are initialized by the renderer.
GPU VRAM was not separately benchmarked; existing hardware monitoring remains.

## Release and co-development
Odometer rollover: v0.0.3.9 -> v0.0.4.0. Active and versioned C# launchers rebuilt.
Snapshot: versions/v0.0.4.0/. Walkthrough copies: root, artifacts/, and docs/.
The consolidated repository is canonical; no retired Abu_Align mirror is made.

Antigravity completed its engine commits while this UI work was in progress.
Its v0.0.3.9 commit captured the initial rig UI modules from the shared folder.
This dedicated UI release finishes and validates that work on top of its commit.
Existing engine logic is preserved; CLI/training changes in this commit are
version headers only. The user requested keeping engine work separate.
