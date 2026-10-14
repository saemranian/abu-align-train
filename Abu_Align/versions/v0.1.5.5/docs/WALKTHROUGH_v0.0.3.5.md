# Abu Align & Train v0.0.3.5 — User Presets

## Change and rationale
The GUI previously held configuration only in Tk variables, requiring users to
re-enter engine options for each workflow. The Project & Dataset Workspace now
has a User Preset dropdown, Load button, and Save As button. Profiles are clean,
UTF-8 JSON files under the canonical project presets/ directory.

All 65 alignment/training configuration variables are captured before legacy
aliases and transient UI variables are created. The dataset remains selected
when a profile is loaded. Separate video/splitter dialogs are outside this scope.
The schema records genuine JSON booleans and the existing GUI string values.
The loader validates schema version, complete field set, types, and fixed-choice
control values before mutation. Custom editable values retain existing GUI
semantics. Engine traces and explicit panel refreshes restore visible panels.

Names are checked for Windows filename restrictions and directory containment.
Existing files require overwrite confirmation. Atomic replacement preserves the
previous file if writing fails. Errors appear in dialogs. Loading is blocked
during active alignment/training so pending handoff settings cannot change.
The dropdown rescans on opening, including profiles copied from another install.
Personal profiles are Git-ignored because they can include local model paths.

## Verification
Run in the vggt conda environment:

```powershell
python -m unittest discover -s tests -p test_gui_presets.py -v
python -c "import py_compile; [py_compile.compile(f,doraise=True) for f in ['abu_align_gui.py','align_cli.py','train_engine.py','opensfm_engine.py','360dvo_engine.py']]"
git diff --check
python version_manager.py --bump
```

Four integration tests passed in 3.669 seconds using real Tk widgets and an
isolated temporary preset directory. Round trips cover all 20 combinations of
five alignment and four training engines. Coverage includes booleans, inactive
settings, model paths, dataset preservation, legacy aliases, Unicode names,
invalid JSON/schema/types/choices, missing files, overwrite cancellation,
successful overwrite, simulated atomic replacement failure, temporary-file
cleanup, filename rejection, dropdown discovery, and the active-run guard.
All five required core modules compiled successfully. Whitespace check passed.
Both requested C# launcher binaries were rebuilt successfully with csc.exe.
No reconstruction or training performance benchmark was run for this GUI change.

## Memory and VRAM hygiene
Preset operations use CPU-only JSON I/O and existing Tk variables; they do not
load any alignment/training model or allocate GPU tensors. Hardware telemetry
was mocked during integration tests. GPU VRAM and process peak RAM were not
measured; no GPU performance or memory improvement is claimed.

## Release layout
The consolidated handover supersedes the retired Abu_Align mirror layout.
Canonical files remain in the project root. The release snapshot is
versions/v0.0.3.5/; versioned launcher is also archived in
versions/historical_binaries/Abu_Align_and_Train_v0.0.3.5.exe.
Walkthrough copies live at walkthrough.md, artifacts/walkthrough.md, and
docs/WALKTHROUGH_v0.0.3.5.md. Usage instructions are in presets/README.md.
