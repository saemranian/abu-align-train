# User presets

In the main GUI, configure alignment and training, then click **Save As…** in
the **User Preset** row. Enter a profile name such as `360 Video Tour - Fast`.
Select a saved profile in the dropdown and click **Load** to restore it.
Saving with an existing name asks before replacing the file.

Profiles are UTF-8 JSON files in this directory, independent of the launch
working directory. They include settings for all five alignment engines and
all four training engines, including inactive panels, cache/window settings,
auto-training, and custom DVO model paths. Dataset selection, running jobs,
telemetry, and separate video/splitter dialog settings are excluded.

Files contain `schema_version: 1` and a `settings` object. Boolean settings are
JSON booleans; other settings retain their GUI strings. Loading validates the
schema, complete field set, types, and fixed-choice controls before applying
anything. Invalid or incompatible profiles show an error. Loading during a
pipeline run is blocked; stop the run first.

To share a profile, copy its JSON file into another installation's `presets/`
directory, then reopen the dropdown. Check any custom model path before use.
Personal JSON files are ignored by Git. Saving uses a temporary file and atomic
replacement so an unsuccessful write preserves the previous profile.
