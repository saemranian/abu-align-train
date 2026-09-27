# v0.0.9.7 — Horizontal orbit direction

Desktop workspace horizontal orbit used positive mouse delta for yaw, producing
the opposite left/right feel to the browser viewer. It now subtracts horizontal
mouse delta. Vertical orbit, sensitivity, zoom and pan are unchanged.
Added regression coverage for right drag, reverse drag, unchanged pitch during
horizontal drag, and Pan mode retaining yaw while translating the view.

Verification: python -m unittest discover -s tests -p test_native_inspection.py -q
12 tests passed in 5.699 seconds. No production dataset or GPU benchmark was run.
No additional buffers or GPU allocations; VRAM/FPS not measured for this input fix.

Changed implementation: workspace_3d.py. Tests: tests/test_native_inspection.py.
Release headers, VERSION.txt, launchers and documentation synchronized.
Backup: versions/v0.0.9.7/ in the consolidated project root.
