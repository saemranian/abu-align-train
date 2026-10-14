# Walkthrough: Abu Align & Train v0.1.4.7
## Persistent Image Grid, A/B Reconstruction Comparison and Illustrated Control-Point Help

### Overview
Release `v0.1.4.7` introduces persistent multi-window image navigation, embedded A/B 3D reconstruction comparison, and an interactive illustrated Help system for spherical control points:
1. **Persistent Image Grid**: Opening an image thumbnail now keeps the grid as an owned utility window above the editor, preserving destination pane, selection, geometry, filter, and scroll position.
2. **3D Reconstruction Comparison Sidebar**: Added a collapsible comparison sidebar beside the GPU navigator. Supports A (current editable alignment) vs. B (read-only JSON/PLY/run reconstruction) with A/B/overlay modes, layer/RGB/sampled distance coloring, P95 legend, camera offset lines, and rigid/similarity fitting.
3. **Background Worker Isolation**: Parsing reconstruction B is isolated in a child process; numeric comparison runs on background workers with generation checks and graceful cancellation.
4. **Searchable Illustrated Help**: Added a searchable illustrated guide with seven task topics, locally drawn diagrams/icons, responsive cards, and direct actions opening tools/settings.

### Verification
- 60 regression tests passed in 53.76 seconds in local environment.
- Verified transform conventions, COLMAP/PLY/native loaders, and disconnected-component rejection.
- Two 593-camera models verified with all camera names matching and finite metrics.
