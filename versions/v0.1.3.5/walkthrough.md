# Walkthrough: Abu Align & Train v0.1.3.5
## 360 Splitter: 3D Trajectory Compounding as Default & Rotation Matrix Resolution

### 1. Executive Summary
Release **v0.1.3.5** resolves the discrepancy between the unrotated nominal rig matrix (`-1 0 0 0 0 -1 0 -1 0`, Screenshot 1) and the full 3D orientation matrix (`-0.99863390335982594 0.03277280884577214 ...`, Screenshot 2).

---

### 2. Why Screenshot 1 and Screenshot 2 Were Different

```
Trajectory Prior (X, Y, Alt, Yaw, Pitch, Roll)
         │
         ├── OLD: override_orientations_from_rig = True
         │        Discards Yaw, Pitch, Roll -> Forces Yaw=0°, Pitch=90°
         │        Result: R = B_back @ Identity = [-1 0 0  0 0 -1  0 -1 0]  (Screenshot 1)
         │
         └── NEW: override_orientations_from_rig = False (Default in v0.1.3.5)
                  Preserves Yaw, Pitch, Roll -> Computes Q_station in SO(3)
                  Result: R = B_back @ Q_station = [-0.9986339... 0.0327728...] (Screenshot 2)
```

1. **Screenshot 1 (`-1 0 0 0 0 -1 0 -1 0`)**:
   - In previous builds of `panoramic_splitter.py`, the parameter `override_orientations_from_rig` was defaulted to `True`.
   - When `True`, the splitter used the $(X, Y, Z)$ position from the trajectory, but **discarded the camera's true heading (`yaw`, `pitch`, `roll`)**, forcing all stations to face pure North ($Yaw = 0^\circ, Pitch = 90^\circ, Roll = 0^\circ$).
   - For the `back` camera, $R = B_{\text{back}} @ I = \begin{bmatrix} -1 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & -1 & 0 \end{bmatrix}$, which formatted as integer string `-1 0 0 0 0 -1 0 -1 0`.
2. **Screenshot 2 (`-0.99863390335982594 0.03277280884577214 ...`)**:
   - ChatGPT's result compounds the **actual 3D rotation of the station** ($Q_{\text{station}}$):
     $$R_{\text{back}} = B_{\text{back}} @ Q_{\text{station}}$$
   - Because the walking rig had a heading of $1.9^\circ$ and a roll tilt of $2.3^\circ$, $Q_{\text{station}}$ is a full floating-point matrix, producing the precise multi-decimal rotation in Screenshot 2.

---

### 3. Changes Implemented in `v0.1.3.5`

1. **Compounding Enabled by Default**:
   `override_orientations_from_rig` now defaults to `False` in both `compound_realitycapture_pose()` and `split_panorama_dataset()`.
2. **CLI Parameter Updated**:
   The CLI now compounds full 3D rotations by default whenever `--dvo-trajectory` is provided. If a user explicitly wishes to flatten orientations, they can supply `--override-orientations`.
3. **17-Digit Float Precision**:
   Updated rotation formatter in `generate_realityscan_xmp()` to `f"{v:.17g}"`, preserving full IEEE 754 precision matching `True_Schema_360_Camera_Rig.xmp`.

---

### 4. Verification

- Ran `tests/test_cubemap_rs.py` — **PASSED (100%)**
- Ran `tests/test_rig_cameras_csv.py` — **PASSED (100%, 10 tests)**
- Ran `tests/test_splitter_masks.py` — **PASSED (100%, 4 tests)**
