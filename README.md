<div align="center">

<h1>Abu Align & Train</h1>
<h3>End-to-End Neural Photogrammetry, 360° Spherical Rig Processing & 3D Gaussian Splatting Studio</h3>

<p>
  <a href="https://github.com/saemranian/abu-align-train/releases/latest">
    <img src="https://img.shields.io/github/v/release/saemranian/abu-align-train?label=Latest%20Release&style=for-the-badge&color=70dbc0" alt="Latest Release">
  </a>
  <a href="https://github.com/saemranian/abu-align-train/actions/workflows/release.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/saemranian/abu-align-train/release.yml?label=Installers%20Build&style=for-the-badge" alt="Build Status">
  </a>
  <a href="THIRD_PARTY_NOTICES.md">
    <img src="https://img.shields.io/badge/License-Apache%202.0%20%2F%20Open%20Source-38bdf8?style=for-the-badge" alt="License">
  </a>
</p>

</div>

---

## One-Click Download (Single Installation File)

No manual terminal commands, Conda setups, or complex dependencies required. Download the **single installation file** for your operating system and launch the studio immediately:

| Operating System | Single-File Installer Download | Instructions |
| :--- | :---: | :--- |
| **Windows 10 / 11 (64-bit)** | [**Download Windows Setup (.exe)**](https://github.com/saemranian/abu-align-train/releases/latest/download/Abu_Align_and_Train_Setup_Windows_x64.exe) | Double-click the `.exe` installer and launch **Abu Align & Train** from your Desktop. |
| **macOS (Apple Silicon & Intel)** | [**Download macOS Disk Image (.dmg)**](https://github.com/saemranian/abu-align-train/releases/latest/download/Abu_Align_and_Train_macOS_Universal.dmg) | Open the `.dmg`, drag **Abu Align & Train.app** to `Applications`, and launch. |
| **Linux (Ubuntu / Debian / Fedora)** | [**Download Linux AppImage (.AppImage)**](https://github.com/saemranian/abu-align-train/releases/latest/download/Abu_Align_and_Train_Linux_x86_64.AppImage) | Mark executable (`chmod +x`) or double-click to run directly—no root required. |

> Looking for all versions and release notes? Visit the [**Official Releases Page**](https://github.com/saemranian/abu-align-train/releases).

---

## What is Abu Align & Train?

**Abu Align & Train** unifies state-of-the-art neural photogrammetry, spherical $S^2$ structure-from-motion, 360° camera rig splitting, AI segmentation masking, and 3D Gaussian Splatting (3DGS) training into a single desktop studio:

1. **Multi-Engine Photogrammetry & Alignment**:
   - **VGGT-Omega / VGG-T3**: Feed-forward transformer 3D reconstruction in seconds.
   - **Mapillary OpenSfM**: True spherical $S^2$ structure-from-motion for 360° equirectangular panoramas.
   - **360DVO**: Direct visual odometry for continuous 360° video walks.
   - **NVIDIA PyCuSFM & COLMAP**: GPU-accelerated feature matching and bundle adjustment.
   - **Niantic ACE0**: Coordinate regression relocalization *(non-commercial research module)*.
2. **360° Rig Designer & Cubemap Splitter**:
   - Virtual multi-camera rig extraction with locked $90^\circ$ / custom FOV projections.
   - Full 6-DoF trajectory prior compounding (`reconstruction.json`, `transforms.json`, `cameras.csv`, `images.txt`).
   - Exact RealityCapture / RealityScan `.xmp` sidecar and `cameras.csv` generation.
3. **AI Preprocessing & Quality Control Suite ("Abu Tools")**:
   - **SAM 2 Masker**: Zero-shot AI tripod/nadir/operator removal across image sequences.
   - **Blur Drop**: Multi-metric sharpness scoring (Laplacian, Tenengrad, FFT, CPBD).
   - **Frame Picker & Video Extractor**: High-speed keyframe extraction.
   - **M54 Control Points Workspace**: Interactive multi-view control point tie-point registration & 3D verification.
4. **3D Gaussian Splatting (3DGS) Training & Live 3D Inspection**:
   - Direct integration with **LichtFeld Studio** (Vulkan + GUT), **Nerfstudio Splatfacto**, **NVIDIA 3DGRUT**, and **PFGS360**.
   - Built-in **Native 3D Workspace** & **Viser WebGL Viewer** with A/B run registration and residual inspection.

---

## Developer & AI Agent Release Workflow

For contributors or AI coding agents updating this repository, publishing a new release with freshly compiled Windows, macOS, and Linux installers is fully automated:

```bash
python push_to_github.py "Brief summary of your changes"
```
This single command increments the base-10 version (`VERSION.txt`), updates `walkthrough.md`, recompiles the launcher binaries, commits, creates a version tag, and pushes to GitHub—automatically triggering **GitHub Actions** to build and attach the new `.exe`, `.dmg`, and `.AppImage` installers to GitHub Releases.

---

## License & Citations

**Abu Align & Train** is distributed for free under the **Apache 2.0 License** (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE.txt](LICENSE.txt) for upstream terms).

If you use the VGGT neural backbone in academic work, please cite:
```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
