# Abu Align and Train

Local tools for camera reconstruction, Manual Control Points, image preparation and Abu 3D Viewer M54. Perspective images and original equirectangular panoramas are supported by the appropriate engines.

## Downloads and current status

[Releases](https://github.com/saemranian/abu-align-train/releases) · [Build status](https://github.com/saemranian/abu-align-train/actions/workflows/release.yml)

**The old v0.1.5.8 downloads are bootstrap installers, not self-contained applications.** Their first run downloads Python and dependencies. They do not establish support for all engines on Mac or Linux.

The replacement pipeline builds a frozen desktop plus a **Windows NVIDIA GPU candidate**. Its single online installer automatically acquires pinned CUDA alignment and native panorama Gaussian training components. Users need no Python, pip, Conda, compiler or CUDA toolkit setup. Internet is required during Windows GPU setup; training requires an NVIDIA SM75+ GPU and CUDA 13 compatible driver (R580+). Check each release's capability manifest: older binaries do not gain these features when source changes. Candidates remain drafts until their build and installation checks pass; signing/notarization remain stable-release gates.

| Download target | Candidate scope |
| --- | --- |
| Windows x64 EXE | Desktop, native GPU viewport, managed CUDA panorama alignment and native GUT Gaussian training |
| Linux x64 AppImage | CPU core; shared browser 3D workspace |
| macOS 14+ arm64 DMG | CPU core; shared browser 3D workspace |
| macOS Intel | Deferred: PyCOLMAP 4.2.0 publishes no Intel Mac wheel; no installer claimed |

The desktop includes Manual Control Points and CPU refinement, native spherical COLMAP, M54, panorama/image utilities and diagnostics. Windows GPU setup supplies COLMAP CUDA extraction/matching and the qualified LichtFeld trainer, with matching source and notices. Mapping uses CPU Ceres. Original panoramas stay equirectangular; no cubemaps. Neural research checkpoints and the standalone FFmpeg video CLI are not bundled. Other engines remain external integrations. Linux/macOS packages currently supply the CPU desktop; CUDA training is not available on macOS. First Windows training startup can spend minutes compiling CUDA kernels; cached runs are faster. See [tested scope and limitations](docs/DISTRIBUTION_FINAL_REVIEW.md).

## Use

Once a qualified candidate is available, download the file for your system, install it and open Abu. Choose **Tools > Distribution and engines** for the exact capabilities and restrictions. Existing developer installations can retain their configured engines. Projects, controls and source geometry stay local; no client imagery belongs in public build artifacts.

Developers can automate the installed runtime with `--doctor`, `--m54 PATH`, `--align` or `--train` (use `--help` for dispatch). A source Python installation exposes `abu-align`, `abu-gui`, `abu-m54` and `abu-doctor`; source installation is optional, not an end-user prerequisite. External engine runtimes can be selected through `ABU_ENGINE_PYTHON` and must be independently compatible and licensed.

## Licensing

The source has **mixed licenses**. [LICENSE.txt](LICENSE.txt) retains the VGGT custom license for material under that grant. It cannot be replaced by an Apache badge. See [third-party scope and primary sources](THIRD_PARTY_NOTICES.md) and the [engine catalog](installer/engines.json). VGGT-Omega and VGG-T3 have noncommercial terms; ACE0 additionally restricts redistribution to source form; PFGS360 and its depth stack require further clearance. Free pricing and automatic model downloads do not remove those restrictions.

## Releases and contributors

See [release procedure](docs/RELEASE_GUIDE.md) and [implementation audit](docs/DISTRIBUTION_AUDIT.md). The publisher exports an explicit public source allowlist, preserves public ancestry, rejects existing versions and never force-pushes. New tags build tested draft candidates. Historical private backups, logs, credentials, model caches and client datasets are excluded from the export. Existing public history is not silently rewritten.
