# Abu Align and Train

Local tools for camera reconstruction, Manual Control Points, image preparation and Abu 3D Viewer M54. Perspective images and original equirectangular panoramas are supported by the appropriate engines.

## Downloads and current status

[Releases](https://github.com/saemranian/abu-align-train/releases) · [Build status](https://github.com/saemranian/abu-align-train/actions/workflows/release.yml)

**The old v0.1.5.8 downloads are bootstrap installers, not self-contained applications.** Their first run downloads Python and dependencies. They do not establish support for all engines on Mac or Linux.

The replacement pipeline builds a **CPU core candidate** containing its interpreter and native dependencies. Installing or starting this core does not execute pip, Conda, curl or a compiler. Candidates remain drafts until all required jobs pass and the owner qualifies signing, installation and licensing. Do not assume an older download has these fixes.

| Download target | Candidate scope |
| --- | --- |
| Windows x64 EXE | CPU core plus Windows native GPU viewport |
| Linux x64 AppImage | CPU core; shared browser 3D workspace |
| macOS 14+ arm64 DMG | CPU core; shared browser 3D workspace |
| macOS Intel | Deferred: PyCOLMAP 4.2.0 publishes no Intel Mac wheel; no installer claimed |

The core includes Manual Control Points and CPU refinement, native spherical COLMAP, M54, panorama/image utilities and diagnostics. **Gaussian training runtimes, neural checkpoints and FFmpeg are not bundled.** Other engines remain external integrations. A Mac installer does not make CUDA engines compatible with Metal.

## Use

Once a qualified candidate is available, download the file for your system, install it and open Abu. Choose **Tools > Distribution and engines** for the exact capabilities and restrictions. Existing developer installations can retain their configured engines. Projects, controls and source geometry stay local; no client imagery belongs in public build artifacts.

Developers can automate the installed runtime with `--doctor`, `--m54 PATH`, or the bundled allowlisted CLI workers. A source Python installation exposes `abu-align`, `abu-gui`, `abu-m54` and `abu-doctor`; source installation is optional, not an end-user prerequisite. External engine runtimes can be selected through `ABU_ENGINE_PYTHON` and must be independently compatible and licensed.

## Licensing

The source has **mixed licenses**. [LICENSE.txt](LICENSE.txt) retains the VGGT custom license for material under that grant. It cannot be replaced by an Apache badge. See [third-party scope and primary sources](THIRD_PARTY_NOTICES.md) and the [engine catalog](installer/engines.json). VGGT-Omega and VGG-T3 have noncommercial terms; ACE0 additionally restricts redistribution to source form; PFGS360 and its depth stack require further clearance. Free pricing and automatic model downloads do not remove those restrictions.

## Releases and contributors

See [release procedure](docs/RELEASE_GUIDE.md) and [implementation audit](docs/DISTRIBUTION_AUDIT.md). The publisher exports an explicit public source allowlist, preserves public ancestry, rejects existing versions and never force-pushes. New tags build tested draft candidates. Historical private backups, logs, credentials, model caches and client datasets are excluded from the export. Existing public history is not silently rewritten.
