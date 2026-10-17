# Distribution implementation audit and corrections

Audit date: 17 October 2026. Baseline: v0.1.5.8, commit 3a2893edb92a90db408fe2b68e812453d8ea7af4. Compared the Antigravity distribution strategy, implemented source, public repository metadata and the earlier Codex distribution report. Final verification results are recorded below and in walkthrough.md.

## Verdict on the previous implementation

The repository and three release assets existed. The architecture ideas were useful, but the implementation did not deliver the standalone runtime or verified platform support described in the plan. The plan's “100 percent legal” conclusion was unsupported.

The public GitHub API reported v0.1.5.8 as a stable release with a 3,098,646 byte Windows EXE, a 1,521,272 byte DMG and a 1,907,192 byte AppImage. Source inspection established that these were source/launcher bootstrap packages. They did not contain the promised embedded Python, native engines or GPU stack. Public evidence: https://github.com/saemranian/abu-align-train/releases/tag/v0.1.5.8 and https://api.github.com/repos/saemranian/abu-align-train/releases/latest .

| Area | Previous problem | Correction in this change |
| --- | --- | --- |
| Runtime | First launch installed unpinned packages; a partial environment could be treated as ready | Hash-locked, prebuilt CPU core; no first-launch provisioning in new installers |
| Workers | Python script subprocesses would relaunch a frozen GUI incorrectly | Bundled allowlisted worker dispatcher and a synthetic refinement subprocess test |
| Windows | Source copy plus Conda discovery; a hardcoded user Python path | Bundled-runtime launcher and current/interpreter-configured worker selection |
| CPU CLI | Unconditionally imported Torch and queried CUDA capability on CPU | Optional Torch for native CPU workflows and guarded precision/device initialization |
| Mac and Linux | Source-only wrappers and unverified Universal/MPS claims | Native per-architecture frozen builds, explicit browser viewport fallback, candid capability labeling |
| Publisher | Reset origin, stage everything and force-push main and tags | Allowlisted public export, preserved public ancestry, immutable versions and atomic fast-forward push |
| Workflow | Stable publication on every main push after a directory listing | Pinned actions/tools, actual UI/worker/installer checks, checksums and draft-only candidates |
| Licensing | Conflated VGGT and Omega, overbroad Apache claim, missing ACE0 source-only rule | Corrected notices, model/engine catalog and explicit distribution limits |
| State and privacy | Broad source/document wildcards and state next to installed files | Approved payload/source lists and per-user installed state |
| Developer integration | Upstream VGGT package identity | Abu package metadata and optional command-line entry points |

## Licensing findings

Free price does not make research or commercial usage compliant. Attribution is not the only requirement, and a first-run model download does not remove the user's license obligations. The plan's assertion that an ACE0 label protects the distributor from liability is incorrect.

VGGT-Omega has a separate FAIR noncommercial grant, not the original VGGT July 2025 change. VGG-T3 is also noncommercial. ACE0's redistribution grant is source-format-only. PFGS360 has no complete top-level grant established in the inspected source; UniK3D is CC-BY-NC-SA-4.0. 3DGRUT's top-level code is Apache-2.0, not generically research-only. LichtFeld is GPLv3-or-later with additional GPU-library permission. The retained root custom VGGT license cannot be replaced by a blanket Apache badge. See THIRD_PARTY_NOTICES.md for primary links.

No upstream engine repository was modified by this change. No research checkpoints, ACE0 package, PFGS360 package, LichtFeld binary, arbitrary CUDA toolkit or client dataset is part of the new core payload. Bundled dependency license files and an SBOM improve traceability; they are not a substitute for the final native-library, codec and source-obligation audit.

## What is deliberately not claimed

The core is a candidate, not proof of a production-quality installation on every PC. It does not include Gaussian training runtimes, neural matching weights or FFmpeg. Windows native M54 uses WGL; other platforms expose the existing shared browser viewer rather than claiming an unimplemented native renderer. A selected CUDA device does not convert CUDA kernels to Metal. Automatic qualified engine packs remain future work until actual binaries and redistribution grants are available.

Signing identities, Apple notarization access, independent clean-machine acceptance and rights-holder clarification cannot be fabricated in code. The workflow keeps results as drafts so those gaps do not become unsupported stable-release claims. Existing public history is preserved; this change does not certify that historical commits contain no sensitive material.

## Verification

Completed before the final candidate build:

- Eleven distribution safety tests passed, covering payload escapes, credential patterns, private/unlisted files, immutable version validation, safe remote handling, engine policy and rejection of the old tiny source-only installers.
- Two spherical-refinement regressions, twelve shared M54 tests and twelve embedded-workspace regressions passed. M54 browser tests used local loopback and synthetic scenes.
- The real frozen Windows executable passed native-library, full GUI construction and refinement-subprocess smoke tests with Conda/Python removed from PATH and external connections denied.
- Synthetic CPU refinement reduced angular RMS from 0.721854 degrees to 2.3741e-10 degrees and wrote eight exports without changing its input. This is a packaging/regression fixture, not a real-image accuracy benchmark.
- A source wheel built and installed into an isolated target; its CPU native-library self test passed.
- The public-source dry run exported 176 explicitly approved files and excluded client data and private snapshots. A pre-existing website overview was preserved locally and excluded from publication.

Final-version installer, memory measurements and live CI results are recorded in walkthrough.md. No all-engine hardware qualification, independent clean-OS acceptance or final legal clearance is implied by these tests.

## Platform dependency gate discovered during verification

The official PyPI 4.2.0 wheel inventory includes macOS 14+ arm64 and no Intel Mac wheel: https://pypi.org/pypi/pycolmap/4.2.0/json . Windows x64, Linux x64 and macOS arm64 are the three candidate targets. The false Universal claim is removed. Intel support needs a separately tested native dependency build and corresponding license inventory; silently falling back to an older spherical-camera backend would change functionality. macOS candidates require version 14 or later for this dependency.

The interpreter inventory revealed a second platform gate: setup-python has 3.11.16 on Linux but its last Windows/macOS 3.11 binaries are 3.11.9. Local Windows testing uses 3.11.16. The explicitly pinned CI compatibility builds therefore stay drafts; maintained interpreters on Windows/macOS must be qualified before any stable promotion. This limitation is visible in the release procedure rather than hidden behind an unbuildable interpreter request.

Final Windows frozen browser check served the packaged M54 HTML over loopback with external networking blocked and Python removed from PATH. The bundle removes Viser client development sources after freezing, preventing its first-launch Node/npm build path. The test does not replace a user-driven WebGL/navigation acceptance session on each target OS.

The final local Windows installer is 180,985,583 bytes (SHA256 6606415997e1ac19c3f68137d6fc63e7286f4549fddf80cd9d2ff702265747e0). A test-identity installation, upgrade, four installed-runtime checks and uninstall all succeeded. The test runtime was removed. The production installer was compiled separately; it was not installed over the owner's existing application. The installed runtime uses the same frozen payload as the production package.

## First public workflow validation

Public candidate v0.1.5.9 (0dc3f87466148a27f7ac327b839f0e9b04ef3a9f) failed GitHub YAML validation before jobs ran: the unquoted pip --only-binary option contained a colon followed by a space. The command now uses a YAML block scalar, and the entire workflow parses successfully with PyYAML. A new v0.1.6.0 candidate replaces it without rewriting the existing tag. No installer release was published by the failed run.

Candidate v0.1.6.0 was also rejected before jobs: runner.temp is unavailable in job-level env expressions. The state path now uses github.workspace. Pinned actionlint 1.7.12 passes the corrected workflow and independently rejects both earlier error fixtures. The publisher now requires this preflight before any public Git mutation, and CI runs it as a prerequisite job. Candidate v0.1.6.1 carries the correction; earlier tags remain unchanged.
