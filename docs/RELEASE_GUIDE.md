# Abu release procedure

## Supported publication channel

The new pipeline builds **CPU core candidates** with a bundled interpreter and libraries. It does not qualify CUDA training, research checkpoints or all hardware automatically. Windows x64, Linux x64 and macOS arm64 have separate jobs. Intel Mac packaging is deferred: pinned PyCOLMAP 4.2.0 has no macOS x86_64 wheel. A tested native source build is required before adding that target. The owner must qualify signing, notarization, final dependency notices and clean-machine installation before calling a build stable.

## The owner says Push to GitHub

1. Finish the requested software changes and appropriate tests. Complete the local mandatory version, C# compilation, mirror, snapshot, walkthrough and Git commit protocol once. Check the version is new and the working tree is clean.
2. Run `python push_to_github.py "Summary" --details "Verification and limitations" --no-bump --dry-run`. This exports only the explicit source manifest and checks obvious credential/client-path patterns. It does not upload anything. Review the manifest and license scope; the scanner is not a complete privacy or legal audit.
3. Run the same command without `--dry-run`. It checks the configured destination, clones public main into isolated temporary staging, replaces only that staged source tree with the approved export, creates a new public commit and tag, and makes an atomic ordinary push. It never resets origin, force-pushes, stages unrelated workspace changes or replaces an existing tag.
4. Follow the GitHub Actions run. All three supported OS/architecture jobs must pass before a draft release can be assembled. A build failure is a failed candidate, not a published working installer. Fix the cause and use a new version; do not rewrite the old tag.
5. Report the local commit, public commit, candidate tag, tests, asset links and remaining qualification. Drafts are intentional. A successful upload is not legal clearance or a stable release.

## Builder

Use Python 3.11.16 and `python -m pip install --require-hashes --only-binary=:all: -r installer/requirements-core.lock` in an isolated build environment. `python installer/build_core.py` builds the real runtime and runs a native-library self test. Then run the frozen binary's `--gui-smoke-test` and `--worker-smoke-test` with `ABU_OFFLINE_TEST=1` and a temporary `ABU_DATA_HOME`.

Compile the C# launcher, then the Inno script on Windows. Mac and Linux scripts consume the already built runtime. They never install Python or pip dependencies on the user's computer. Build tools are checksum pinned in installer/build-tools.json. Updating a mutable upstream tool requires reviewing and changing its expected hash explicitly.

`installer/prepare_candidate.py` collects installers, architecture-specific manifests, a CycloneDX inventory and collected notices/core source. The assembly job recomputes checksums over all platform installers. Do not flatten different platform SBOMs onto the same filename. No test images, logs, caches or private backup snapshots belong in the upload directory.

## Privacy and existing public history

installer/public-files.json and installer/core-files.json are explicit independent allowlists. Adding a local file does not make it public. Keep client datasets entirely outside these lists, including thumbnails, masks, reconstructions and test logs. The publisher exports a clean current public tree, but does not erase already published history. Any past credential exposure requires rotation and a separately approved history cleanup.

## Source packaging and licensing

The source wheel is an optional developer path. It uses the Abu package identity and CLI entry points and retains required VGGT source/header licenses. Neural and Gaussian training runtimes remain external; the wheel must not imply they are included. Full compiled dependency license qualification and model rights are release gates. Consult THIRD_PARTY_NOTICES.md and installer/engines.json rather than assuming every integration is Apache licensed.

## Interpreter qualification

Local Windows verification uses Conda CPython 3.11.16. The official setup-python inventory supplies 3.11.16 for Linux but only 3.11.9 binaries for Windows/macOS in the 3.11 family. CI pins those explicitly for draft compatibility builds; it does not pretend 3.11.16 binaries exist on every runner. A maintained Windows/macOS interpreter build (or a tested upgrade of Python/NumPy and all native wheels) is a mandatory stable-release gate. Do not promote the draft compatibility artifacts as production installers. Official inventory: https://github.com/actions/python-versions/blob/main/versions-manifest.json .

The publisher now runs checksum-pinned actionlint 1.7.12 before cloning or making any public Git changes. This validates GitHub expression contexts as well as YAML. `python installer/workflow_lint.py` runs the same preflight directly; CI also runs it before platform builds. Only the build-tool archive is downloaded when absent; no image or model data is involved.
