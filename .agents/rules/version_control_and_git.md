---
trigger: always_on
description: Mandatory rule to bump base-10 version, archive snapshots, sync parity, and git commit after any modification.
---

# MANDATORY RULE: Base-10 Version Control & Git Commit on Every Modification

After **EVERY** modification, bugfix, refactor, or feature addition to the Abu Align & Train software:
**YOU MUST EXECUTE THE FOLLOWING 5-STEP PROTOCOL IMMEDIATELY BEFORE RESPONDING TO THE USER. NEVER WAIT FOR OR EXPECT THE USER TO REMIND YOU.**

1. **Base-10 Version Increment**: Run `python version_manager.py --bump`. Recompile C# launchers (`csc.exe`).
2. **100% Parity**: Copy all modified files from root to `Abu_Align/`.
3. **Backup Snapshot**: Create `Abu_Align/versions/vX.Y.Z.W/` and copy all active scripts/executables/walkthroughs into it.
4. **Documentation**: Update `walkthrough.md` with full details.
5. **Git Commit**: Run `git add -A` and `git commit -m "vX.Y.Z.W: <detailed log>"`.
