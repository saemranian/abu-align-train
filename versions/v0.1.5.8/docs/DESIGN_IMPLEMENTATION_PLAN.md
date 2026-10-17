# Phased implementation plan

Scope: resolve the concrete presentation defects identified in DESIGN_CRITIQUE.md
without mixing in the existing uncommitted engine/GUI/renderer work.

## Phase 1 — Give settings room
- Collapse the profile editor behind an explicit Profiles button.
- Explain the empty profile state in the expanded editor.
- Remove redundant section introductions and wrap training summary fields as units.
- Acceptance: profiles open by mouse or keyboard and still load/save; settings
  controls stay inside their parent at 960×720 and larger. Phase status: complete — 9 desktop tests passed (20.954s).

## Phase 2 — Make activity inspectable
- Add on-demand log search with next/previous, match count, Escape to close,
  and Ctrl+F to open. Search existing output without changing or deleting it.
- Pause follow mode when inspecting a match; explain how to resume following.
- Acceptance: case-insensitive wraparound search, empty/not-found states,
  no mutation of log contents, and working compact layout. Phase status: complete — 10 desktop tests passed (20.707s).

## Phase 3 — Explain viewport controls where they are used
- Add an on-demand Controls guide covering orbit, pan, zoom, reset, and camera
  selection; keep the normal viewport uncluttered.
- Acceptance: guide toggles at the minimum window size; preview and export
  configuration remain unchanged. Phase status: complete — 8 rig tests passed (14.997s).

## Phase 4 — Verify and release
- Run the final suite against the release snapshot; compile all release modules.
- Update critique disposition, user-facing walkthrough, and implementation record.
- Bump using the base-10 odometer, rebuild launchers, preserve a snapshot, and
  commit only this work plus necessary version changes. Phase status: complete — 22 isolated-snapshot tests passed (41.399s); 18 modules compiled.

## Deliberate limits
A unified execution mode, job-state/event redesign, draggable rig split, and
full browser model-load validation are follow-on projects. They require deeper
contracts or dedicated validation; they are not disguised as small style changes.
