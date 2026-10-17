# Design critique — Abu Align & Train

Audit baseline: v0.0.4.9. Evidence: inspected desktop captures from the preceding
release, current presentation code, and existing layout tests. This is an expert
heuristic critique, not a usability study or measured task-completion benchmark.

## Verdict
The interface has a coherent palette, but it is still an engineering control panel
wearing a calmer theme. We have polished surfaces faster than we have simplified
choices. Calling it minimal overstates the result: a small window spends too much
space explaining and configuring the workspace before exposing useful settings.

## Findings
| Priority | Criticism | Evidence and consequence | Disposition |
|---|---|---|---|
| High | Permanent chrome consumes the work area. | Header, dataset, profile row, tabs, section introduction, engine cards, and footer leave core settings below the fold in the reviewed 1240×900 window. | Phase 1: disclose profiles on demand; shorten section introductions. |
| High | Activity is a transcript rather than an inspection tool. | Read-only scrolling output has no text search. A user must manually scan thousands of lines to find an error. | Phase 2: find next/previous with feedback; preserve raw logs. |
| High | The training summary can overflow before engine settings do. | Iterations, target run, and auto-training are packed in one horizontal row. Engine panel tests do not cover those controls. | Phase 1: wrap complete labeled fields and verify bounds. |
| Medium | Profiles have no visible empty-state explanation. | A blank selector and disabled Load button look unfinished; the explanation lives in a tooltip. | Phase 1: explicit inline guidance. |
| Medium | The rig viewport hides its interaction model. | Mouse orbit/zoom hints omit pan, keyboard selection, and reset. These exist but require discovery by accident. | Phase 3: on-demand Controls guide beside Display. |
| Medium | Fixed settings width is wasteful in wide rig windows. | Left column is fixed at 560 pixels regardless of available width. | Defer draggable split until persistence and minimum-width behavior can be tested. |
| Medium | Three launch actions demand knowledge before action. | Align & train, Align only, and Train only remain adjacent. The main label depends on a toggle on another tab. | Defer a unified execution-mode model; do not cosmetically hide modes while callbacks still differ. |
| Medium | Progress lacks a coherent job model. | Generic stages, engine-specific log parsing, and a single percentage can imply precision they do not have. | Defer until engine event contracts support explicit phases and indeterminate work. |
| Medium | The browser inspector is visually quieter but not validated end-to-end. | Last release compiled its presentation edits without a model-load/browser interaction pass. | Document limitation; no claim that palette work proves 3D correctness. |
| Low | Too many descriptions repeat what controls already say. | Section introductions and engine descriptions compete with actual input fields. | Phase 1: keep task headings and move detailed explanations to existing tooltips. |

## What should remain
Keep native file dialogs, read-only logs, reversible preview controls, clear focus
outlines, shared graphite/mint colors, pinned execution actions, and explicit
engine choices. Aesthetic purity is not a reason to hide errors or remove experts'
controls. Preserve all input and engine behavior during this presentation pass.

## What this audit does not establish
No novice-user testing, screen-reader certification, cross-monitor DPI certification,
GPU-rendering benchmark, or photogrammetry quality comparison was performed.
The deferred items are real remaining design debt, not quietly completed tasks.

## Implementation disposition
Phases 1–3 implemented in v0.0.5.0: profile disclosure and inline guidance,
wrapped training fields, leaner task headings, non-destructive log search, and
on-demand rig controls help. Deferred findings remain open as explicitly scoped.
