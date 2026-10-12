# Design critique — second pass

Baseline: v0.0.5.0. Evidence: current source and prior desktop screenshots;
this is a heuristic review, not new user research. The first audit is preserved.

## Verdict
The first pass recovered space and made logs searchable. It did not resolve the
central interaction ambiguity: the user still has to understand three launch
buttons plus an automatic-training toggle to predict a run. That is product
complexity, not merely visual clutter. Hiding profiles helped density but did
nothing to make execution intent explicit.

## Findings and decisions
1. **High: execution intent is split across controls.** Align & train, Align only,
   and Train only compete, while a Training checkbox changes the primary label.
   Replace the visible trio with an explicit mode selector and one primary action.
   Route to the existing engine callbacks; do not invent new pipeline semantics.
   Keep configuration profiles compatible with the existing auto-training field.
2. **Medium: log inspection has a weak exit.** Search pauses Follow log, but returning
   to live output requires finding a separate checkbox. Add a direct Resume live
   action that closes search, enables following, and jumps to the latest line.
3. **Medium: camera selection does not scale.** Previous/Next is reasonable for
   eight cameras and tedious for a large rig. Add a numbered camera selector,
   keep it synchronized with mouse/keyboard selection, and clamp after rig changes.
4. **Still open: the rig split remains fixed.** Replacing it safely needs changes
   in the shared GUI constructor and DPI/layout validation. Do not mix that with
   the current uncommitted GUI work.
5. **Still open: progress precision is questionable.** A cosmetic progress bar
   cannot repair missing engine event contracts. Preserve diagnostics and avoid
   inventing an overall ETA or a false completion percentage.
6. **Still open: visual acceptance is incomplete.** Source/tests establish behavior,
   not novice comprehension or cross-monitor DPI correctness. Do not claim that
   an automated test suite proves those qualities.

## Principles for this pass
One visible primary action. Explicit execution intent. A clear exit from inspection.
Direct access for large selections. Preserve user settings, engine semantics, and
pre-existing uncommitted work. Avoid hiding meaningful uncertainty behind styling.

## Implementation disposition
Findings 1–3 addressed in v0.0.5.1. The explicit mode selector, Resume live,
and direct camera selection passed their phase checks. The main launch area was
visually reviewed in an isolated preview. Findings 4–6 remain documented limits.
