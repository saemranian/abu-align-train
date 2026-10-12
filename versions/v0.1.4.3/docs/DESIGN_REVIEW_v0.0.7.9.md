# UI and UX review — v0.0.7.9

## Critique

The interface is visually quieter, but compact controls still need to behave
correctly under pressure. A clean palette cannot compensate for clipped labels,
disappearing scene space, or a search result count that has stopped telling the truth.

1. **The rig toolbar assumed labels never changed.** Its fixed placements only
   recalculated when the window width changed. Expanding Display or Controls
   changed the requested button widths without updating their allocated space.
2. **Help competed with the scene it explained.** At minimum size, opening Display
   and Controls together consumed most of the viewport. Visual inspection
   confirmed a compressed scene with overlapping preview annotations.
3. **Live search was only live when typing.** Appending, trimming, or clearing
   logs could leave a stale count and selected match until another search action.
4. **Scrolling discarded device intent.** Every wheel event moved one unit,
   regardless of its magnitude; even a zero-delta event moved down. Precision
   scrolling and faster wheel movement were therefore inconsistent.

## Implementation phases

### 1. Preserve the scene — implemented
Replace fixed toolbar allocations with geometry that reacts to requested sizes.
Keep independently wrapped rows and consistent gaps. Show Controls as a bordered
overlay inside the scene instead of allocating another full-height content row.
Escape dismisses the guide before leaving focus view.

Acceptance: expanded labels fit without overlaps at 1120×760; opening the guide
does not reduce viewport height; camera geometry and configuration remain intact.

### 2. Make feedback accurate — implemented
Refresh visible log search after buffer edits, coalescing updates over 150 ms.
Preserve the selected match when it survives buffer trimming. Refreshes neither
scroll the console nor change Follow log. Closing search cancels pending work.

Acceptance: appending updates the count, clearing reports no matches, and passive
updates do not move the reading position or resume live scrolling.

### 3. Respect input — implemented
Accumulate Windows wheel deltas in 120-unit notches, preserve multi-notch events,
discard stale fractional movement when direction reverses, and ignore zero input.
Continue consuming the event before option widgets can change their values.

Acceptance: four quarter-notch events equal one step; three notches equal three
steps; the selected option remains unchanged.

### 4. Verify and release
Exercise all six alignment engines at 960×720, 1240×900, and 1600×1000; test rig
layout, presets, and interaction behavior. Compile the snapshot and verify source
parity. Keep concurrent engine changes outside this UI release.

### Final source-row correction
Visual review also found the path field crowding out the Video action. Source
and output rows now reserve their button widths and give remaining space to the
path field. A regression check asserts that all browse actions are visible.

## Remaining limitations

The rig still requires a wide settings panel. The temporary help overlay covers
part of the scene until dismissed. Full high-DPI coverage and real user testing
remain outstanding. These checks do not benchmark engine performance or GPU memory.
