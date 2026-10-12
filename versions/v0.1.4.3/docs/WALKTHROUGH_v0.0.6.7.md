# Abu Align & Train v0.0.6.7 — Smart Cache Hit UnboundLocalError Resolution & Zero-Copy State Binding

Version **v0.0.6.7** resolves the `UnboundLocalError: cannot access local variable 'cached_neural' where it is not associated with a value` encountered on Smart Cache hits in `abu_align_2026-09-25_11-23-45.log`.

---

## 1. Problem Description & Root Cause Analysis

In `logs/abu_align_2026-09-25_11-23-45.log`:
- **Dataset**: 22 photos (`Container_03/Sony/100MSDCF/CC/Sel 1\Del Copy Sel 1\Portrait`)
- **Action**: Running alignment with existing neural cache available (`[SMART CACHE HIT] Loaded neural poses & 3D points in 0.26s!`).
- **Crash Traceback**:
  ```text
  Traceback (most recent call last):
    File "align_cli.py", line 2558, in <module>
      main()
    File "align_cli.py", line 2053, in main
      reconstruction, dense_3d, dense_rgb = run_vggt_reconstruction(...)
    File "align_cli.py", line 953, in run_vggt_reconstruction
      if cached_neural is None and use_cache:
  UnboundLocalError: cannot access local variable 'cached_neural' where it is not associated with a value
  ```

### Root Cause
- When fixing the GPU VRAM accumulation issue in `v0.0.6.5`, `del cached_neural` was placed inside the `if cached_neural is not None:` block to immediately purge the 5.5 GB dictionary of cached PyTorch tensors from memory.
- In Python, `del <var>` removes the variable binding from the local function namespace.
- Later in the function (at line 953), the code evaluated:
  `if cached_neural is None and use_cache:`
- Because the variable `cached_neural` was unbound by `del`, accessing it raised `UnboundLocalError`.

---

## 2. Technical Solution

1. **State Flag `was_cached`**:
   Introduced an explicit boolean flag `was_cached` initialized before cache checking:
   ```python
   cached_neural = None
   was_cached = False
   if use_cache:
       cached_neural = load_neural_cache(cache_root, cache_key, device="cpu", dtype=dtype)
       if cached_neural is not None:
           was_cached = True
   ```
2. **Safe Memory Reclamation**:
   Replaced bare unbinding with explicit reference nullification:
   ```python
   del cached_neural
   cached_neural = None
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```
3. **Cache Save Guard**:
   Updated the post-inference cache save condition to use `was_cached`:
   ```python
   # Save to neural cache if not already cached
   if not was_cached and use_cache:
       saved_dir = save_neural_cache(...)
   ```

---

## 3. Verification & Testing

- Added `test_cache_hit_variable_binding` to `tests/test_vggt_omega_optimization.py` to ensure local variable scoping remains valid under cache hit paths.
- Ran full test suite:
  ```bash
  python -m unittest discover tests
  .............................................
  Ran 45 tests in 28.746s
  OK
  ```

---

## 4. Versioning & Executables
- Bumped version to `v0.0.6.7` via `version_manager.py`.
- Recompiled `Abu_Align.exe` and `Abu_Align_and_Train.exe`.
- Created version archive snapshot `versions/v0.0.6.7/`.
