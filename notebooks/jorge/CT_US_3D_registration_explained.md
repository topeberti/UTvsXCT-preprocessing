### CT_US_3D_registration.ipynb — Cell-by-Cell Explanation

### Cell 0 — Plan (Markdown)
- **Purpose**: High-level roadmap for registering 3D CT and US volumes.
- **Content**: Steps for geometry unification, preprocessing, masking, initialization, multi-resolution rigid→optional deformable registration, validation, and outputs.

### Cell 1 — Environment check
- **Validates runtime**: Ensures Python ≥ 3.10.
- **Dependency report**: Prints versions of `numpy`, `scipy`, `matplotlib`, `SimpleITK`; shows SimpleITK build.

### Cell 2 — Parameters
- **Paths**: `CT_PATH`, `US_PATH`, optional `CT_MASK_PATH`, `US_MASK_PATH`, and `OUT_DIR`.
- **Physical spacings**: `US_NATIVE_SPACING_MM`, `CT_NATIVE_SPACING_MM` (critical for correct resampling).
- **Two-stage geometry**:
  - **Coarse**: common isotropic spacing via `COARSE_SPACING_MM`.
  - **Fine**: CT to `CT_FINE_SPACING_MM` isotropic; US kept native if `US_KEEP_NATIVE_FOR_FINE=True`.
- **Preprocessing**: CT clip percentiles; US log compression and Gaussian smoothing.
- **Rigid registration**: MI bins, sampling percent, pyramid shrink/sigmas, optimizer params.
- **Optional deformable**: Toggle B-spline (`USE_BSPLINE`) and its grid spacing and iterations.

### Cell 3 — I/O, normalization, resampling, masks, and utilities
- **I/O helpers**:
  - **`sitk_from_numpy`**: Builds a `sitk.Image` (sets spacing, origin, direction) from a 3D NumPy array.
  - **`read_image_any`**: Reads `.npy` or `.tif`/other via SimpleITK; overrides spacing when provided.
  - **`write_image`**: Saves images with compression.
- **Normalization**:
  - **Percentiles** utilities and **`normalize_ct`** (clip→rescale to [0,1]).
  - **`normalize_us`**: Optional smoothing → percentile rescale → optional log compression.
- **Resampling**:
  - **`resample_to_spacing` / `resample_isotropic`**: Compute new size from spacing ratios; preserve origin/direction; choose interpolator.
- **Masks & simple metrics**:
  - **`otsu_mask`**: Otsu threshold → closing → keep largest component; returns `UInt8` mask.
  - **`apply_mask`**, **`dice_coefficient`** for binary masks.
- **Transforms & QC**:
  - **`init_rigid_by_moments`**: Center-of-mass initializer.
  - **`transform_and_resample`**: Warp moving into fixed geometry with a transform.
  - **`show_overlay_slices`**: Quick fixed gray + moving overlay slice viewer.

### Cell 4 — Registration functions
- **`register_rigid_mi`** (rigid 6-DOF):
  - Mattes Mutual Information with random voxel sampling; supports fixed/moving masks.
  - Multi-resolution pyramid (shrink factors, smoothing sigmas in physical units).
  - Optimizer: RegularStepGradientDescent with physical scales; optional initial transform; verbose callbacks.
- **`register_bspline_refine`** (light deformable):
  - Initializes coarse B-spline grid over the fixed image.
  - MI + random sampling; LBFGSB optimizer if available; multiresolution pyramid.
  - Composes B-spline on top of the provided initial transform (usually rigid) and returns composite.

### Cell 5 — Load and preprocess (coarse and fine)
- **Read raw** CT and US using native spacings; print sizes/spacings.
- **Coarse stage**:
  - Resample both to `COARSE_SPACING_MM` isotropic; normalize; make coarse masks (CT via Otsu; US via threshold).
  - Save normalized coarse images and masks as TIFF.
- **Fine stage**:
  - Resample CT to `CT_FINE_SPACING_MM` isotropic; keep US native (or resample if configured).
  - Normalize; create fine masks; save TIFFs.
- **Metadata**: Write JSON with spacing/origin/direction to aid external tools.

### Cell 6 — Two-stage rigid registration and export
- **`save_transform_safely`**: Writes `.tfm` when possible, otherwise `.h5` (Composite transforms).
- **Stage 1 (coarse)**:
  - Rigid MI on coarse normalized images with masks; logs per-iteration metric; saves transform.
  - QC: resample US→CT (coarse) and save overlay.
- **Stage 2 (fine)**:
  - Rigid MI on fine CT (isotropic) and native US (anisotropic), initialized from Stage 1; save transform; QC overlay.
- **Export CT→US space**:
  - Invert Stage 2 transform (US→CT) to get CT→US; save inverse.
  - Resample both normalized and raw CT onto native US grid; save for visualization/analysis; optional overlay QC.

### Cell 7 — Optional B-spline refinement
- **If enabled**:
  - Run deformable refinement on top of fine rigid; save composite transform; QC overlay in CT space.
  - Attempt to invert composite for CT→US export (often not analytically available); otherwise warn.

### Cell 8 — Quantitative validation
- **Transform selection**: Prefer composite B-spline if available; else rigid Stage 2; supports loading from disk.
- **Dice in CT space**: Warp US mask → CT space; compute Dice vs. CT mask; save warped mask.
- **Dice in US space**: If inverse available, warp CT mask → US space; compute Dice; save warped mask.

### Cell 9 — Tuning notes (Markdown)
- **Guidance**: What to adjust if alignment is off, optimizer stalls, speckle dominates, or local distortions exist.

### Cells 10–17 — Empty
- **Reserved**: No content.

### Order of operations at runtime

1) Load parameters and prepare output directory
- Read paths, spacings, and all registration/preprocessing constants from Cell 2.
- Ensure `OUT_DIR` exists.

2) Load raw volumes with correct physical metadata
- `ct_raw = read_image_any(CT_PATH, spacing_mm=CT_NATIVE_SPACING_MM)`
- `us_raw = read_image_any(US_PATH, spacing_mm=US_NATIVE_SPACING_MM)`

3) Coarse preprocessing (robust global alignment space)
- Resample both to isotropic `COARSE_SPACING_MM`: `ct_coarse`, `us_coarse`.
- Normalize: `ct_coarse_n = normalize_ct(ct_coarse)`, `us_coarse_n = normalize_us(us_coarse)`.
- Masks: `ct_mask_coarse = otsu_mask(ct_coarse_n)`, `us_mask_coarse` via simple threshold.
- Save coarse normalized images and masks to `OUT_DIR`.

4) Fine preprocessing (retain US native, refine CT)
- Resample CT to `CT_FINE_SPACING_MM` → `ct_fine`; US kept native if `US_KEEP_NATIVE_FOR_FINE=True`.
- Normalize: `ct_fine_n`, `us_fine_n`.
- Masks: `ct_mask_fine = otsu_mask(ct_fine_n)`, `us_mask_fine` via threshold.
- Save fine normalized images and masks.

5) Stage 1: Coarse rigid registration
- Call `register_rigid_mi(fixed=ct_coarse_n, moving=us_coarse_n, fixed_mask=ct_mask_coarse, moving_mask=us_mask_coarse, ... )`.
- Output: `rigid_tx_stage1` (US→CT). Save transform and coarse QC overlay (`us_on_ct_coarse`).

6) Stage 2: Fine rigid registration (initialized from Stage 1)
- Call `register_rigid_mi(fixed=ct_fine_n, moving=us_fine_n, initial_transform=rigid_tx_stage1, ...)` with tighter pyramid.
- Output: `rigid_tx_stage2` (US→CT). Save transform and fine QC overlay (`us_on_ct_fine`).

7) Export CT onto native US grid
- Compute inverse: `tx_ct_to_us = rigid_tx_stage2.GetInverse()`.
- Resample CT:
  - Visualization: `ct_in_us_vis = transform_and_resample(ct_fine_n, us_fine_n, tx_ct_to_us)`.
  - Analysis (preserve units): `ct_in_us_raw = transform_and_resample(ct_raw, us_fine_n, tx_ct_to_us)`.
- Save both.

8) Optional deformable refinement (if `USE_BSPLINE=True`)
- `bspline_tx, _ = register_bspline_refine(fixed=ct_fine_n, moving=us_fine_n, initial_tx=rigid_tx_stage2, ...)`.
- Save composite transform; save QC overlay (`us_bspline_on_ct`).
- Attempt inverse for CT→US export; if unsupported, warn and skip deformable CT→US export.

9) Quantitative validation
- Warp `us_mask_fine` → CT fine space with chosen US→CT transform; compute Dice against `ct_mask_fine` and save.
- If inverse available, warp `ct_mask_fine` → US space; compute Dice vs `us_mask_fine` and save.

10) Artifacts saved (typical)
- Transforms: `rigid_stage1.(tfm|h5)`, `rigid_stage2.(tfm|h5)`, `rigid_stage2_inverse.(tfm|h5)`, and optionally `composite_rigid_bspline.h5`.
- Images: `ct_coarse_norm.tif`, `us_coarse_norm.tif`, `ct_fine_norm.tif`, `us_fine_norm.tif`.
- Masks: `ct_mask_*`, `us_mask_*`, plus warped masks (`us_mask_in_ct.tif`, optional `ct_mask_in_us.tif`).
- Overlays/QC: `us_on_ct_coarse.tif`, `us_on_ct_fine.tif`, optional `us_bspline_on_ct.tif`, `ct_in_us_space_VIS.tif`, `ct_in_us_space_RAW.tif`.

11) Key knobs affecting behavior
- `MI_BINS`, `SAMPLING_PERCENT`, `PYR_SHRINK`, `PYR_SIGMAS`, `MAX_ITERS`, `LEARNING_RATE`, `MIN_STEP`, `RANDOM_SEED`.
- Masks strongly influence MI stability; spacing correctness is critical for physical scaling and optimizer behavior.
