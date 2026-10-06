### Purpose
This document explains each section of the notebook `desarrollo/UTvsXCT register/simpleITK_rotation_angles_optimization_berti.ipynb`. The notebook estimates sample tilt angles for UT and XCT 3D volumes and rotates the XCT volume to align it with the UT reference.

### Prerequisites
- Network paths in the notebook must be reachable and the `.tif` stacks present.
- Sufficient RAM for handling large 3D volumes (rotation with `reshape=True` increases size).
- Python dependencies: numpy, tifffile, scipy, matplotlib, tqdm, joblib (plus project-local import for `YZ_XZ_inclination`).

### 1) Imports and environment setup
- Loads scientific/python tooling: `numpy`, `tifffile` (for TIFF stacks), `scipy.ndimage.rotate` (slice rotations), `matplotlib` (visualization), `tqdm` (progress bars).
- Adds the repository path to `sys.path` to import `preprocess_tools.register.YZ_XZ_inclination`.
- Note: `SimpleITK`, `cv2`, and `scipy.optimize.minimize` are imported but not used in this notebook.

### 2) Paths and volume loading
- Defines network file paths for:
  - `file_XCT`: XCT 3D image stack.
  - `file_UT`: UT 3D image stack.
- Loads volumes as NumPy arrays with `tifffile.imread` into `volume_XCT` and `volume_UT`.

### 3) Angle estimation in YZ and XZ planes
- Calls `YZ_XZ_inclination(volume_UT, 'UT')` and `YZ_XZ_inclination(volume_XCT, 'XCT')`.
- Returns two angles for each volume:
  - `angle_yz`: tilt visible in the YZ plane (rotation about the X axis).
  - `angle_xz`: tilt visible in the XZ plane (rotation about the Y axis).
- Prints the angles for both modalities.

### 4) Compute alignment rotation deltas
- Calculates how much to rotate the XCT to match UT:
  - `rotation_angle_yz = angle_yz_xct - angle_yz_ut`
  - `rotation_angle_xz = angle_xz_xct - angle_xz_ut`

### 5) Quick 2D preview (optional)
- Creates a max-intensity projection (MIP) of `volume_XCT` and applies a 2D rotation for a fast visual check.
- Uses `scipy.ndimage.rotate(..., reshape=True, cval=40)`; new pixels are filled with value 40.

### 6) 3D rotation function (sequential version)
- Defines `rotate_volume(volume, angle_yz, angle_xz)` that rotates the volume in two steps:
  - YZ-plane rotation (equivalent to rotation about X axis): rotate each XZ slice `volume[:, :, i]` by `angle_yz`. The intermediate volume shape grows because `reshape=True`.
  - XZ-plane rotation (equivalent to rotation about Y axis): rotate each YZ slice `intermediate[:, i, :]` by `angle_xz` to produce the final volume.
- Uses `cval=40` for the background introduced by rotation.

### 7) 3D rotation function (parallel version, overrides the first)
- Redefines `rotate_volume(volume, angle_yz, angle_xz, n_jobs=-1)` using `joblib.Parallel` to speed up slice rotations.
- Behavior details:
  - If `angle_yz == 0`, skips the first stage and uses the input volume.
  - Computes target shapes by rotating a representative slice with `reshape=True`.
  - Parallelizes per-slice rotation for both stages with progress bars via `tqdm`.
  - Returns the rotated volume; this definition replaces the previous one (same name).

### 8) Example usage
- Runs `rotated_volume_XCT = rotate_volume(volume_XCT, angle_yz_xct, angle_xz_xct)` to align XCT to the UT-derived orientation.

### 9) Visualization and shape checks
- Displays MIPs of the rotated volume for sanity checks.
- Prints shapes before and after rotation. Shapes can change due to `reshape=True` enlarging the bounding box.

### 10) Post-rotation angle verification
- Recomputes `YZ_XZ_inclination` on the rotated XCT volume.
- Prints angles near zero (or the UT reference), confirming successful orientation correction.

### Notes and tips
- **Fill value (`cval`)**: Set to 40; adjust if your background intensity differs.
- **Angle conventions**: `angle_yz` rotates around X (YZ plane), `angle_xz` rotates around Y (XZ plane).
- **Order matters**: YZ rotation first, then XZ; swapping can change outcomes.
- **Performance**: Use the parallel version; control CPU usage via `n_jobs` (default uses all cores).
- **Memory**: Rotations with `reshape=True` increase array sizes; ensure sufficient memory.
- **Unused imports**: `SimpleITK`, `cv2`, and `scipy.optimize.minimize` are imported but not used here.

### Adapting to your data
- Update the file paths to your UT/XCT volumes.
- Ensure `preprocess_tools` is importable (repository on `sys.path` or installed package).
- If needed, tweak `cval`, angle signs, or rotation order to match your acquisition conventions.


