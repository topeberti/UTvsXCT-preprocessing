import numpy as np
import cv2

def frontwall_orientation(volume, reference_axis, compare_axis):
    """
    Estimate the inclination angle of the front wall in a 2D plane from a 3D volume.

    The function analyzes a 3D volume and calculates the inclination of a front-facing structure
    (e.g., a wall) in a 2D plane defined by two axes. It does this by projecting the volume onto 
    that plane and fitting a line to the structure using a binary mask and least-squares linear regression.

    Parameters
    ----------
    volume : numpy.ndarray
        3D numpy array with shape (Z, Y, X). The axis indices must be:
        0 = Z (depth), 1 = Y (vertical), 2 = X (horizontal).
    
    reference_axis : int
        Index of the reference axis (commonly 0 for Z).
    
    compare_axis : int
        Index of the axis compared against the reference axis (commonly 1 for Y or 2 for X).

    Returns
    -------
    angle_deg : float or None
        Estimated inclination angle (in degrees) of the front wall in the plane defined by 
        (reference_axis, compare_axis). Returns None if the data is insufficient for fitting.
    
    Raises
    ------
    ValueError
        If the provided axes are invalid or not distinct.
    """

    # Validate that both axes are valid and different
    for axis in (reference_axis, compare_axis):
        if axis not in [0, 1, 2]:
            raise ValueError("Axes must be 0 (Z), 1 (Y), or 2 (X).")
    if reference_axis == compare_axis:
        raise ValueError("reference_axis and compare_axis must be different.")

    # Determine the axis along which to collapse the volume to get the 2D projection
    collapse_axis = list({0, 1, 2} - {reference_axis, compare_axis})[0]

    def compute_angle(volume, collapse_axis):
        # Project the volume by taking the maximum intensity along the collapse axis
        b_scan = np.max(volume, axis=collapse_axis)

        # Create a binary mask to isolate strong signal regions (e.g., wall)
        mask = (b_scan > 100).astype(np.uint8) * 255

        # Apply the mask to the B-scan to isolate meaningful pixels
        bscan_uint8 = b_scan.astype(np.uint8)
        bscan_masked = cv2.bitwise_and(bscan_uint8, bscan_uint8, mask=mask)

        # Find coordinates of non-zero pixels in the masked B-scan
        ys, xs = np.where(bscan_masked > 0)

        # If not enough points are found, angle estimation is not possible
        if len(xs) < 2:
            return None

        # Fit a line y = m*x + b using least squares regression
        A = np.vstack([xs, np.ones_like(xs)]).T
        m, _ = np.linalg.lstsq(A, ys, rcond=None)[0]

        # Convert the slope to degrees
        return np.degrees(np.arctan(m))

    # Compute and return the inclination angle
    angle = compute_angle(volume, collapse_axis)
    return angle
