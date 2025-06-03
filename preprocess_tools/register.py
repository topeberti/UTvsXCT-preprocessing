import numpy as np
import cv2
from scipy.ndimage import rotate


def angles_estimation(volume):
    """
    Estimate the front wall orientation from a 3D ultrasound volume.

    This function processes a 3D ultrasound or XCT volume to estimate the orientation 
    of the front wall (typically the first strong reflector surface) in two 
    orthogonal planes: YZ and XZ.

    Parameters
    ----------
    volume : numpy.ndarray
        3D ultrasound volume with shape (Z, Y, X), where:
        - Z is the depth or scan axis,
        - Y is the vertical axis (height),
        - X is the horizontal axis (width).
        
        It is essential that the volume follows this axis order (Z, Y, X)
        for the orientation angles to be computed correctly.

    Returns
    -------
    angle_deg_YZ : float or None
        Estimated angle in degrees of the front wall in the YZ plane (depth vs height).

    angle_deg_XZ : float or None
        Estimated angle in degrees of the front wall in the XZ plane (depth vs width).

    Notes
    -----
    The function projects the volume onto the YZ and XZ planes using maximum intensity 
    projection along the X and Y axes, respectively. It then uses a least squares fit 
    to estimate the angle of the front wall in each plane.

    """
    # --- YZ plane ---
    b_scan_yz = np.max(volume, axis=2)  # collapse X axis
    mask_yz = (b_scan_yz > 100).astype(np.uint8) * 255
    bscan_yz_uint8 = b_scan_yz.astype(np.uint8)
    bscan_mask_yz = cv2.bitwise_and(bscan_yz_uint8, bscan_yz_uint8, mask=mask_yz)

    ys, xs = np.where(bscan_mask_yz > 0)
    if len(xs) < 2:
        angle_deg_YZ = None
    else:
        A = np.vstack([xs, np.ones_like(xs)]).T
        m, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        angle_deg_YZ = np.degrees(np.arctan(m))

    # --- XZ plane ---
    b_scan_xz = np.max(volume, axis=1)  # collapse Y axis
    mask_xz = (b_scan_xz > 100).astype(np.uint8) * 255
    bscan_xz_uint8 = b_scan_xz.astype(np.uint8)
    bscan_mask_xz = cv2.bitwise_and(bscan_xz_uint8, bscan_xz_uint8, mask=mask_xz)

    ys, xs = np.where(bscan_mask_xz > 0)
    if len(xs) < 2:
        angle_deg_XZ = None
    else:
        A = np.vstack([xs, np.ones_like(xs)]).T
        m, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        angle_deg_XZ = np.degrees(np.arctan(m))

    return angle_deg_YZ, angle_deg_XZ


def rotate_volume(volume):
    """
    Rotates a 3D volume to align it based on estimated angles.

    The rotation is performed in two steps:
    1. Rotate the volume in the YZ plane (around the X axis) using an estimated angle.
    2. Rotate the resulting volume in the XZ plane (around the Y axis) using another estimated angle.

    Parameters:
    volume (numpy.ndarray): 3D array representing the volume to be rotated.

    Returns:
    numpy.ndarray: The rotated 3D volume.
    """

    # Rotate volume in the YZ Plane (around X axis)
    # Estimate the angle needed to align the volume in the YZ plane
    angle_YZ, _ = angle.angles_estimation(volume)
    # Perform the rotation around the X axis (axes 0 and 1 correspond to Y and Z dimensions)
    volume_YZ = rotate(volume, angle=angle_YZ, axes=(0, 1), reshape=False)

    # Rotate volume in the XZ Plane (around Y axis)
    # Estimate the angle needed to align the volume in the XZ plane from the already rotated volume
    _, angle_XZ = angle.angles_estimation(volume_YZ)
    # Perform the rotation around the Y axis (axes 0 and 2 correspond to X and Z dimensions)
    rotated_volume = rotate(volume_YZ, angle=angle_XZ, axes=(0, 2), reshape=False)

    return rotated_volume
