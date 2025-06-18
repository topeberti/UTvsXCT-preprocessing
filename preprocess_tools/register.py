import numpy as np
import cv2
from scipy.ndimage import rotate
import numpy as np
import cv2
from scipy.ndimage import rotate
from scipy.signal import find_peaks
from tqdm import tqdm


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


def UT_surface_coordinates(volume_UT, signal_percentage=1.0):
    """
    Extracts the surface coordinates from an Ultrasound (UT) volume.

    For each (x, y) column in the volume, it takes the A-scan (signal along z),
    finds the maximum value, and keeps it only if it exceeds a threshold.
    Returns the coordinates (z, y, x) of the surface points.

    Parameters:
        volume_UT (numpy.ndarray): 3D UT volume (z, y, x)
        signal_percentage (float): Percentage of the signal depth to process (0.0 to 1.0).
                                  For example, if 0.5, only the first 50% of the signal 
                                  along the z-axis will be processed. Default is 1.0 (100%).

    Returns:
        numpy.ndarray: Array of surface points (z, y, x)
    """
    threshold = 60  # Minimum amplitude to consider as surface
    y_max = volume_UT.shape[1]
    x_max = volume_UT.shape[2]
    
    # Calculate the maximum z-index to consider based on the percentage
    z_max = volume_UT.shape[0]
    z_limit = int(z_max * signal_percentage)
    
    surface_coords = []  # List to store (z, y, x) surface coordinates

    # Loop over each x-y location in the volume
    for x in range(x_max):
        for y in range(y_max):
            A_scan = volume_UT[:z_limit, y, x]  # Signal along z-axis up to z_limit
            max_val = np.max(A_scan)     # Maximum value in the signal

            # If the max value is above the threshold, store the index
            if max_val > threshold:
                z = np.argmax(A_scan)    # Position of the max value
                surface_coords.append((z, y, x))

    if not surface_coords:
        raise ValueError("No surface points found above the threshold.")

    return np.array(surface_coords)

def XCT_surface_coordinates(volume_XCT, signal_percentage=1.0):
    """
    Extracts the surface coordinates from a X-ray CT (XCT) volume.

    For each (x, y) column in the volume, it finds the first peak in the 
    signal along z that exceeds a fixed threshold. This peak is considered 
    the surface point.

    Parameters:
        volume_XCT (numpy.ndarray): 3D XCT volume (z, y, x)
        signal_percentage (float): Percentage of the signal depth to process (0.0 to 1.0).
                                  For example, if 0.5, only the first 50% of the signal 
                                  along the z-axis will be processed. Default is 1.0 (100%).

    Returns:
        numpy.ndarray: Array of surface points (z, y, x)
    """
    threshold = 175      # Minimum peak height to be considered
    
    # Calculate the maximum z-index to consider based on the percentage
    z_max = volume_XCT.shape[0]
    z_limit = int(z_max * signal_percentage)
    
    # Pre-compute the mask of columns that have values above threshold
    # Only consider the portion of the volume up to z_limit
    max_along_z = np.max(volume_XCT[:z_limit], axis=0)  # Shape: (y_max, x_max)
    valid_columns = max_along_z >= threshold
    
    # Get coordinates of valid columns
    y_coords, x_coords = np.where(valid_columns)
    
    # Initialize array to store first peak z-coordinate for each valid column
    surface_coords = []
    
    # Process only the valid columns
    for i in tqdm(range(len(y_coords)), desc="Finding surface points"):
        y, x = y_coords[i], x_coords[i]
        signal = volume_XCT[:z_limit, y, x]  # Only process up to z_limit
        
        # Find the first peak above threshold
        peaks, _ = find_peaks(signal, height=threshold)
        
        if len(peaks) > 0:
            z = peaks[0]
            surface_coords.append((z, y, x))
    if not surface_coords:
        raise ValueError("No surface points found above the threshold (175).")
        
    return np.array(surface_coords)

from preprocess_tools.onlypores import material_mask

def XCT_surface_coordinates_2(volume_XCT, signal_percentage=1.0):
    """
    Extracts the surface coordinates from a X-ray CT (XCT) volume.

    For each (x, y) column in the volume, it finds the first peak in the 
    signal along z that exceeds a fixed threshold. This peak is considered 
    the surface point.

    Parameters:
        volume_XCT (numpy.ndarray): 3D XCT volume (z, y, x)
        signal_percentage (float): Percentage of the signal depth to process (0.0 to 1.0).
                                  For example, if 0.5, only the first 50% of the signal 
                                  along the z-axis will be processed. Default is 1.0 (100%).

    Returns:
        numpy.ndarray: Array of surface points (z, y, x)
    """

     # Calculate the maximum z-index to consider based on the percentage
    z_max = volume_XCT.shape[0]
    z_limit = int(z_max * signal_percentage)

    xct_mask = material_mask(volume_XCT)

    xct_mask = xct_mask[:z_limit, :, :]  # Take only the first percentage of the volume

    threshold = 175
    max_along_z = np.max(volume_XCT[:z_limit], axis=0)  # Shape: (y_max, x_max)
    valid_columns = max_along_z >= threshold

    # Get coordinates of valid columns
    y_coords, x_coords = np.where(valid_columns)
    
    # Initialize array to store first peak z-coordinate for each valid column
    surface_coords = []
    
    # Process only the valid columns
    for i in tqdm(range(len(y_coords)), desc="Finding surface points"):
        y, x = y_coords[i], x_coords[i]
        signal = volume_XCT[:z_limit, y, x]  # Only process up to z_limit
        
        # Find the first nonzero pixel in the signal
        z = np.argmax(signal > 0)

        surface_coords.append((z, y, x))
        
    return np.array(surface_coords)
    

def YZ_XZ_inclination(volume, volumeType='XCT', signal_percentage=1.0):
    """
    Calculates the inclination of the surface in a 3D volume with respect to
    the YZ and XZ planes by fitting a plane to the extracted surface points.

    Depending on the volumeType ('XCT' or 'UT'), it calls the appropriate
    surface extraction function, fits a plane z = ax + by + c, and computes
    the tilt angles from the plane coefficients.

    Parameters:
        volume (numpy.ndarray): 3D image volume (z, y, x)
        volumeType (str): 'XCT' for CT scan, 'UT' for ultrasound
        signal_percentage (float): Percentage of signal depth to process for both UT and XCT.
                                  Default is 1.0 (100%).

    Returns:
        tuple: (angle_yz, angle_xz) in degrees
    """

    # Choose appropriate method based on volume type
    if volumeType == 'UT':
        surface_coords = UT_surface_coordinates(volume, signal_percentage)
    elif volumeType == 'XCT':
        surface_coords = XCT_surface_coordinates(volume, signal_percentage)
    else:
        raise ValueError("volumeType must be either 'XCT' or 'UT'")

    # Extract Z (depth), Y (height), and X (width) coordinates
    Z = surface_coords[:, 0]
    Y = surface_coords[:, 1]
    X = surface_coords[:, 2]

    # Fit a plane z = ax + by + c using least squares
    A = np.c_[X, Y, np.ones_like(X)]
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = C  # Plane coefficients

    # Compute inclination angles in degrees
    angle_yz = np.degrees(np.arctan(b))  # Tilt relative to YZ plane
    angle_xz = np.degrees(np.arctan(a))  # Tilt relative to XZ plane

    return angle_yz, angle_xz  # Return tilt angles