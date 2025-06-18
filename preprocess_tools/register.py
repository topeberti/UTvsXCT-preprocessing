import numpy as np
import cv2
from scipy.ndimage import rotate
from scipy.signal import find_peaks
from tqdm import tqdm  # Progress bar for long operations

def rotate_volume(volume):
    """
    Rotates a 3D volume to align it based on estimated angles.

    This function aligns a 3D volume by performing sequential rotations to correct
    the orientation of the front wall surface. The process happens in two steps:
    1. First rotation in the YZ plane (around the X axis)
    2. Second rotation in the XZ plane (around the Y axis) on the already rotated volume

    Parameters
    ----------
    volume : numpy.ndarray
        3D array representing the volume to be rotated, with shape (Z, Y, X).

    Returns
    -------
    numpy.ndarray
        The rotated 3D volume with the same shape as the input volume but aligned
        so that the front wall is perpendicular to the Z axis.

    Notes
    -----
    The function uses the angles_estimation function from this module to determine
    the rotation angles. The reshape=False parameter in the rotate function ensures
    that the output volume has the same dimensions as the input.
    """
    # Step 1: Rotate volume in the YZ Plane (around X axis)
    # Estimate the angle needed to align the volume in the YZ plane
    angle_YZ, _ = YZ_XZ_inclination(volume)
    
    # Perform the rotation around the X axis (axes 0 and 1 correspond to Z and Y dimensions)
    # Note: reshape=False keeps the dimensions consistent
    volume_YZ = rotate(volume, angle=angle_YZ, axes=(0, 1), reshape=False)

    # Step 2: Rotate volume in the XZ Plane (around Y axis)
    # Estimate the angle needed to align the volume in the XZ plane from the already rotated volume
    _, angle_XZ = angles_estimation(volume_YZ)
    
    # Perform the rotation around the Y axis (axes 0 and 2 correspond to Z and X dimensions)
    rotated_volume = rotate(volume_YZ, angle=angle_XZ, axes=(0, 2), reshape=False)

    return rotated_volume


def UT_surface_coordinates(volume_UT, signal_percentage=1.0):
    """
    Extracts the surface coordinates from an Ultrasound (UT) volume.

    This function analyzes an ultrasound volume to identify the front surface (typically
    the first strong reflector) by finding the maximum intensity along each vertical column.
    Only points with intensity above a threshold are considered valid surface points.

    Parameters
    ----------
    volume_UT : numpy.ndarray
        3D ultrasound volume with shape (Z, Y, X), where:
        - Z is the depth or scan axis,
        - Y is the vertical axis (height),
        - X is the horizontal axis (width).
    
    signal_percentage : float, optional
        Percentage of the signal depth to process (0.0 to 1.0).
        For example, if 0.5, only the first 50% of the signal 
        along the z-axis will be processed. Default is 1.0 (100%).

    Returns
    -------
    numpy.ndarray
        Array of surface points with shape (N, 3), where each row contains
        the coordinates (z, y, x) of a detected surface point.
        
    Raises
    ------
    ValueError
        If no surface points are found above the threshold.

    Notes
    -----
    The function uses a threshold value of 60 to determine valid surface points.
    For each column in the volume, the function finds the position of the maximum
    intensity value along the z-axis and includes it in the results only if that
    maximum exceeds the threshold.
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
            # Extract the A-scan (1D signal along z-axis) for this (x,y) position
            A_scan = volume_UT[:z_limit, y, x]
            # Find the maximum intensity in this A-scan
            max_val = np.max(A_scan)

            # Only include points with intensity above the threshold
            if max_val > threshold:
                # Find the depth (z) at which the maximum intensity occurs
                z = np.argmax(A_scan)
                # Store the 3D coordinates of this surface point
                surface_coords.append((z, y, x))

    # Check if any surface points were found
    if not surface_coords:
        raise ValueError("No surface points found above the threshold.")

    # Convert list of points to numpy array
    return np.array(surface_coords)

def XCT_surface_coordinates(volume_XCT, signal_percentage=1.0):
    """
    Extracts the surface coordinates from an X-ray CT (XCT) volume.

    This function analyzes an XCT volume to identify the front surface by finding
    the first peak in the intensity profile along each vertical column that exceeds
    a specified threshold. The function uses a more sophisticated approach than the UT
    version, employing the find_peaks function to locate the first significant intensity
    change that represents the material boundary.

    Parameters
    ----------
    volume_XCT : numpy.ndarray
        3D XCT volume with shape (Z, Y, X), where:
        - Z is the depth or scan axis,
        - Y is the vertical axis (height),
        - X is the horizontal axis (width).
    
    signal_percentage : float, optional
        Percentage of the signal depth to process (0.0 to 1.0).
        For example, if 0.5, only the first 50% of the signal 
        along the z-axis will be processed. Default is 1.0 (100%).

    Returns
    -------
    numpy.ndarray
        Array of surface points with shape (N, 3), where each row contains
        the coordinates (z, y, x) of a detected surface point.
        
    Raises
    ------
    ValueError
        If no surface points are found above the threshold (175).

    Notes
    -----
    The function uses a threshold value of 175 to identify significant peaks.
    It first identifies columns that have at least one value above the threshold,
    then for each such column, it finds the first peak (local maximum) above
    the threshold. A progress bar is displayed during processing using tqdm.
    """
    threshold = 175  # Minimum peak height to be considered as a surface point
    
    # Calculate the maximum z-index to consider based on the percentage
    z_max = volume_XCT.shape[0]
    z_limit = int(z_max * signal_percentage)
    
    # Pre-compute which columns have values above threshold
    # This avoids processing columns that don't have any significant signal
    max_along_z = np.max(volume_XCT[:z_limit], axis=0)  # Shape: (y_max, x_max)
    valid_columns = max_along_z >= threshold
    
    # Get coordinates of valid columns
    y_coords, x_coords = np.where(valid_columns)
    
    # Initialize list to store surface points
    surface_coords = []
    
    # Process only the valid columns (using tqdm for progress tracking)
    for i in tqdm(range(len(y_coords)), desc="Finding surface points"):
        y, x = y_coords[i], x_coords[i]
        
        # Extract the signal along the z-axis for this (x,y) position
        signal = volume_XCT[:z_limit, y, x]
        
        # Find the first peak above threshold using scipy.signal.find_peaks
        # This identifies local maxima in the signal
        peaks, _ = find_peaks(signal, height=threshold)
        
        # If at least one peak was found, store the first one as the surface point
        if len(peaks) > 0:
            z = peaks[0]  # Take the first peak (closest to the surface)
            surface_coords.append((z, y, x))
            
    # Check if any surface points were found
    if not surface_coords:
        raise ValueError("No surface points found above the threshold (175).")
        
    # Convert list of points to numpy array
    return np.array(surface_coords)

from preprocess_tools.onlypores import material_mask

def XCT_surface_coordinates_2(volume_XCT, signal_percentage=1.0):
    """
    Alternative method to extract surface coordinates from an X-ray CT (XCT) volume.

    This function provides an alternative approach to the XCT_surface_coordinates function.
    It uses the material_mask function from the onlypores module to create a mask of the
    material, then finds the first non-zero pixel in each column as the surface point.
    This can be more robust in cases where the first peak detection method fails.

    Parameters
    ----------
    volume_XCT : numpy.ndarray
        3D XCT volume with shape (Z, Y, X), where:
        - Z is the depth or scan axis,
        - Y is the vertical axis (height),
        - X is the horizontal axis (width).
    
    signal_percentage : float, optional
        Percentage of the signal depth to process (0.0 to 1.0).
        For example, if 0.5, only the first 50% of the signal 
        along the z-axis will be processed. Default is 1.0 (100%).

    Returns
    -------
    numpy.ndarray
        Array of surface points with shape (N, 3), where each row contains
        the coordinates (z, y, x) of a detected surface point.

    Notes
    -----
    This function uses the same threshold value (175) as XCT_surface_coordinates
    to identify valid columns, but instead of finding peaks, it identifies the first
    non-zero value in each column using np.argmax(signal > 0). A progress bar is
    displayed during processing using tqdm.
    """
    # Calculate the maximum z-index to consider based on the percentage
    z_max = volume_XCT.shape[0]
    z_limit = int(z_max * signal_percentage)

    # Generate material mask using the function from onlypores module
    xct_mask = material_mask(volume_XCT)
    
    # Limit the mask to the specified percentage of the volume depth
    xct_mask = xct_mask[:z_limit, :, :]

    # Use same threshold as in XCT_surface_coordinates for consistency
    threshold = 175
    
    # Identify columns with values above threshold
    max_along_z = np.max(volume_XCT[:z_limit], axis=0)  # Shape: (y_max, x_max)
    valid_columns = max_along_z >= threshold

    # Get coordinates of valid columns
    y_coords, x_coords = np.where(valid_columns)
    
    # Initialize list to store surface points
    surface_coords = []
    
    # Process only the valid columns (using tqdm for progress tracking)
    for i in tqdm(range(len(y_coords)), desc="Finding surface points"):
        y, x = y_coords[i], x_coords[i]
        
        # Extract the signal along the z-axis for this (x,y) position
        signal = volume_XCT[:z_limit, y, x]
        
        # Find the first non-zero pixel in the signal
        # This identifies the first point where the signal becomes positive
        z = np.argmax(signal > 0)

        # Store the 3D coordinates of this surface point
        surface_coords.append((z, y, x))
        
    # Convert list of points to numpy array
    return np.array(surface_coords)
    

def YZ_XZ_inclination(volume, volumeType='XCT', signal_percentage=1.0):
    """
    Calculates the inclination of the surface in a 3D volume with respect to
    the YZ and XZ planes by fitting a plane to the extracted surface points.

    This function is a high-level interface that combines surface extraction with
    plane fitting to determine the tilt angles of a sample surface. It can process
    both XCT and UT volumes and is more comprehensive than the angles_estimation function.

    Parameters
    ----------
    volume : numpy.ndarray
        3D image volume with shape (Z, Y, X), where:
        - Z is the depth or scan axis,
        - Y is the vertical axis (height),
        - X is the horizontal axis (width).
    
    volumeType : str, optional
        Type of the volume data. Must be either 'XCT' for X-ray CT or 'UT' for ultrasound.
        Default is 'XCT'.
    
    signal_percentage : float, optional
        Percentage of signal depth to process (0.0 to 1.0).
        For example, if 0.5, only the first 50% of the signal 
        along the z-axis will be processed. Default is 1.0 (100%).

    Returns
    -------
    tuple
        (angle_yz, angle_xz) - Tilt angles in degrees:
        - angle_yz: Tilt angle in the YZ plane (around X axis)
        - angle_xz: Tilt angle in the XZ plane (around Y axis)

    Raises
    ------
    ValueError
        If volumeType is not 'XCT' or 'UT'.

    Notes
    -----
    The function fits a plane of the form z = ax + by + c to the detected surface points.
    The coefficients a and b determine the tilt angles relative to the XZ and YZ planes,
    respectively. This provides a more accurate orientation estimate than simple 2D projections
    since it uses the full 3D point cloud of the surface.
    """
    # Choose appropriate surface extraction method based on volume type
    if volumeType == 'UT':
        # Use the ultrasound-specific method to extract surface points
        surface_coords = UT_surface_coordinates(volume, signal_percentage)
    elif volumeType == 'XCT':
        # Use the XCT-specific method to extract surface points
        surface_coords = XCT_surface_coordinates(volume, signal_percentage)
    else:
        raise ValueError("volumeType must be either 'XCT' or 'UT'")

    # Extract coordinates from the surface points array
    Z = surface_coords[:, 0]  # Depth coordinate
    Y = surface_coords[:, 1]  # Height coordinate
    X = surface_coords[:, 2]  # Width coordinate

    # Fit a plane z = ax + by + c using least squares
    # This creates a matrix [X Y 1] for solving the system of equations
    A = np.c_[X, Y, np.ones_like(X)]
    
    # Solve for coefficients [a, b, c] using least squares
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = C  # Unpack plane coefficients
    
    # The coefficients a and b represent the partial derivatives dz/dx and dz/dy,
    # which correspond to the slopes in the XZ and YZ planes
    
    # Convert slopes to angles in degrees
    angle_yz = np.degrees(np.arctan(b))  # Tilt relative to YZ plane (around X axis)
    angle_xz = np.degrees(np.arctan(a))  # Tilt relative to XZ plane (around Y axis)

    return angle_yz, angle_xz
