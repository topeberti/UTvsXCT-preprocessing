import numpy as np
import cv2
from pathlib import Path
import tifffile as tiff
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
import math
from scipy.signal import find_peaks
from utilities.onlypores import material_mask  # Custom module


def UT_surface_coordinates(volume_UT):
    """
    Extracts the surface coordinates from an Ultrasound (UT) volume.

    For each (x, y) column in the volume, it takes the A-scan (signal along z),
    finds the maximum value, and keeps it only if it exceeds a threshold.
    Returns the coordinates (z, y, x) of the surface points.

    Parameters:
        volume_UT (numpy.ndarray): 3D UT volume (z, y, x)

    Returns:
        numpy.ndarray: Array of surface points (z, y, x)
    """
    threshold = 60  # Minimum amplitude to consider as surface
    y_max = volume_UT.shape[1]
    x_max = volume_UT.shape[2]

    surface_coords = []  # List to store (z, y, x) surface coordinates

    # Loop over each x-y location in the volume
    for x in range(x_max):
        for y in range(y_max):
            A_scan = volume_UT[:, y, x]  # Signal along z-axis
            max_val = np.max(A_scan)     # Maximum value in the signal

            # If the max value is above the threshold, store the index
            if max_val > threshold:
                z = np.argmax(A_scan)    # Position of the max value
                surface_coords.append((z, y, x))

    if not surface_coords:
        raise ValueError("No surface points found above the threshold.")

    return np.array(surface_coords)

def XCT_surface_coordinates(volume_XCT):
    """
    Extracts the surface coordinates from a X-ray CT (XCT) volume.

    For each (x, y) column in the volume, it finds the first peak in the 
    signal along z that exceeds a fixed threshold. This peak is considered 
    the surface point.

    Parameters:
        volume_XCT (numpy.ndarray): 3D XCT volume (z, y, x)

    Returns:
        numpy.ndarray: Array of surface points (z, y, x)
    """

    y_max = volume_XCT.shape[1]
    x_max = volume_XCT.shape[2]

    surface_coords = []  # List to store (z, y, x) surface coordinates
    threshold = 175      # Minimum peak height to be considered

    # Loop over each x-y location in the volume
    for x in range(x_max):
        for y in range(y_max):
            xct_signal = volume_XCT[:, y, x]  # Signal along z-axis

            # Find peaks in the signal above the threshold
            peaks, properties = find_peaks(xct_signal, height=threshold)

            # If any peaks are found, store the first one
            if len(peaks) > 0:
                z = peaks[0]  # First peak index
                surface_coords.append((z, y, x))

    if not surface_coords:
        raise ValueError("No surface points found above the threshold (175).")

    return np.array(surface_coords)


def YZ_XZ_inclination(volume, volumeType='XCT'):
    """
    Calculates the inclination of the surface in a 3D volume with respect to
    the YZ and XZ planes by fitting a plane to the extracted surface points.

    Depending on the volumeType ('XCT' or 'UT'), it calls the appropriate
    surface extraction function, fits a plane z = ax + by + c, and computes
    the tilt angles from the plane coefficients.

    Parameters:
        volume (numpy.ndarray): 3D image volume (z, y, x)
        volumeType (str): 'XCT' for CT scan, 'UT' for ultrasound

    Returns:
        tuple: (angle_yz, angle_xz) in degrees
    """

    # Choose appropriate method based on volume type
    if volumeType == 'UT':
        surface_coords = UT_surface_coordinates(volume)
    elif volumeType == 'XCT':
        surface_coords = XCT_surface_coordinates(volume)
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
