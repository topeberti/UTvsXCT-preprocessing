from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks
from tqdm import tqdm  # Progress bar for long operations
from PIL import Image
from skimage.filters import threshold_otsu
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import regionprops
import cv2
import scipy.ndimage
from joblib import Parallel, delayed

def UT_surface_coordinates(volume_UT, signal_percentage=1.0):
    """
    Extracts the surface coordinates from an Ultrasound (UT) volume.

    This function analyzes an ultrasound volume to identify the front surface (typically
    the first strong reflector) by finding the maximum intensity along each vertical column.
    Only points with intensity above a threshold are considered valid surface points.

    Parameters
    ----------
    volume_UT : numpy.ndarray
        3D ultrasound volume with shape (X, Y, Z), where:
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
        the coordinates (x, y, z) of a detected surface point.
        
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
    
    surface_coords = []  # List to store (x, y, z) surface coordinates

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
                surface_coords.append((x, y, z))

    # Check if any surface points were found
    if not surface_coords:
        raise ValueError("No surface points found above the threshold.")

    # Convert list of points to numpy array
    return np.array(surface_coords)

def PAUT_surface_coordinates(volume_PAUT, signal_percentage=1.0, min_depth = 0, max_depth = None):
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

    if max_depth is None:
        max_depth = volume_PAUT.shape[0]

    threshold = 60  # Minimum amplitude to consider as surface
    y_max = volume_PAUT.shape[1]
    x_max = volume_PAUT.shape[2]

    # Calculate the maximum z-index to consider based on the percentage
    z_max = volume_PAUT.shape[0]
    z_limit = int(z_max * signal_percentage)
    surface_coords = []  # List to store (z, y, x) surface coordinates

    # Loop over each x-y location in the volume
    for x in tqdm(range(x_max), desc="Finding surface points"):
        for y in range(y_max):
            A_scan = volume_PAUT[:z_limit, y, x]  # Signal along z-axis

            max_val = np.max(A_scan)     # Maximum value in the signal

            # If the max value is above the threshold, store the index
            if max_val > threshold:
                z = np.argmax(A_scan)    # Position of the max value
                if min_depth <= z <= max_depth:
                    surface_coords.append((z, y, x))

    if not surface_coords:
        raise ValueError("No surface points found above the threshold.")

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
        3D XCT volume with shape (X, Y, Z), where:
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
        the coordinates (x, y, z) of a detected surface point.

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
        surface_coords.append((x, y, z))
        
    # Convert list of points to numpy array
    return np.array(surface_coords)
    

def YZ_XZ_inclination(volume, volumeType='XCT', signal_percentage=1.0, min_depth = 0, max_depth = None):
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
    elif volumeType == 'PAUT':
        # Use the ultrasound-specific method to extract surface points
        surface_coords = PAUT_surface_coordinates(volume, signal_percentage, min_depth, max_depth)
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


#### MONOELEMENT VS XCT REGISTERING FUNCTIONS ####

def calculate_new_dimensions(original_resolution, new_resolution, original_dimensions):
    """
    Calculate new image dimensions when changing resolution.
    
    This function computes the new pixel dimensions required when converting
    an image from one resolution to another while maintaining the same
    real-world size.
    
    Parameters
    ----------
    original_resolution : float
        Original pixel size in real-world units (e.g., mm/pixel).
    new_resolution : float
        Target pixel size in real-world units (e.g., mm/pixel).
    original_dimensions : tuple of int
        Original image dimensions as (width, height) in pixels.
    
    Returns
    -------
    tuple of int
        New dimensions as (new_width, new_height) in pixels.
    
    Examples
    --------
    >>> calculate_new_dimensions(0.1, 0.05, (100, 200))
    (200, 400)
    """
    # Calculate the original dimensions in real-world units
    original_width, original_height = original_dimensions
    real_world_width = original_width * original_resolution
    real_world_height = original_height * original_resolution

    # Calculate the new dimensions in pixels
    new_width = int(real_world_width / new_resolution)
    new_height = int(real_world_height / new_resolution)

    return new_width, new_height

def resize_image(original_image, size, show=False):
    """
    Resize a PIL Image to the specified dimensions.
    
    This function resizes a PIL Image object to new dimensions and optionally
    displays the original and new sizes. The resized image is returned as a
    numpy array.
    
    Parameters
    ----------
    original_image : PIL.Image
        The original PIL Image object to be resized.
    size : tuple of int
        Target dimensions as (width, height) in pixels.
    show : bool, optional
        If True, prints the original and resized image dimensions.
        Default is False.
    
    Returns
    -------
    numpy.ndarray
        The resized image as a numpy array.
    
    Examples
    --------
    >>> from PIL import Image
    >>> img = Image.new('RGB', (100, 200))
    >>> resized = resize_image(img, (50, 100), show=True)
    The original image size is 100 wide x 200 tall
    The resized image size is 50 wide x 100 tall
    """
    width, height = original_image.size
    if show:
        print(f"The original image size is {width} wide x {height} tall")

    resized_image = original_image.resize(size)
    width, height = resized_image.size
    if show:
        print(f"The resized image size is {width} wide x {height} tall")
    return np.array(resized_image)

def find_rectangle_centers(image):
    """
    Find the center coordinates of rectangular objects in a binary image.
    
    This function uses OpenCV's connected components analysis to identify
    separate objects in a binary image and returns their centroid coordinates.
    The background (first component) is ignored.
    
    Parameters
    ----------
    image : numpy.ndarray
        Binary image where objects are represented by True/1 values and
        background by False/0 values.
    
    Returns
    -------
    numpy.ndarray
        Array of centroid coordinates with shape (N, 2), where N is the
        number of detected objects. Each row contains (x, y) coordinates.
    
    Raises
    ------
    AssertionError
        If the input image is not binary (contains values other than 0 and 1).
    
    Notes
    -----
    This function expects a strictly binary image and will raise an assertion
    error if non-binary values are detected.
    """
    # Ensure the image is binary
    assert np.array_equal(image, image.astype(bool)), "Image must be binary"

    # Find connected components in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image.astype('uint8'))

    # The first component is the background, so ignore it
    return centroids[1:]

def paint_binary_points(shape, points):
    """
    Create a binary image with points marked at specified coordinates.
    
    This function creates a binary image of the specified shape and marks
    points at the given coordinates with white pixels (value 255).
    
    Parameters
    ----------
    shape : tuple of int
        Desired image shape as (height, width).
    points : array-like
        Array of point coordinates with shape (N, 2), where each row
        contains (y, x) coordinates of a point to mark.
    
    Returns
    -------
    numpy.ndarray
        Binary image of type uint8 with points marked as white pixels (255)
        and background as black pixels (0).
    
    Notes
    -----
    Point coordinates are rounded to the nearest integer before painting.
    The function expects coordinates in (y, x) format but internally
    swaps them for proper indexing.
    """
    # Create an empty image of the specified shape
    image = np.zeros(shape, dtype=np.uint8)

    # Iterate over the points
    for point in points:
        # Round the coordinates to the nearest integer
        y, x = tuple(int(round(coord)) for coord in point)
        # Draw the point on the image
        image[x, y] = 255

    return image.astype(np.uint8)

def label_objects(image):
    """
    Label detected objects in a binary image with different intensity values.
    
    This function assigns different grayscale values to connected objects
    in a binary image based on their position. The topmost object gets
    value 100, the leftmost remaining object gets value 175, and all
    other objects get value 255.
    
    Parameters
    ----------
    image : numpy.ndarray
        Binary image containing connected objects to be labeled.
    
    Returns
    -------
    numpy.ndarray
        Labeled image where:
        - Background pixels remain 0
        - Topmost object pixels have value 100
        - Leftmost remaining object pixels have value 175
        - All other object pixels have value 255
    
    Notes
    -----
    This function is specifically designed for scenarios where you need
    to distinguish between a limited number of objects based on their
    spatial arrangement (top, left, others).
    """
    labeled_image, num_features = label(image)
    output_image = np.zeros_like(image)

    # Get all non-zero pixel coordinates
    indices = np.where(labeled_image > 0)
    indices = list(zip(indices[0], indices[1]))
    
    # Label the object nearest to the top edge (smallest row index)
    top_object = min(indices, key=lambda x: x[0])
    output_image[top_object] = 100
    indices.remove(top_object)

    if indices:
        # Label the object nearest to the left edge (smallest column index)
        left_object = min(indices, key=lambda x: x[1])
        output_image[left_object] = 175
        indices.remove(left_object)
    
    # Label the remaining objects with maximum intensity
    for idx in indices:
        output_image[idx] = 255
    
    return output_image.astype(np.uint8)

def find_holes_minimums(image, labeled_image):
    """
    Find the center coordinates of minimum intensity values in labeled regions.
    
    This function identifies the center of the minimum intensity value within
    each labeled region of an image. When multiple pixels share the minimum
    value, it selects the one closest to the geometric center of those pixels.
    
    Parameters
    ----------
    image : numpy.ndarray
        8-bit grayscale input image containing intensity values.
    labeled_image : numpy.ndarray
        Labeled image with distinct integer values for each region.
        Background should be 0, and each object should have a unique
        positive integer label.
    
    Returns
    -------
    numpy.ndarray
        Array of centroid coordinates with shape (N, 2), where N is the
        number of labeled regions. Each row contains (x, y) coordinates
        of the center of minimum intensity for each region.
    
    Notes
    -----
    The function uses skimage.measure.regionprops to analyze each labeled
    region and find the coordinates with minimum intensity values. If
    multiple pixels have the same minimum value, the algorithm selects
    the one closest to the mean position of all minimum pixels.
    """
    regions = regionprops(labeled_image, intensity_image=image)
    centers = []

    for region in regions:
        # Get the coordinates of the region
        coords = region.coords
        # Get the intensity values of the region
        intensities = image[coords[:, 0], coords[:, 1]]
        # Find the minimum intensity value
        min_value = np.min(intensities)
        # Get the coordinates of the minimum intensity value
        min_coords = coords[intensities == min_value]
        
        if len(min_coords) > 1:
            # If there are multiple minimums, find the most centered one
            center = np.mean(min_coords, axis=0)
            distances = np.linalg.norm(min_coords - center, axis=1)
            min_coords = min_coords[np.argmin(distances)]
        else:
            min_coords = min_coords[0]
        
        # Return coordinates in (x, y) format by reversing the order
        centers.append(min_coords[::-1])
    
    return np.array(centers)

def ut_preprocessing(ut):
    """
    Preprocess ultrasound volume to extract and locate pore centers.
    
    This function processes a 3D ultrasound volume to identify and locate
    the centers of the three largest pores or holes in the sample. The
    process involves maximum projection, Otsu thresholding, hole filling,
    and region analysis to find pore locations.
    
    Parameters
    ----------
    ut : numpy.ndarray
        3D ultrasound volume with shape (X, Y, Z) where Z is the depth
        axis and Y, X are the spatial dimensions.
    
    Returns
    -------
    numpy.ndarray
        Binary image with the same spatial dimensions as the input's Y-X
        plane, where pore centers are marked as white pixels (255) and
        background is black (0).
    
    Notes
    -----
    The algorithm follows these steps:
    1. Create maximum intensity projection along the Z-axis
    2. Apply Otsu thresholding for segmentation
    3. Fill holes in the binary mask
    4. Extract pores as holes within the filled regions
    5. Label connected components and select the 3 largest regions
    6. Find the minimum intensity centers within each region
    7. Paint the centers on a binary image
    """
    # Get the frontwall slice through maximum projection
    ut_max_proj = np.max(ut, axis=2)

    # Apply Otsu thresholding for automatic segmentation
    thresh = threshold_otsu(ut_max_proj)
    binary = ut_max_proj > thresh

    # Fill holes in the binary mask to create solid regions
    mask = binary_fill_holes(binary)

    # Invert the binary image to highlight dark regions
    inverted = np.invert(binary)

    # Extract pores as the intersection of filled regions and dark areas
    circles = np.logical_and(mask, inverted)

    # Label connected components to identify individual holes
    labeled, _ = label(circles)

    # Analyze region properties to find the largest pores
    props = regionprops(labeled)

    # Sort regions by area in descending order and get the 3 largest
    sorted_regions = sorted(props, key=lambda x: x.area, reverse=True)
    largest_regions_labels = [region.label for region in sorted_regions[:3]]

    # Create images to hold the results
    largest_regions_image = np.zeros_like(labeled)
    labeled_image = np.zeros_like(labeled)

    # Extract the three largest regions with their original intensities
    for lbl in largest_regions_labels:
        largest_regions_image[labeled == lbl] = ut_max_proj[labeled == lbl]
        labeled_image[labeled == lbl] = lbl

    # Find the centers of minimum intensity within each region
    ut_centers = find_holes_minimums(largest_regions_image, labeled_image)

    # Create a binary image marking the pore centers
    centers_painted_ut = paint_binary_points(largest_regions_image.shape, ut_centers)

    return centers_painted_ut.astype(ut.dtype)

def xct_preprocessing(xct, original_resolution=0.025, new_resolution=1.0):
    """
    Preprocess X-ray CT volume to extract and locate pore centers.
    
    This function processes a 3D X-ray CT volume to identify pores and
    resize the result to match ultrasound resolution for registration
    purposes. It performs segmentation, hole detection, and resizing.
    
    Parameters
    ----------
    xct : numpy.ndarray
        3D X-ray CT volume with shape (X, Y, Z) where Z is the depth
        axis and Y, X are the spatial dimensions.
    original_resolution : float, optional
        Original pixel size of the XCT data in real-world units
        (e.g., mm/pixel). Default is 0.025.
    new_resolution : float, optional
        Target pixel size for resizing in real-world units
        (e.g., mm/pixel). Default is 1.
    
    Returns
    -------
    numpy.ndarray
        Binary image with pore centers marked as white pixels (255)
        at the target resolution, ready for registration with UT data.
    
    Notes
    -----
    The algorithm follows these steps:
    1. Create maximum intensity projection along the Z-axis
    2. Calculate new dimensions based on resolution change
    3. Apply Otsu thresholding for segmentation
    4. Fill holes and extract pores as dark regions within solid areas
    5. Resize the binary pore image to target resolution
    6. Find pore centers using connected component analysis
    7. Paint centers on a binary image
    """
    # Get the maximum projection of the XCT so the holes are visible
    max_proj = np.max(xct, axis=2)

    # Calculate target dimensions for resolution conversion
    original_dimensions = (xct.shape[0], xct.shape[1])
    new_dimensions = calculate_new_dimensions(original_resolution, new_resolution, original_dimensions)

    # Segment and extract pores using Otsu thresholding
    thresh = threshold_otsu(max_proj)
    binary = max_proj > thresh

    # Fill holes in the binary mask
    mask = binary_fill_holes(binary)

    # Invert to highlight dark regions (pores)
    inverted = np.invert(binary)

    # Extract pores as intersection of filled regions and dark areas
    circles = np.logical_and(mask, inverted)

    # Resize the pore image to target resolution
    circles = resize_image(Image.fromarray(circles), new_dimensions[::-1])

    # Find pore centers using connected component analysis
    xct_centers = find_rectangle_centers(circles)

    # Create binary image marking the pore centers
    centers_painted_xct = paint_binary_points(circles.shape, xct_centers)

    return centers_painted_xct.astype(xct.dtype)

def extract_points(image):
    """
    Extract coordinates and intensity values of non-zero pixels from an image.
    
    This function scans through all pixels in an image and returns the
    coordinates and intensity values of pixels that have non-zero values.
    This is typically used to extract point locations from labeled or
    binary images.
    
    Parameters
    ----------
    image : numpy.ndarray
        2D image array where non-zero pixels represent points of interest.
    
    Returns
    -------
    numpy.ndarray
        Array with shape (N, 3) where N is the number of non-zero pixels.
        Each row contains [row, column, intensity_value] for each point.
    
    Notes
    -----
    The function returns coordinates in (row, column) format, which
    corresponds to (y, x) in image coordinate systems. The third column
    contains the pixel intensity value at that location.
    """
    points = []
    # Scan through all pixels in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > 0:
                # Store [row, column, intensity] for each non-zero pixel
                points.append([i, j, image[i, j]])
    return np.array(points)

def rigid_body_transformation_matrix(points_A, points_B):
    """
    Calculate the rigid body transformation matrix to align points_B to points_A.
    
    This function computes the optimal rigid body transformation (rotation and
    translation) that aligns a set of 2D points (points_B) to another set of
    2D points (points_A) using the Procrustes analysis algorithm with SVD.
    
    Parameters
    ----------
    points_A : numpy.ndarray
        Target point set with shape (N, 2), where N is the number of points
        and each row contains (x, y) coordinates.
    points_B : numpy.ndarray
        Source point set with shape (N, 2) that will be transformed to
        align with points_A.
    
    Returns
    -------
    numpy.ndarray
        3x3 homogeneous transformation matrix that can be used to transform
        points_B to align with points_A. The matrix includes rotation and
        translation components.
    
    Notes
    -----
    The algorithm uses the following steps:
    1. Compute centroids of both point sets
    2. Center the points by subtracting their centroids
    3. Compute the cross-covariance matrix
    4. Use SVD to find the optimal rotation matrix
    5. Ensure proper orientation (right-handed coordinate system)
    6. Calculate the translation vector
    7. Construct the homogeneous transformation matrix
    
    The transformation matrix can be applied to homogeneous coordinates
    or used with affine transformation functions.
    """
    # Compute the centroids of both sets of points

    centroid_A = np.mean(points_A, axis=0)
    centroid_B = np.mean(points_B, axis=0)

    # Center the points by subtracting the centroids
    centered_A = points_A - centroid_A
    centered_B = points_B - centroid_B
    
    # Compute the cross-covariance matrix
    H = np.dot(centered_B.T, centered_A)
    
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Compute the translation vector
    t = centroid_A - np.dot(R, centroid_B)
    
    # Construct the homogeneous transformation matrix
    transformation_matrix = np.eye(3)
    transformation_matrix[:2, :2] = R
    transformation_matrix[:2, 2] = t
    
    return transformation_matrix

def scale_transformation_matrix(transformation_matrix, scale_factor):
    """
    Scale the translation components of a 2D rigid body transformation matrix.
    
    This function applies a scaling factor to the translation components
    of a transformation matrix while preserving the rotation components.
    This is useful when working with different coordinate systems or
    pixel resolutions.
    
    Parameters
    ----------
    transformation_matrix : numpy.ndarray
        3x3 homogeneous transformation matrix containing rotation
        and translation components.
    scale_factor : float
        Scaling factor to apply to the translation components.
        Values > 1 increase translation distances, values < 1 decrease them.
    
    Returns
    -------
    numpy.ndarray
        3x3 scaled transformation matrix where only the translation
        components (last column, first two rows) have been scaled.
    
    Notes
    -----
    This function only scales the translation vector while keeping the
    rotation matrix unchanged. This is typically used when converting
    transformations between different pixel resolutions or coordinate
    systems where the spatial relationship needs to be preserved but
    the units need to be adjusted.
    """
    # Create a copy to avoid modifying the original matrix
    scaled_transformation_matrix = transformation_matrix.copy()

    # Scale only the translation components (first two elements of last column)
    scaled_transformation_matrix[:2, 2] *= scale_factor

    return scaled_transformation_matrix

def apply_transform_parameters_sequential(matrix, ut, xct, ut_resolution=1, xct_resolution=0.025):
    """
    Apply transformation matrix to XCT volume sequentially slice by slice.
    
    This function applies a 2D affine transformation to each slice of a 3D
    XCT volume sequentially. The output is resized to match the resolution
    and dimensions needed for registration with the UT volume.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        3x3 or 2x3 transformation matrix defining the affine transformation
        to apply to each XY slice of the volume.
    ut : numpy.ndarray
        UT volume used as reference for output dimensions, with shape (X, Y, Z).
    xct : numpy.ndarray
        XCT volume to be transformed, with shape (X, Y, Z) for 3D volumes
        or (Y, X) for 2D images.
    original_resolution : float, optional
        Original pixel resolution of the XCT data in real-world units.
        Default is 0.025.
    
    Returns
    -------
    numpy.ndarray
        Transformed volume with shape matching the target resolution.
        For 3D input, returns 3D output. For 2D input, returns 2D output.
    
    Notes
    -----
    This function processes slices sequentially, which may be slower than
    parallel processing but uses less memory. The transformation is applied
    to XY planes while preserving the Z dimension structure.
    
    The output shape is calculated to match the target resolution while
    maintaining the same real-world dimensions as the UT reference volume.
    """
    # Calculate output dimensions based on resolution scaling
    big_shape = calculate_new_dimensions(ut_resolution, xct_resolution, ut.shape[:2])

    transformed_volume = []

    if len(xct.shape) == 3:
        # Process 3D volume slice by slice
        for i in tqdm(range(xct.shape[2]), desc="Transforming slices",total=xct.shape[2]):
            # Apply transformation to each XY slice
            transformed_slice = scipy.ndimage.affine_transform(
                xct[:, :, i], matrix[:2, :], output_shape=big_shape
            )
            transformed_volume.append(transformed_slice)
        
        # Convert list to numpy array and rearrange axes
        transformed_volume = np.array(transformed_volume)

        return transformed_volume
    else:
        # Process 2D image
        # Apply transformation to the 2D image
        transformed_slice = scipy.ndimage.affine_transform(
            xct, matrix[:2, :], output_shape=big_shape
        )
        transformed_volume.append(transformed_slice)

        # Convert and rearrange axes
        transformed_volume = np.array(transformed_volume)

        return transformed_volume

def apply_transform_parameters_paralel(matrix, ut, xct, ut_resolution=1, xct_resolution=0.025):
    """
    Apply transformation matrix to XCT volume using parallel processing.
    
    This function applies a 2D affine transformation to each slice of a 3D
    XCT volume using parallel processing with joblib. This approach is
    faster than sequential processing for large volumes but uses more memory.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        3x3 or 2x3 transformation matrix defining the affine transformation
        to apply to each XY slice of the volume.
    ut : numpy.ndarray
        UT volume used as reference for output dimensions, with shape (X, Y, Z).
    xct : numpy.ndarray
        XCT volume to be transformed, with shape (X, Y, Z).
    original_resolution : float, optional
        Original pixel resolution of the XCT data in real-world units.
        Default is 0.025.
    
    Returns
    -------
    numpy.ndarray
        Transformed 3D volume with shape matching the target resolution.
    
    Notes
    -----
    This function uses joblib.Parallel to process multiple slices
    simultaneously across available CPU cores. The number of cores
    used is determined by n_jobs=-1 (all available cores).
    
    A progress bar using tqdm would be shown during processing to
    track the transformation progress across all slices.
    """
    # Calculate output dimensions based on resolution scaling
    big_shape = calculate_new_dimensions(ut_resolution, xct_resolution, ut.shape[:2])

    def func(slice, matrix, shape):
        """Apply affine transform to a single slice."""
        return scipy.ndimage.affine_transform(slice, matrix, output_shape=shape)

    def process_slice(z, volume, matrix, shape):
        """Extract and process a single Z-slice from the volume."""
        current_slice = volume[:, :, z]  # Extract the (X, Y) slice
        return func(current_slice, matrix, shape)  # Apply the transformation

    # Use all available CPU cores for parallel processing
    n_jobs = -1
    n_slices = xct.shape[2]  # Number of slices in the Z dimension

    # Apply transformation to all slices in parallel
    transformed_volume = Parallel(n_jobs=n_jobs)(
        delayed(process_slice)(z, xct, matrix[:2, :], big_shape) for z in tqdm(range(n_slices), desc="Transforming slices", total=n_slices)
    )

    # Reassemble the transformed slices into the final 3D array
    transformed_volume = np.stack(transformed_volume, axis=2)

    return transformed_volume

def apply_transform_parameters(matrix, ut, xct, ut_resolution=1, xct_resolution=0.025, parallel=False):
    """
    Apply transformation matrix to XCT volume with choice of processing method.
    
    This is a wrapper function that allows choosing between sequential and
    parallel processing for applying a transformation matrix to an XCT volume.
    The choice depends on memory constraints and performance requirements.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        3x3 or 2x3 transformation matrix defining the affine transformation
        to apply to each XY slice of the volume.
    ut : numpy.ndarray
        UT volume used as reference for output dimensions, with shape (X, Y, Z).
    xct : numpy.ndarray
        XCT volume to be transformed, with shape (X, Y, Z) for 3D volumes
        or (Y, X) for 2D images.
    original_resolution : float, optional
        Original pixel resolution of the XCT data in real-world units.
        Default is 0.025.
    parallel : bool, optional
        If True, uses parallel processing (faster but more memory intensive).
        If False, uses sequential processing (slower but memory efficient).
        Default is False.
    
    Returns
    -------
    numpy.ndarray
        Transformed volume with shape matching the target resolution.
    
    Notes
    -----
    Choose parallel=True for faster processing when:
    - Working with large volumes
    - Multiple CPU cores are available
    - Memory usage is not a constraint
    
    Choose parallel=False for memory efficiency when:
    - Working with limited memory
    - Single-core processing is preferred
    - Memory usage needs to be minimized
    """
    if parallel:
        return apply_transform_parameters_paralel(matrix, ut, xct, ut_resolution, xct_resolution)
    else:
        return apply_transform_parameters_sequential(matrix, ut, xct, ut_resolution, xct_resolution)

def register_ut_xct_monoelement(ut, xct, reference_resolution = 1, registered_resolution = 0.025):
    """
    Register ultrasound and X-ray CT volumes using pore-based alignment.
    
    This function performs automatic registration between UT and XCT volumes
    by detecting pores in both modalities and computing the rigid body
    transformation needed to align them. The registration is based on
    matching the three largest pores found in both volumes.
    
    Parameters
    ----------
    ut : numpy.ndarray
        3D ultrasound volume with shape (Z,Y,X) where Z is the depth
        axis and Y, X are the spatial dimensions.
    xct : numpy.ndarray
        3D X-ray CT volume with shape (Z,Y,X) where Z is the depth
        axis and Y, X are the spatial dimensions.
    
    Returns
    -------
    tuple
        A tuple containing:
        - parameters (numpy.ndarray): 3x3 transformation matrix scaled for
          application to full-resolution XCT data
        - ut_centers (numpy.ndarray): Labeled UT pore centers image
        - xct_centers (numpy.ndarray): Labeled XCT pore centers image
        - transformed_xct_centers (numpy.ndarray): XCT centers transformed
          to UT coordinate system for verification
    
    Raises
    ------
    ValueError
        If fewer than 3 pores are detected in either volume, or if the
        number of detected pores differs between UT and XCT volumes.
    
    Notes
    -----
    The registration process follows these steps:
    1. Preprocess both volumes to extract pore locations
    2. Label pores based on spatial arrangement (top, left, others)
    3. Extract point coordinates from labeled images
    4. Sort points by their labels to ensure correspondence
    5. Compute rigid body transformation between point sets
    6. Scale transformation for application to full-resolution data
    7. Apply transformation to XCT centers for verification
    
    The function requires at least 3 corresponding pores in both volumes
    for robust registration. The pores are matched based on their spatial
    arrangement rather than geometric proximity.
    """
    print('Preprocessing')

    
    ut = np.swapaxes(ut, 0, 1)
    ut = np.swapaxes(ut, 1, 2)

    xct = np.swapaxes(xct, 0, 1)
    xct = np.swapaxes(xct, 1, 2)

    # Extract and label pore centers from both volumes
    ut_centers = label_objects(ut_preprocessing(ut))
    xct_centers = label_objects(xct_preprocessing(xct,original_resolution = registered_resolution, new_resolution = reference_resolution))


    # Extract point coordinates and intensity values from labeled images
    ut_points = extract_points(ut_centers)
    xct_points = extract_points(xct_centers)


    # Sort points by their intensity labels to ensure correspondence
    sorted_indices_ut = np.argsort(ut_points[:, -1])
    sorted_indices_xct = np.argsort(xct_points[:, -1])

    # Apply sorting to get corresponding point sets
    sorted_ut_points = ut_points[sorted_indices_ut]
    sorted_xct_points = xct_points[sorted_indices_xct]

    # Validate that we have sufficient points for registration
    if len(sorted_ut_points) < 3:
        print('Not enough UT points')
        raise ValueError('Not enough UT points')

    if len(sorted_xct_points) < 3:
        print('Not enough XCT points')
        raise ValueError('Not enough XCT points')

    # Ensure we have the same number of corresponding points
    if len(sorted_ut_points) != len(sorted_xct_points):
        print('Different number of points')
        raise ValueError('Different number of points')
    
    print('Preprocessed')

    print('Registering')

    # Compute transformation from XCT to UT coordinate system
    transformation_matrix = rigid_body_transformation_matrix(
        sorted_xct_points[:, :2], sorted_ut_points[:, :2]
    )
    
    # Scale transformation for application to full-resolution XCT data
    scaled_transformation_matrix = scale_transformation_matrix(transformation_matrix, reference_resolution/registered_resolution)

    print('Registered')

    # Store the scaled transformation as the final parameters
    parameters = scaled_transformation_matrix

    # Apply transformation to XCT centers for verification
    transformed_xct_centers = apply_transform_parameters(
        transformation_matrix, ut, ((xct_centers > 0) * 255).astype(xct.dtype), ut_resolution=reference_resolution, xct_resolution=reference_resolution
    ) # We use the reference resolution twice because we are reshaping the resized XCT

    return parameters, ut_centers, xct_centers, transformed_xct_centers

def apply_registration(ut, xct, parameters, ut_resolution, xct_resolution, parallel=False):
    """
    Apply pre-computed registration parameters to align XCT volume with UT volume.
    
    This function applies a previously computed transformation matrix to
    register an XCT volume with a UT volume. It is typically used after
    registration parameters have been computed using register_ut_xct_monoelement().
    
    Parameters
    ----------
    ut : numpy.ndarray
        Reference UT volume with shape (X, Y, Z) used to determine output
        dimensions and coordinate system.
    xct : numpy.ndarray
        XCT volume to be transformed and registered, with shape (X, Y, Z).
    parameters : numpy.ndarray
        3x3 transformation matrix obtained from the registration process,
        containing rotation, translation, and scaling components.
    
    Returns
    -------
    numpy.ndarray
        Transformed XCT volume aligned with the UT coordinate system.
        The output has the same dimensions and coordinate system as the
        reference UT volume.
    
    Notes
    -----
    This function is designed to be used as the second step in a two-stage
    registration process:
    1. First, use register_ut_xct_monoelement() to compute registration parameters
    2. Then, use this function to apply those parameters to the full XCT volume
    
    The transformation preserves the spatial relationships while aligning
    the XCT data to match the UT coordinate system and resolution.
    """
    transformation_matrix = parameters

    print('Applying transformation')

    # Apply the transformation matrix to the entire XCT volume
    transformed_volume = apply_transform_parameters(transformation_matrix, ut, xct, ut_resolution, xct_resolution, parallel)

    print('Transformation applied')

    return transformed_volume