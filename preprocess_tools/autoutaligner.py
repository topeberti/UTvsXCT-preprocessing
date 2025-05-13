import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
from skimage import feature
from scipy.signal import hilbert

def get_lines2( image):
    # get the first and last pixel of the image
    linea = np.where(image >= 1)
    x0 = linea[0][0]
    x1 = linea[0][-1]
    y0 = linea[1][0]
    y1 = linea[1][-1]

    # create a image with the same size as the input image
    image2 = np.zeros_like(image)

    import cv2

    # draw the line on the image2
    cv2.line(image2, (y0, x0), (y1, x1), (1, 1, 1), 2)

    return image2

def find_brightest_ut(volume):

    """
    This function finds the brightest slice in a 3D volume.
    
    Parameters:
    volume (numpy.ndarray): 3D array representing the volume of an image.
    
    Returns:
    int: Index of the brightest slice in the volume.
    """

    ut = volume.copy()

    ut = np.swapaxes(ut, 0, 1)
    ut = np.swapaxes(ut, 1, 2)

    cropped = crop_image_center(ut)

    # Find the brightest slice
    brightest_slice = np.argmax(np.sum(cropped, axis=(0, 1)))

    return brightest_slice

def auto_gate(volume):
    """
    This function finds the range around the brightest slice in a 3D volume.
    
    Parameters:
    volume (numpy.ndarray): 3D array representing the volume of an image.
    
    Returns:
    tuple: Range around the brightest slice (brightest_slice - 5, brightest_slice + 5).
    """
    # Find the brightest slice
    brightest_slice = find_brightest_ut(volume)

    resolution = 0.02

    range = int(np.round(0.1/resolution))

    return (brightest_slice - range, brightest_slice + range)

def crop_image_center(image):
    """
    This function crops the center of an image.
    
    Parameters:
    image (numpy.ndarray): 2D array representing the image.
    
    Returns:
    numpy.ndarray: Cropped image.
    """
    # Get the height and width of the image
    height, width = image.shape[:2]

    # Calculate the new height
    new_height = height // 2

    # Calculate the starting point for the height crop
    start_height = height // 4

    # Crop the image
    cropped_image = image[start_height:start_height+new_height, :]

    return cropped_image

def crop_volume(volume):
    """
    This function crops the center of each slice in a 3D volume.
    
    Parameters:
    volume (numpy.ndarray): 3D array representing the volume of an image.
    
    Returns:
    numpy.ndarray: Volume with each slice cropped.
    """
    aux = crop_image_center(volume[0])
    cropped = np.zeros((volume.shape[0], aux.shape[0], aux.shape[1]), dtype=volume.dtype)
    for i in range(volume.shape[0]):
        cropped[i] = crop_image_center(volume[i])
    
    return cropped

def angle_max(volume):
    """
    This function calculates the rotation angle of the largest component in a 3D volume.
    
    Parameters:
    volume (numpy.ndarray): 3D array representing the volume of an image.
    
    Returns:
    float: Rotation angle of the largest component in degrees.
    """
    max_proj = np.max(volume, axis=0)

    middle_slice = max_proj

    #otsu threshold
    threshold_value = threshold_otsu(middle_slice)
    print('threshold value is: ', threshold_value)
    thresholded_slice = middle_slice > threshold_value

    # Label the objects in the thresholded slice
    labeled_slice = label(thresholded_slice)

    # Get the properties of each labeled region
    regions = regionprops(labeled_slice)

    # Find the largest connected component
    largest_component = max(regions, key=lambda region: region.area)

    # Create a mask to keep only the largest component
    mask = np.zeros_like(labeled_slice)
    mask[labeled_slice == largest_component.label] = 1
    mask = binary_fill_holes(mask).astype(int)

    # Apply the mask to the thresholded slice
    thresholded_slice = thresholded_slice * mask

    #fill holes in the mask
    mask = binary_fill_holes(mask).astype(int)

    # extract edges using canny edge detector
    mask = feature.canny(mask > 0, sigma=0) > 0

    # Label the objects in the thresholded slice
    mask = label(mask)

    # Get the properties of each labeled region
    regions = regionprops(mask)

    # Find the largest connected component if there is one, if not the first
    if len(regions) == 1:
        largest_component = regions[0]
    else:
        largest_component = sorted(regions, key=lambda region: region.area)[-1]

    # delete everything that is not the largest component from mask
    mask[mask != largest_component.label] = 0
    try:
        mask = get_lines2(mask)
    except:
        print("line not smoothed")

    # Compute the rotation angle of the largest component
    rotation_angle = largest_component.orientation

    # Convert the angle from radians to degrees
    rotation_angle_degrees = np.degrees(rotation_angle)

    # Print the rotation angle
    print(
        f"The rotation angle of the largest component is {rotation_angle_degrees} degrees."
    )

    return -rotation_angle_degrees

def is_RF(volume):
    """
    This function checks if the maximum value in a 3D volume is greater than 128.
    
    Parameters:
    volume (numpy.ndarray): 3D array representing the volume of an image.
    
    Returns:
    bool: True if the maximum value is greater than 128, False otherwise.
    """

    if np.max(volume) > 128:
        return True
    
    return False

def hillbert_transform(volume):

    """
    This function applies the Hilbert transform to a 3D volume.
    
    Parameters:
    volume (numpy.ndarray): 3D array representing the volume of an image.
    
    Returns:
    numpy.ndarray: Amplitude envelope of the Hilbert transform.
    """

    volume = volume.copy()

    volume = volume.astype(np.int16)

    volume = volume - 128

    data_hilbert = hilbert(volume, axis=0)
    amplitude_envelope = np.abs(data_hilbert).astype(np.uint8)

    return amplitude_envelope

def align(data, gate):
    """
    This function aligns the data in a 3D volume based on the maximum value in a gated range.
    
    Parameters:
    data (numpy.ndarray): 3D array representing the volume of an image.
    gate (tuple): Range for gating.
    
    Returns:
    numpy.ndarray: Aligned data.
    """
    # now do it for the whole volume
    rolled_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            signal = data[:, i, j]
            gated_data = signal[gate[0] : gate[1]]
            max_gated_data_index = np.argmax(np.abs(gated_data), axis=0)
            rolled = np.roll(signal, -max_gated_data_index)
            rolled_data[:, i, j] = rolled
    return rolled_data

def double_align(data,rf, gate):
    """
    This function aligns the data and RF signals in a 3D volume based on the maximum value in a gated range.
    
    Parameters:
    data (numpy.ndarray): 3D array representing the volume of an image.
    rf (numpy.ndarray): 3D array representing the RF signals.
    gate (tuple): Range for gating.
    
    Returns:
    tuple: Aligned data and RF signals.
    """
    # now do it for the whole volume
    rolled_data = np.zeros_like(data)
    rolled_RF = np.zeros_like(rf)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            signal = data[:, i, j]
            gated_data = signal[gate[0] : gate[1]]
            max_gated_data_index = np.argmax(np.abs(gated_data), axis=0)
            rolled = np.roll(signal, -max_gated_data_index)
            rolled_data[:, i, j] = rolled

            rfsignal = rf[:, i, j]
            rolled_rf_signal = np.roll(rfsignal, -max_gated_data_index)
            rolled_RF[:, i, j] = rolled_rf_signal

    return rolled_data,rolled_RF