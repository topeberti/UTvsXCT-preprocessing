"""
Pore Detection and Material Segmentation Module

This module provides functions for detecting pores in 3D volumes using various thresholding techniques.
It includes functionality for Sauvola and Otsu thresholding, material masking, and pore extraction
from X-ray CT scan data. The module supports parallel processing to improve performance on large datasets.
"""

import numpy as np
from skimage import measure
from skimage.measure import regionprops
from skimage import filters
import fill_voids
from joblib import Parallel, delayed
from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects
from skimage.measure import label
import psutil

def sauvola_thresholding_concurrent(volume, window_size, k):
    """
    Apply Sauvola thresholding in a 3D volume along axis 1 (y-axis).
    
    Sauvola thresholding is an adaptive thresholding method that works well for images with varying
    background intensity. This function applies the algorithm slice by slice to a 3D volume and then
    removes small objects to reduce noise.

    Parameters:
    - volume: 3D numpy array - Input volume to be thresholded
    - window_size: int - Size of the local window for Sauvola thresholding (default: 49)
    - k: float - Parameter that controls threshold sensitivity (default: 0.05)
    - min_size: int - Minimum size of objects to keep after thresholding (default: 100)

    Returns:
    - binary_volume: 3D binary numpy array after Sauvola thresholding
    """

    #if window size is even, increase it by 1 to make it odd
    # this is necessary for Sauvola thresholding implentation
    if window_size % 2 == 0:
        window_size += 1

    def sauvola_slice(slice_):
        """Apply Sauvola thresholding to a single 2D slice."""
        sauvola_threshold = threshold_sauvola(slice_, window_size=window_size, k=k, r=128)
        # Values below threshold are considered features (pores)
        return slice_ > sauvola_threshold

    # Apply Sauvola thresholding to each slice along axis 1 (y-axis) in parallel
    binary_volume = np.array(Parallel(n_jobs=-1)(delayed(sauvola_slice)(volume[:, i, :]) for i in range(volume.shape[1])))

    # The result shape is (num_y, z, x), so we need to transpose axes to (z, y, x)
    binary_volume = np.transpose(binary_volume, (1, 0, 2))

    return binary_volume

def sauvola_thresholding_nonconcurrent(volume, window_size, k):
    """
    Apply Sauvola thresholding in a 3D volume along axis 1 (y-axis),
    in a non-concurrent and memory-efficient way.

    Parameters:
    - volume: 3D numpy array - Input volume to be thresholded
    - window_size: int - Size of the local window for Sauvola thresholding
    - k: float - Parameter that controls threshold sensitivity

    Returns:
    - binary_volume: 3D binary numpy array after Sauvola thresholding
    """
    if window_size % 2 == 0:
        window_size += 1
    binary_volume = np.zeros_like(volume, dtype=bool)
    for i in range(volume.shape[1]):
        slice_ = volume[:, i, :]
        sauvola_threshold = threshold_sauvola(slice_, window_size=window_size, k=k, r=128)
        binary_volume[:, i, :] = slice_ > sauvola_threshold
    return binary_volume

def sauvola_thresholding(volume, window_size, k):
    """
    Apply Sauvola thresholding to a 3D volume, automatically choosing concurrent or non-concurrent
    implementation based on available system memory.

    Parameters:
    - volume: 3D numpy array - Input volume to be thresholded
    - window_size: int - Size of the local window for Sauvola thresholding
    - k: float - Parameter that controls threshold sensitivity

    Returns:
    - binary_volume: 3D binary numpy array after Sauvola thresholding
    """
    # Estimate memory required for concurrent processing (rough estimate: 2x input size)
    required_mem_bytes = volume.nbytes * 2
    available_mem_bytes = psutil.virtual_memory().available
    if available_mem_bytes > required_mem_bytes:
        return sauvola_thresholding_concurrent(volume, window_size, k)
    else:
        return sauvola_thresholding_nonconcurrent(volume, window_size, k)

def otsu_thresholding(volume):
    """
    Apply Otsu thresholding to a 3D volume.

    Parameters:
    - volume: 3D numpy array

    Returns:
    - binary_volume: 3D binary numpy array after Otsu thresholding
    """

    # Apply Otsu thresholding on the non-zero values
    threshold_value = filters.threshold_otsu(volume)

    print('thresholding with value: ', threshold_value)

    binary = volume > threshold_value

    return binary

def onlypores(xct, frontwall = 0, backwall = 0, sauvola_radius = 30, sauvola_k = 0.125):
    """
    Extract pores from a 3D volume using Sauvola thresholding.

    Parameters:
    - xct: 3D numpy array - Input volume
    - frontwall: int - Index of the front wall (default: 0)
    - backwall: int - Index of the back wall (default: 0)

    Returns:
    - onlypores: 3D binary numpy array - Detected pores
    - sample_mask: 3D binary numpy array - Material mask
    - binary: 3D binary numpy array - Thresholded volume
    """
    print('masking')
    
    # Find the bounding box of non-zero values
    non_zero = np.where(xct > 0)
    if len(non_zero[0]) == 0:  # If there are no non-zero values
        print("No non-zero values found in the volume")
        return None, None, None
        
    # Get the min and max indices for each dimension
    min_z, max_z = np.min(non_zero[0]), np.max(non_zero[0])
    min_y, max_y = np.min(non_zero[1]), np.max(non_zero[1])
    min_x, max_x = np.min(non_zero[2]), np.max(non_zero[2])
    
    # Crop the volume to the bounding box with a small margin (if possible)
    margin = 2  # Add a small margin for the filter window
    min_z = max(0, min_z - margin)
    min_y = max(0, min_y - margin)
    min_x = max(0, min_x - margin)
    max_z = min(xct.shape[0] - 1, max_z + margin)
    max_y = min(xct.shape[1] - 1, max_y + margin)
    max_x = min(xct.shape[2] - 1, max_x + margin)
    
    # Extract the cropped volume
    cropped_volume = xct[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    
    print('Thresholding')
    # Apply Sauvola thresholding to the cropped volume
    binary_cropped = sauvola_thresholding(cropped_volume, window_size=sauvola_radius, k=sauvola_k)

    #set to zero all the slices before the front wall and after the back wall
    if frontwall > 0:
        binary_cropped[:frontwall, :, :] = True
    if backwall > 0:
        binary_cropped[backwall:, :, :] = True
    
    # Create a binary volume of original size filled with False
    binary = np.zeros(xct.shape, dtype=bool)
    
    # Place the processed data back into the full-sized array
    binary[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = binary_cropped
    
    sample_mask_cropped = material_mask(cropped_volume)
    
    sample_mask = np.zeros_like(binary)
    sample_mask[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = sample_mask_cropped
    
    # Invert binary
    binary_inverted = np.invert(binary)
    onlypores = np.logical_and(binary_inverted, sample_mask)
    
    return onlypores, sample_mask, binary

def material_mask(xct): #Material mask but not parallel
    """
    Generate a material mask for a 3D volume.

    Parameters:
    - xct: 3D numpy array - Input volume

    Returns:
    - sample_mask: 3D binary numpy array - Material mask
    """
    
    threshold_value = filters.threshold_otsu(xct)
    binary = xct > threshold_value
    max_proj = np.max(binary, axis=0)
    labels = measure.label(max_proj)
    props = regionprops(labels)
    minr, minc, maxr, maxc = props[0].bbox
    binary_cropped = binary[:, minr:maxr, minc:maxc]
    sample_mask_cropped = fill_voids.fill(binary_cropped, in_place=False)
    sample_mask = np.zeros_like(binary)
    sample_mask[:, minr:maxr, minc:maxc] = sample_mask_cropped

    return sample_mask