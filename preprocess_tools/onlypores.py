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

def sauvola_thresholding(volume, window_size=49, k=0.05, min_size=100):
    """
    Apply Sauvola thresholding and remove small objects from a 3D volume.
    
    Sauvola thresholding is an adaptive thresholding method that works well for images with varying
    background intensity. This function applies the algorithm slice by slice to a 3D volume and then
    removes small objects to reduce noise.

    Parameters:
    - volume: 3D numpy array - Input volume to be thresholded
    - window_size: int - Size of the local window for Sauvola thresholding (default: 49)
    - k: float - Parameter that controls threshold sensitivity (default: 0.05)
    - min_size: int - Minimum size of objects to keep after thresholding (default: 100)

    Returns:
    - binary_volume: 3D binary numpy array after Sauvola thresholding and small object removal
    """

    def sauvola_slice(slice_):
        """Apply Sauvola thresholding to a single 2D slice."""
        sauvola_threshold = threshold_sauvola(slice_, window_size=window_size, k=k)
        # Values below threshold are considered features (pores)
        return slice_ < sauvola_threshold

    def remove_small_objects_slice(slice_):
        """Remove small objects from a binary 2D slice to reduce noise."""
        labeled_slice = label(slice_)  # Label connected components
        # Remove small objects and invert the result to keep pores
        return remove_small_objects(labeled_slice, min_size=min_size) <= 0

    # Apply Sauvola thresholding to each slice in parallel using all available cores (-1)
    sauvola_noisy = np.array(Parallel(n_jobs=-1)(delayed(sauvola_slice)(volume[i]) for i in range(volume.shape[0])))

    # Remove small objects from each slice in parallel to clean up the thresholded result
    binary_volume = np.array(Parallel(n_jobs=-1)(delayed(remove_small_objects_slice)(sauvola_noisy[i]) for i in range(sauvola_noisy.shape[0])))

    return binary_volume

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

def onlypores(xct, frontwall = 0, backwall = 0):
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
    
    print(f'Cropped volume shape: {cropped_volume.shape}')
    
    print('Thresholding')
    # Apply Sauvola thresholding to the cropped volume
    binary_cropped = sauvola_thresholding(cropped_volume)

    print(binary_cropped.max())

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

def material_mask_parallel(xct):
    """
    Generate a material mask for a 3D volume using parallel processing.

    Parameters:
    - xct: 3D numpy array - Input volume

    Returns:
    - sample_mask: 3D binary numpy array - Material mask
    """

    # Function to apply Otsu thresholding and process each chunk
    def process_chunk(xct_chunk):
        threshold_value = filters.threshold_otsu(xct_chunk)
        binary = xct_chunk > threshold_value
        max_proj = np.max(binary, axis=0)
        labels = measure.label(max_proj)
        props = regionprops(labels)
        minr, minc, maxr, maxc = props[0].bbox
        binary_cropped = binary[:, minr:maxr, minc:maxc]
        sample_mask_cropped = fill_voids.fill(binary_cropped, in_place=False)
        sample_mask = np.zeros_like(binary)
        sample_mask[:, minr:maxr, minc:maxc] = sample_mask_cropped
        return sample_mask

    print('computing otsu')

    # Number of chunks (adjust depending on the size of your array and available cores)
    num_chunks = 16  # You can increase or decrease this based on testing

    # Assuming xct is your large array, divide it into chunks
    chunks = np.array_split(xct, num_chunks)

    # Use joblib to apply the function in parallel
    sample_masks = Parallel(n_jobs=-1, backend='loky')(delayed(process_chunk)(chunk) for chunk in chunks)

    # Combine the results back into a single array
    sample_mask = np.concatenate(sample_masks, axis=0)

    return sample_mask

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