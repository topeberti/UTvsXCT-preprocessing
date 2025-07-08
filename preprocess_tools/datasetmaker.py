"""
Dataset Maker for UT vs XCT Preprocessing

This module provides tools for creating datasets from ultrasonic testing (UT) and 
X-ray computed tomography (XCT) data. It handles preprocessing, patching, and dataset 
generation for machine learning applications in non-destructive testing.

The main workflow includes:
1. Preprocessing images to align UT and XCT data
2. Dividing images into patches for analysis
3. Computing volumetric and area fractions of pores
4. Creating datasets for machine learning models

Author: Alberto Vicente
Date: 2025
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import measure
from skimage.measure import regionprops
from skimage.util import view_as_windows
from skimage.measure import label, regionprops
import scipy.ndimage
from joblib import Parallel, delayed


def divide_into_patches(image, patch_size, step_size):
    """
    Divide a 3D image into overlapping patches.
    
    This function takes a 3D image and divides it into smaller patches along the 
    spatial dimensions (keeping the z-dimension intact). The patches can overlap 
    based on the step size.
    
    Args:
        image (np.ndarray): 3D input image with shape (z, x, y)
        patch_size (int): Size of the square patches in the x-y plane
        step_size (int): Step size between patches (determines overlap)
        
    Returns:
        np.ndarray: Array of patches with shape (n_patches, z, patch_size, patch_size)
        
    Example:
        >>> patches = divide_into_patches(image, patch_size=64, step_size=32)
    """
    patches = view_as_windows(image, (image.shape[0], patch_size, patch_size), 
                             step=(image.shape[0], step_size, step_size))
    return patches.reshape(-1, image.shape[0], patch_size, patch_size)

def calculate_patch_shape(image_shape, patch_size, step_size):
    """
    Calculate the expected shape of patches array without actually creating patches.
    
    This function calculates how many patches will be generated and their dimensions
    when dividing an image into patches.
    
    Args:
        image_shape (tuple): Shape of the input image (z, x, y)
        patch_size (int): Size of square patches in x-y plane
        step_size (int): Step size between patches
        
    Returns:
        tuple: Expected shape (n_patches, z, patch_size, patch_size)
        
    Example:
        >>> shape = calculate_patch_shape((100, 512, 512), 64, 32)
    """
    # Calculate the number of patches along each dimension
    num_patches_h = ((image_shape[1] - patch_size) // step_size) + 1
    num_patches_w = ((image_shape[2] - patch_size) // step_size) + 1

    # The outgoing shape would be (num_patches_h, num_patches_w, image_shape[0], patch_size, patch_size)
    return (num_patches_h * num_patches_w, image_shape[0], patch_size, patch_size)


def calculate_pixels(ut_resolution, xct_resolution, ut_pixels):
    """
    Convert pixel dimensions from UT resolution to XCT resolution.
    
    This function converts the number of pixels in UT coordinate system to the 
    equivalent number of pixels in XCT coordinate system based on their respective
    resolutions.
    
    Args:
        ut_resolution (float): UT resolution in mm/pixel
        xct_resolution (float): XCT resolution in mm/pixel  
        ut_pixels (int): Number of pixels in UT coordinate system
        
    Returns:
        float: Equivalent number of pixels in XCT coordinate system
        
    Example:
        >>> xct_pixels = calculate_pixels(ut_resolution=1.0, xct_resolution=0.025, ut_pixels=3)
        >>> # Returns 120.0 (3 * 1.0 / 0.025)
    """
    # Calculate the ratio of the resolutions
    resolution_ratio = ut_resolution / xct_resolution 
    
    # Calculate the equivalent number of pixels in the xct resolution
    xct_pixels = ut_pixels * resolution_ratio
    
    return xct_pixels

def crop_image_to_patch_size(image, patch_size):
    """
    Crop a 3D image to make its spatial dimensions divisible by patch_size.
    
    This function removes pixels from the edges of the image to ensure that
    the x and y dimensions are evenly divisible by the patch size. The cropping
    is done symmetrically from both sides when possible.
    
    Args:
        image (np.ndarray): 3D input image with shape (z, x, y)
        patch_size (int): Target patch size for divisibility
        
    Returns:
        np.ndarray: Cropped image with spatial dimensions divisible by patch_size
        
    Example:
        >>> cropped = crop_image_to_patch_size(image, patch_size=64)
    """
    z, x, y = image.shape

    # Calculate how many pixels to remove from each dimension
    crop_x = x % patch_size
    crop_y = y % patch_size

    # Distribute cropping symmetrically
    crop_x_before = crop_x // 2
    crop_x_after = crop_x - crop_x_before

    crop_y_before = crop_y // 2
    crop_y_after = crop_y - crop_y_before

    # Apply cropping (handle edge case where crop_after is 0)
    cropped_image = image[:, crop_x_before:-crop_x_after or None, crop_y_before:-crop_y_after or None]

    return cropped_image

def nearest_lower_divisible(num, divisor):
    """
    Find the nearest lower number that is divisible by the divisor.
    
    Args:
        num (int/float): Input number
        divisor (int/float): Divisor value
        
    Returns:
        int: Nearest lower number divisible by divisor
        
    Example:
        >>> nearest_lower_divisible(127, 10)  # Returns 120
    """
    if num % divisor == 0:
        return num
    else:
        remainder = num % divisor
        return int(num - remainder)

def nearest_higher_divisible(num, divisor):
    """
    Find the nearest higher number that is divisible by the divisor.
    
    Args:
        num (int/float): Input number
        divisor (int/float): Divisor value
        
    Returns:
        int: Nearest higher number divisible by divisor
        
    Example:
        >>> nearest_higher_divisible(123, 10)  # Returns 130
    """
    if num % divisor == 0:
        return num
    else:
        remainder = num % divisor
        return int(num + divisor - remainder)
    
def nearest_bounding_box(minr, minc, maxr, maxc, divisor):
    """
    Adjust bounding box coordinates to be divisible by a given divisor.
    
    This function adjusts the bounding box coordinates so that the resulting
    dimensions are divisible by the specified divisor. Min coordinates are
    rounded up, max coordinates are rounded down.
    
    Args:
        minr (int): Minimum row coordinate
        minc (int): Minimum column coordinate  
        maxr (int): Maximum row coordinate
        maxc (int): Maximum column coordinate
        divisor (int/float): Divisor for coordinate adjustment
        
    Returns:
        tuple: Adjusted (minr, minc, maxr, maxc) coordinates
        
    Example:
        >>> bbox = nearest_bounding_box(15, 23, 127, 189, 10)
        >>> # Returns (20, 30, 120, 180)
    """
    minr = nearest_higher_divisible(minr, divisor)
    minc = nearest_higher_divisible(minc, divisor)
    maxr = nearest_lower_divisible(maxr, divisor)
    maxc = nearest_lower_divisible(maxc, divisor)

    return minr, minc, maxr, maxc

def preprocess(onlypores, mask, ut_rf, xct_resolution=0.025, ut_resolution=1):
    """
    Preprocess XCT and UT images by cropping to remove background and align coordinate systems.
    
    This function:
    1. Finds the bounding box of the sample in XCT data using the mask
    2. Crops XCT images (onlypores and mask) to this bounding box
    3. Converts the bounding box to UT coordinate system and crops UT data accordingly
    
    Args:
        onlypores (np.ndarray): 3D XCT image showing only pores (z, x, y)
        mask (np.ndarray): 3D XCT mask defining the sample region (z, x, y)
        ut_rf (np.ndarray): 3D UT radiofrequency data (z, x, y)
        xct_resolution (float, optional): XCT pixel resolution in mm. Defaults to 0.025.
        ut_resolution (float, optional): UT pixel resolution in mm. Defaults to 1.
        
    Returns:
        tuple: (onlypores_cropped, mask_cropped, ut_rf_cropped) - cropped versions of inputs
        
    Example:
        >>> onlypores_crop, mask_crop, ut_crop = preprocess(onlypores, mask, ut_rf)
    """
    # Calculate scaling factor between XCT and UT coordinate systems
    scaling_factor = xct_resolution / ut_resolution

    # XCT processing: find sample bounding box using maximum projection of mask 
    max_proj = np.max(mask, axis=0)

    # Label connected components and get region properties
    labels = measure.label(max_proj)
    props = regionprops(labels)

    # Get bounding box of the first (and typically only) labeled region
    minr_xct, minc_xct, maxr_xct, maxc_xct = props[0].bbox

    # Adjust bounding box to ensure proper scaling alignment
    minr_xct, minc_xct, maxr_xct, maxc_xct = nearest_bounding_box(
        minr_xct, minc_xct, maxr_xct, maxc_xct, 1/scaling_factor)

    # Crop the XCT volumes using the adjusted bounding box
    mask_cropped = mask[:, minr_xct:maxr_xct, minc_xct:maxc_xct]
    onlypores_cropped = onlypores[:, minr_xct:maxr_xct, minc_xct:maxc_xct]

    # UT processing: convert bounding box coordinates to UT resolution
    # Convert bounding box to UT resolution
    minr_ut = int(minr_xct * scaling_factor)
    minc_ut = int(minc_xct * scaling_factor)
    maxr_ut = int(maxr_xct * scaling_factor)
    maxc_ut = int(maxc_xct * scaling_factor)

    # Crop the UT volume using the converted coordinates
    ut_rf_cropped = ut_rf[:, minr_ut:maxr_ut, minc_ut:maxc_ut]

    return onlypores_cropped, mask_cropped, ut_rf_cropped

def patch(onlypores_cropped, mask_cropped, ut_rf_cropped, ut_patch_size=3, ut_step_size=1, 
          xct_resolution=0.025, ut_resolution=1):
    """
    Divide cropped images into patches for analysis.
    
    This function:
    1. Converts UT patch dimensions to equivalent XCT dimensions
    2. Crops all volumes to ensure they are divisible by patch sizes
    3. Divides volumes into overlapping patches
    4. Validates that UT and XCT patches are aligned
    
    Args:
        onlypores_cropped (np.ndarray): Cropped XCT pore image (z, x, y)
        mask_cropped (np.ndarray): Cropped XCT mask (z, x, y)  
        ut_rf_cropped (np.ndarray): Cropped UT RF data (z, x, y)
        ut_patch_size (int, optional): UT patch size in pixels. Defaults to 3.
        ut_step_size (int, optional): UT step size in pixels. Defaults to 1.
        xct_resolution (float, optional): XCT resolution in mm/pixel. Defaults to 0.025.
        ut_resolution (float, optional): UT resolution in mm/pixel. Defaults to 1.
        
    Returns:
        tuple: (patches_onlypores, patches_mask, patches_ut, original_shape)
               Returns 0,0,0,0 if patch alignment fails
               
    Example:
        >>> patches = patch(onlypores_crop, mask_crop, ut_crop, ut_patch_size=5)
    """
    # Compute equivalent XCT patch dimensions based on resolution scaling
    xct_patch_size = int(np.round(calculate_pixels(ut_resolution, xct_resolution, ut_patch_size)))
    xct_step_size = int(np.round(calculate_pixels(ut_resolution, xct_resolution, ut_step_size)))

    # Crop volumes to ensure dimensions are divisible by patch sizes
    ut_rf_cropped = crop_image_to_patch_size(ut_rf_cropped, ut_patch_size)
    onlypores_cropped = crop_image_to_patch_size(onlypores_cropped, xct_patch_size)
    mask_cropped = crop_image_to_patch_size(mask_cropped, xct_patch_size)

    # Ensure UT and XCT patches will have the same number of patches
    ut_shape = calculate_patch_shape(ut_rf_cropped.shape, ut_patch_size, ut_step_size)
    xct_shape = calculate_patch_shape(onlypores_cropped.shape, xct_patch_size, xct_step_size)

    if not (ut_shape[0] == xct_shape[0]):
        print('Patches are not the same')
        return 0, 0, 0, 0
    
    # Divide volumes into patches
    patches_ut = divide_into_patches(ut_rf_cropped, ut_patch_size, ut_step_size)

    patches_onlypores = divide_into_patches(onlypores_cropped, xct_patch_size, xct_step_size)
    # Note: Commented code below was for centering patches - kept for reference
    # center_size = int(patches_onlypores.shape[2] / ut_patch_size)
    # patches_onlypores = patches_onlypores[:, :, center_size:-center_size, center_size:-center_size]
    
    patches_mask = divide_into_patches(mask_cropped, xct_patch_size, xct_step_size)
    # patches_mask = patches_mask[:, :, center_size:-center_size, center_size:-center_size]

    return patches_onlypores, patches_mask, patches_ut, ut_rf_cropped.shape

def clean_material(sum_mask, volfrac, areafrac):
    """
    Filter out patches with insufficient material content.
    
    This function identifies patches that contain less than 80% material and marks
    their volume and area fractions as invalid (-1). This helps exclude patches
    that are mostly background or edge regions.
    
    Args:
        sum_mask (np.ndarray): Sum of mask values for each patch
        volfrac (np.ndarray): Volume fraction values for each patch
        areafrac (np.ndarray): Area fraction values for each patch
        
    Returns:
        tuple: (cleaned_volfrac, cleaned_areafrac) with invalid patches marked as -1
        
    Example:
        >>> volfrac_clean, areafrac_clean = clean_material(sum_mask, volfrac, areafrac)
    """
    # Get the value of a patch that is full material (maximum mask sum)
    full_material = np.max(sum_mask)
    
    # Calculate the percentage of material in each patch
    mat_percentage = sum_mask / full_material
    
    # Mark patches with less than 80% material as invalid
    indexes = np.where(mat_percentage < 0.8)[0]
    volfrac[indexes] = -1
    areafrac[indexes] = -1

    return volfrac, areafrac

def layer_cleaning(mask_cropped, onlypores_cropped):
    """
    Remove pores from layers that cannot be detected by ultrasonic testing.
    
    This function identifies layer boundaries in the sample and removes pores from
    the last few layers, as UT typically cannot detect pores very close to the
    back wall due to physical limitations of the technique.
    
    Args:
        mask_cropped (np.ndarray): Cropped mask defining the sample region (z, x, y)
        onlypores_cropped (np.ndarray): Cropped XCT pore image (z, x, y)
        
    Returns:
        np.ndarray: Cleaned pore image with back layers removed
        
    Example:
        >>> cleaned_pores = layer_cleaning(mask_cropped, onlypores_cropped)
    """
    # Define layer thickness in mm and convert to pixels
    layer_thickness = 0.508  # mm
    layer_thickness = int(np.round(layer_thickness / 0.025))  # Convert to pixels

    # Find front and back wall positions from mask
    indices = np.where(mask_cropped == 1)[0]
    frontwall = indices[0]
    backwall = indices[-1]

    # Calculate layer edge positions
    edges = [frontwall]

    while True:
        edge = edges[-1]
        if edge + layer_thickness > backwall:
            break
        edges.append(edge + layer_thickness)

    # Remove pores from the last two layers (UT cannot detect them)
    onlypores_cleaned = np.copy(onlypores_cropped)
    onlypores_cleaned[edges[-3]:] = 0
    
    return onlypores_cleaned

     

def create_dataset(patches_onlypores, patches_mask, patches_ut):
    """
    Create datasets from patches for machine learning training.
    
    This function processes patches to compute volumetric and area fractions of pores,
    combines them with UT data, and creates a DataFrame dataset for machine learning models.
    
    The function computes:
    - Volume fraction: ratio of pore volume to total material volume in each patch
    - Area fraction: ratio of pore area to total material area in maximum projection
    
    Args:
        patches_onlypores (np.ndarray): XCT pore patches (n_patches, z, x, y)
        patches_mask (np.ndarray): XCT mask patches (n_patches, z, x, y)
        patches_ut (np.ndarray): UT RF patches (n_patches, z, x, y)
        
    Returns:
        pd.DataFrame: Generated DataFrame with UT features and pore fraction targets
        
    Example:
        >>> df = create_dataset(pore_patches, mask_patches, ut_patches)
    """
    # Compute volume sums for each patch (sum over z-dimension)
    sum_onlypores_patches = np.sum(patches_onlypores, axis=1)
    sum_mask_patches = np.sum(patches_mask, axis=1)

    # Create maximum projections for area calculations
    proj_onlypores = np.max(patches_onlypores, axis=1)
    proj_mask = np.max(patches_mask, axis=1)

    # Calculate volume fractions (3D analysis)
    sum_onlypores = np.sum(sum_onlypores_patches, axis=(1, 2))
    sum_mask = np.sum(sum_mask_patches, axis=(1, 2))

    # Handle division by zero and mark non-material regions
    zero_indices = np.where(sum_mask == 0)
    volfrac = sum_onlypores / (sum_mask + 1e-6)
    volfrac[zero_indices] = -1

    # Calculate area fractions (2D projections)
    sum_onlypores_area = np.sum(proj_onlypores, axis=(1, 2)).astype(np.int16)
    sum_mask_area = np.sum(proj_mask, axis=(1, 2)).astype(np.int16)

    areafrac = sum_onlypores_area / (sum_mask_area + 1e-6)
    zero_indices = np.where(sum_mask_area == 0)
    areafrac[zero_indices] = -1

    # Clean fractions by removing patches with insufficient material
    volfrac, areafrac = clean_material(sum_mask, volfrac, areafrac)

    # Prepare UT data for DataFrame (reshape from 3D patches to 1D feature vectors)
    ut_patches_reshaped = patches_ut.transpose(0, 2, 3, 1)
    ut_patches_reshaped = ut_patches_reshaped.reshape(ut_patches_reshaped.shape[0], -1)

    # Create column names for UT features
    columns_ut = []
    for i in range(ut_patches_reshaped.shape[1]):
        columns_ut.append(f'ut_rf_{i}')
    columns_ut = np.array(columns_ut)

    # Combine UT features with target variables (volfrac and areafrac)
    patch_vs_volfrac = np.hstack((ut_patches_reshaped, volfrac.reshape(-1, 1), areafrac.reshape(-1, 1)))

    # Create DataFrame with appropriate column names
    df_patch_vs_volfrac = pd.DataFrame(patch_vs_volfrac, 
                                     columns=np.append(columns_ut, ['volfrac', 'areafrac']))

    return df_patch_vs_volfrac

def main(onlypores, mask, ut_rf, xct_resolution=0.025, ut_resolution=1, 
         ut_patch_size=3, ut_step_size=1):
    """
    Main pipeline for creating UT vs XCT datasets.
    
    This function orchestrates the complete workflow for creating machine learning
    datasets from UT and XCT data:
    1. Preprocessing to align coordinate systems and crop regions of interest
    3. Patching both UT and XCT data into analysis windows
    4. Optional pore cleaning (currently commented out)
    5. Dataset creation with volume and area fraction targets
    
    Args:
        onlypores (np.ndarray): 3D XCT image showing only pores (z, x, y)
        mask (np.ndarray): 3D XCT mask defining sample boundaries (z, x, y)
        ut_rf (np.ndarray): 3D UT radiofrequency data (z, x, y)
        xct_resolution (float, optional): XCT pixel resolution in mm. Defaults to 0.025.
        ut_resolution (float, optional): UT pixel resolution in mm. Defaults to 1.
        ut_patch_size (int, optional): UT patch size in pixels. Defaults to 3.
        ut_step_size (int, optional): UT step size for patch overlap. Defaults to 1.
        
    Returns:
        tuple: (original_shape, num_samples, dataframe)
               - original_shape: Shape of the original cropped UT data
               - num_samples: Number of patches/samples in the dataset
               - dataframe: The generated DataFrame with UT features and targets
               
    Example:
        >>> shape, n_samples, df = main(onlypores, mask, ut_rf, output_folder)
        >>> print(f"Created {n_samples} samples")
    """
    
    print('Preprocessing and patching the images...')
    # Step 1: Preprocess images to align coordinate systems and crop ROI
    onlypores_cropped, mask_cropped, ut_rf_cropped = preprocess(
        onlypores, mask, ut_rf, xct_resolution, ut_resolution)    
    
    print('Patching the images...')
    # Step 3: Divide images into patches for analysis
    patches_onlypores, patches_mask, patches_ut, shape = patch(
        onlypores_cropped, mask_cropped, ut_rf_cropped, 
        ut_patch_size, ut_step_size, xct_resolution, ut_resolution)
    
    print('Cleaning the pores...')
    # Step 4: Optional pore cleaning (currently commented out)
    # patches_onlypores = clean_pores_3D(patches_onlypores)
    
    print('Creating the datasets...')
    # Step 5: Create final dataset with features and targets
    df = create_dataset(patches_onlypores, patches_mask, patches_ut)

    return shape, df