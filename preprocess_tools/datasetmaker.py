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
from scipy.signal import fftconvolve
from joblib import Parallel, delayed


VOLFRAC_REGION_NAMES = ('front', 'middle', 'back')


def make_disk_kernel(radius_px):
    """
    Create a 2D binary disk kernel for FFT convolution.

    Args:
        radius_px (float): Disk radius in pixels. Rounded to nearest integer.
            Must round to at least 1.

    Returns:
        np.ndarray: Float64 2D array with 1s inside the disk and 0s outside.
            Shape is (2*r+1, 2*r+1) where r = int(round(radius_px)).

    Raises:
        ValueError: If radius rounds to less than 1.
    """
    r = int(round(radius_px))
    if r < 1:
        raise ValueError(f"radius_px must round to at least 1, got {radius_px}")
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x ** 2 + y ** 2 <= r ** 2).astype(np.float64)


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
          xct_resolution=0.025, ut_resolution=1, xct_patch_reduced_size=None):
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
        xct_patch_reduced_size (int, optional): If provided, center-crops each XCT pore patch
            to this size (in pixels) after patch extraction. Useful to reduce target area
            while keeping UT patches unchanged. Defaults to None (no reduction).
        
    Returns:
        tuple: (patches_onlypores, patches_mask, patches_ut, patch_grid_shape)
               Returns 0,0,0,0 if patch alignment fails
               - patches_onlypores: XCT pore patches array
                 (this is the only output reduced when xct_patch_reduced_size is provided)
               - patches_mask: XCT mask patches array  
               - patches_ut: UT RF patches array
               - patch_grid_shape: (num_patches_h, num_patches_w) for volfrac reconstruction
               
    Example:
        >>> patches = patch(onlypores_crop, mask_crop, ut_crop, ut_patch_size=5)
        >>> pore_patches, mask_patches, ut_patches, grid_shape = patches
    """

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

    if xct_patch_reduced_size:
        reduced_size = xct_patch_reduced_size
        start = (xct_patch_size - reduced_size) // 2
        end = start + reduced_size
        patches_onlypores = patches_onlypores[:, :, start:end, start:end]
    
    patches_mask = divide_into_patches(mask_cropped, xct_patch_size, xct_step_size)

    # Calculate patch grid shape for reconstructing volfrac image from flat dataset
    num_patches_h = ((ut_rf_cropped.shape[1] - ut_patch_size) // ut_step_size) + 1
    num_patches_w = ((ut_rf_cropped.shape[2] - ut_patch_size) // ut_step_size) + 1
    patch_grid_shape = (num_patches_h, num_patches_w)

    return patches_onlypores, patches_mask, patches_ut, patch_grid_shape

def clean_material(sum_mask, volfrac, areafrac, material_threshold=0.8):
    """
    Filter out patches with insufficient material content.

    This function identifies patches that contain less than the specified material threshold and marks
    their volume and area fractions as invalid (-1). This helps exclude patches
    that are mostly background or edge regions.
    
    Args:
        sum_mask (np.ndarray): Sum of mask values for each patch
        volfrac (np.ndarray): Volume fraction values for each patch
        areafrac (np.ndarray): Area fraction values for each patch
        material_threshold (float, optional): Threshold for valid material content. Defaults to 0.8.
        
    Returns:
        tuple: (cleaned_volfrac, cleaned_areafrac) with invalid patches marked as -1
        
    Example:
        >>> volfrac_clean, areafrac_clean = clean_material(sum_mask, volfrac, areafrac)
    """
    # Get the value of a patch that is full material (maximum mask sum)
    full_material = np.max(sum_mask)
    
    # Calculate the percentage of material in each patch
    mat_percentage = sum_mask / full_material

    # Mark patches with less than the specified material threshold as invalid
    indexes = np.where(mat_percentage < material_threshold)[0]
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

     

def calculate_patch_fractions(patches_onlypores, patches_mask):
    """
    Calculate pore volume and area fractions for each XCT patch.

    This function computes the same target variables used by `create_dataset`,
    but returns them as arrays instead of attaching them to a flattened UT
    dataframe.

    Args:
        patches_onlypores (np.ndarray): XCT pore patches (n_patches, z, x, y)
        patches_mask (np.ndarray): XCT mask patches (n_patches, z, x, y)

    Returns:
        tuple: (volfrac, areafrac), each with shape (n_patches,)

    Example:
        >>> volfrac, areafrac = calculate_patch_fractions(pore_patches, mask_patches)
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

    # Handle division by zero and mark non-material regions
    areafrac = sum_onlypores_area / (sum_mask_area + 1e-6)
    zero_indices = np.where(sum_mask_area == 0)
    areafrac[zero_indices] = -1

    # Clean fractions by removing patches with insufficient material
    volfrac, areafrac = clean_material(sum_mask, volfrac, areafrac)

    return volfrac, areafrac


def calculate_conical_fraction_maps(
    onlypores_cropped,
    mask_cropped,
    xct_pixels_per_ut_pixel,
    radius_front_mm,
    radius_back_mm,
    xct_resolution=0.025,
    material_threshold=0.8,
):
    """
    Compute depth-region VVF maps using a depth-varying circular beam footprint.

    Instead of summing pores in a fixed block per UT pixel, this function
    applies a 2D FFT convolution with a disk kernel whose radius matches the
    physical UT beam footprint at each depth region (front, middle, back).
    The beam radius varies linearly from `radius_front_mm` at the frontwall to
    `radius_back_mm` at the backwall; the average radius for each region is used.

    Args:
        onlypores_cropped (np.ndarray): XCT pore image aligned to the UT grid (z, x, y).
        mask_cropped (np.ndarray): XCT material mask aligned to the UT grid (z, x, y).
        xct_pixels_per_ut_pixel (int): XCT pixels per UT pixel in x-y (e.g. 40).
        radius_front_mm (float): Beam radius at the frontwall in mm.
        radius_back_mm (float): Beam radius at the backwall in mm.
        xct_resolution (float): XCT pixel size in mm/pixel. Defaults to 0.025.
        material_threshold (float): Minimum fraction of max mask content for a
            valid UT pixel. Defaults to 0.8.

    Returns:
        tuple: (volfrac_maps, areafrac_image, depth_rois) with the same shapes
            and semantics as `calculate_fraction_maps`.
            - volfrac_maps: shape (3, ut_x, ut_y), dtype float32
            - areafrac_image: shape (ut_x, ut_y), dtype float32
            - depth_rois: list of front/middle/back ROI dicts

    Raises:
        ValueError: If XCT spatial dimensions are not divisible by
            xct_pixels_per_ut_pixel, or if the mask contains no material.
    """
    block_size = xct_pixels_per_ut_pixel
    z, x, y = onlypores_cropped.shape

    if x % block_size != 0 or y % block_size != 0:
        raise ValueError(
            f"XCT spatial shape {(x, y)} must be divisible by {block_size}"
        )

    ut_x = x // block_size
    ut_y = y // block_size

    depth_rois = calculate_depth_rois(mask_cropped)

    # Infer specimen depth from mask to parameterise the linear beam model
    material_by_z = np.sum(mask_cropped, axis=(1, 2))
    material_indices = np.flatnonzero(material_by_z > 0)
    if material_indices.size == 0:
        raise ValueError("Cannot compute conical fraction maps from an empty mask")
    specimen_depth_mm = (material_indices[-1] - material_indices[0] + 1) * xct_resolution

    # UT pixel center coordinates in XCT pixel space
    half = block_size // 2
    cx = np.arange(ut_x) * block_size + half
    cy = np.arange(ut_y) * block_size + half

    volfrac_maps = np.full((len(VOLFRAC_REGION_NAMES), ut_x, ut_y), -1, dtype=np.float32)
    mask_sums_at_centers = np.zeros((len(VOLFRAC_REGION_NAMES), ut_x, ut_y), dtype=np.float64)

    for region_index, depth_roi in enumerate(depth_rois):
        start_z = depth_roi['start_z']
        end_z = depth_roi['end_z']

        # Average beam radius for this depth region (linear interpolation)
        mid_px_from_fw = (
            depth_roi['start_from_frontwall'] + depth_roi['end_from_frontwall']
        ) / 2.0
        t = np.clip(mid_px_from_fw * xct_resolution / specimen_depth_mm, 0.0, 1.0)
        radius_mm = radius_front_mm + (radius_back_mm - radius_front_mm) * t
        radius_px = int(round(radius_mm / xct_resolution))

        kernel = make_disk_kernel(radius_px)

        pore_sum_2d = np.sum(onlypores_cropped[start_z:end_z], axis=0).astype(np.float64)
        mask_sum_2d = np.sum(mask_cropped[start_z:end_z], axis=0).astype(np.float64)

        pore_conv = fftconvolve(pore_sum_2d, kernel, mode='same')
        mask_conv = fftconvolve(mask_sum_2d, kernel, mode='same')

        pore_at_centers = pore_conv[np.ix_(cx, cy)]
        mask_at_centers = mask_conv[np.ix_(cx, cy)]

        mask_sums_at_centers[region_index] = mask_at_centers

        valid = mask_at_centers > 0
        volfrac_maps[region_index, valid] = (
            pore_at_centers[valid] / mask_at_centers[valid]
        ).astype(np.float32)

    # Apply material threshold per region
    for region_index in range(len(VOLFRAC_REGION_NAMES)):
        full_material = np.max(mask_sums_at_centers[region_index])
        if full_material > 0:
            material_pct = mask_sums_at_centers[region_index] / full_material
            volfrac_maps[region_index, material_pct < material_threshold] = -1

    # Areafrac: max projection over full depth, convolved with midpoint disk
    radius_mid_px = int(round(
        ((radius_front_mm + radius_back_mm) / 2.0) / xct_resolution
    ))
    kernel_area = make_disk_kernel(radius_mid_px)

    proj_pore = np.max(onlypores_cropped, axis=0).astype(np.float64)
    proj_mask = np.max(mask_cropped, axis=0).astype(np.float64)

    proj_pore_conv = fftconvolve(proj_pore, kernel_area, mode='same')
    proj_mask_conv = fftconvolve(proj_mask, kernel_area, mode='same')

    pore_area_at_centers = proj_pore_conv[np.ix_(cx, cy)]
    mask_area_at_centers = proj_mask_conv[np.ix_(cx, cy)]

    areafrac_image = np.full((ut_x, ut_y), -1, dtype=np.float32)
    valid_area = mask_area_at_centers > 0
    areafrac_image[valid_area] = (
        pore_area_at_centers[valid_area] / mask_area_at_centers[valid_area]
    ).astype(np.float32)

    # Material threshold for areafrac using full-depth mask column sums
    sum_mask_full = np.sum(mask_cropped, axis=0).astype(np.float64)
    sum_mask_at_centers = sum_mask_full[np.ix_(cx, cy)]
    full_material_area = np.max(sum_mask_at_centers)
    if full_material_area > 0:
        material_pct_area = sum_mask_at_centers / full_material_area
        areafrac_image[material_pct_area < material_threshold] = -1

    return volfrac_maps, areafrac_image, depth_rois


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
    volfrac, areafrac = calculate_patch_fractions(patches_onlypores, patches_mask)

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


def crop_xy_center(image, target_shape):
    """
    Center-crop the x-y dimensions of a 3D image to a target shape.

    Args:
        image (np.ndarray): 3D image with shape (z, x, y)
        target_shape (tuple): Target spatial shape (x, y)

    Returns:
        np.ndarray: Center-cropped image with shape (z, target_x, target_y)
    """
    target_x, target_y = target_shape
    _, x, y = image.shape

    if target_x > x or target_y > y:
        raise ValueError(
            f"Target shape {target_shape} is larger than image spatial shape {(x, y)}"
        )

    start_x = (x - target_x) // 2
    start_y = (y - target_y) // 2

    return image[:, start_x:start_x + target_x, start_y:start_y + target_y]


def align_to_ut_grid(onlypores_cropped, mask_cropped, ut_rf_cropped, xct_pixels_per_ut_pixel):
    """
    Crop aligned XCT and UT volumes to their largest common UT-resolution grid.

    Args:
        onlypores_cropped (np.ndarray): Cropped XCT pore image (z, x, y)
        mask_cropped (np.ndarray): Cropped XCT mask (z, x, y)
        ut_rf_cropped (np.ndarray): Cropped UT RF data (z, x, y)
        xct_pixels_per_ut_pixel (int): Number of XCT pixels per UT pixel in x-y

    Returns:
        tuple: (onlypores_grid, mask_grid, ut_rf_grid)
    """
    max_ut_x = min(
        ut_rf_cropped.shape[1],
        onlypores_cropped.shape[1] // xct_pixels_per_ut_pixel,
        mask_cropped.shape[1] // xct_pixels_per_ut_pixel,
    )
    max_ut_y = min(
        ut_rf_cropped.shape[2],
        onlypores_cropped.shape[2] // xct_pixels_per_ut_pixel,
        mask_cropped.shape[2] // xct_pixels_per_ut_pixel,
    )

    if max_ut_x <= 0 or max_ut_y <= 0:
        raise ValueError(
            "No overlapping UT/XCT grid could be created with "
            f"{xct_pixels_per_ut_pixel} XCT pixels per UT pixel"
        )

    ut_rf_grid = crop_xy_center(ut_rf_cropped, (max_ut_x, max_ut_y))
    xct_grid_shape = (
        max_ut_x * xct_pixels_per_ut_pixel,
        max_ut_y * xct_pixels_per_ut_pixel,
    )
    onlypores_grid = crop_xy_center(onlypores_cropped, xct_grid_shape)
    mask_grid = crop_xy_center(mask_cropped, xct_grid_shape)

    return onlypores_grid, mask_grid, ut_rf_grid


def calculate_depth_rois(mask_cropped):
    """
    Calculate front, middle, and back depth ROIs from the material mask.

    The first and last z slices containing material define the sample depth. The
    depth is split into front, middle, and back regions. ROI starts and ends are
    stored both as absolute z indices and relative to the material frontwall.

    Args:
        mask_cropped (np.ndarray): XCT material mask aligned to the UT grid (z, x, y)

    Returns:
        list: One dictionary per depth region. `end_z` and
            `end_from_frontwall` are kept as exclusive endpoints. Explicit
            inclusive and exclusive endpoint keys are also included.
    """
    material_by_z = np.sum(mask_cropped, axis=(1, 2))
    material_indices = np.flatnonzero(material_by_z > 0)

    if material_indices.size == 0:
        raise ValueError("Cannot calculate depth ROIs from an empty material mask")

    material_frontwall_z = int(material_indices[0])
    material_backwall_z = int(material_indices[-1] + 1)

    depth_regions = np.array_split(
        np.arange(material_frontwall_z, material_backwall_z),
        len(VOLFRAC_REGION_NAMES))

    depth_rois = []
    for region_name, z_indices in zip(VOLFRAC_REGION_NAMES, depth_regions):
        if z_indices.size == 0:
            raise ValueError(
                "Material mask depth is too small to split into "
                f"{len(VOLFRAC_REGION_NAMES)} regions"
            )

        start_z = int(z_indices[0])
        end_z = int(z_indices[-1] + 1)
        depth_rois.append({
            'region': region_name,
            'start_z': start_z,
            'end_z': end_z,
            'start_from_frontwall': start_z - material_frontwall_z,
            'end_from_frontwall': end_z - material_frontwall_z,
            'end_exclusive_z': end_z,
            'end_inclusive_z': end_z - 1,
            'end_exclusive_from_frontwall': end_z - material_frontwall_z,
            'end_inclusive_from_frontwall': end_z - 1 - material_frontwall_z,
            'material_frontwall_z': material_frontwall_z,
            'material_backwall_z': material_backwall_z,
        })

    return depth_rois


def calculate_depth_volfrac_maps(pore_blocks, mask_blocks, depth_rois,
                                 material_threshold=0.8):
    """
    Calculate front, middle, and back volume fraction maps from XCT blocks.

    Args:
        pore_blocks (np.ndarray): Pore blocks with shape
            (z, ut_x, block_size, ut_y, block_size)
        mask_blocks (np.ndarray): Material mask blocks with shape
            (z, ut_x, block_size, ut_y, block_size)
        depth_rois (list): Depth ROI dictionaries from `calculate_depth_rois`.
        material_threshold (float, optional): Minimum material content for valid
            output pixels. Defaults to 0.8.

    Returns:
        np.ndarray: Volume fraction maps with shape (3, ut_x, ut_y), ordered as
            front, middle, and back.
    """
    _, ut_x, _, ut_y, _ = pore_blocks.shape
    volfrac_maps = np.full(
        (len(VOLFRAC_REGION_NAMES), ut_x, ut_y), -1, dtype=np.float32)
    mask_sums = np.zeros_like(volfrac_maps, dtype=np.float64)

    for region_index, depth_roi in enumerate(depth_rois):
        start_z = depth_roi['start_z']
        end_z = depth_roi['end_z']
        pore_sum = np.sum(pore_blocks[start_z:end_z], axis=(0, 2, 4))
        mask_sum = np.sum(mask_blocks[start_z:end_z], axis=(0, 2, 4))
        mask_sums[region_index] = mask_sum

        valid = mask_sum > 0
        volfrac_maps[region_index, valid] = (
            pore_sum[valid] / mask_sum[valid]
        )

    if material_threshold is not None:
        for region_index in range(len(VOLFRAC_REGION_NAMES)):
            full_material = np.max(mask_sums[region_index])
            if full_material > 0:
                material_percentage = mask_sums[region_index] / full_material
                volfrac_maps[region_index, material_percentage < material_threshold] = -1

    return volfrac_maps


def calculate_fraction_maps(onlypores_cropped, mask_cropped, xct_pixels_per_ut_pixel,
                            material_threshold=0.8):
    """
    Rescale XCT pores to UT resolution by calculating fractions per UT pixel.

    Volume fraction is calculated separately for front, middle, and back depth
    regions. The three regions are created from the sample depth defined by the
    material mask.

    Args:
        onlypores_cropped (np.ndarray): XCT pore image aligned to the UT grid (z, x, y)
        mask_cropped (np.ndarray): XCT material mask aligned to the UT grid (z, x, y)
        xct_pixels_per_ut_pixel (int): Number of XCT pixels per UT pixel in x-y
        material_threshold (float, optional): Minimum material content for a valid
            output pixel. Defaults to 0.8.

    Returns:
        tuple: (volfrac_maps, areafrac_image, depth_rois)
               - volfrac_maps has shape (3, ut_x, ut_y), ordered as
                 front, middle, and back.
               - areafrac_image has shape (ut_x, ut_y)
               - depth_rois stores the depth ROI bounds
    """
    block_size = xct_pixels_per_ut_pixel
    z, x, y = onlypores_cropped.shape

    if x % block_size != 0 or y % block_size != 0:
        raise ValueError(
            f"XCT spatial shape {(x, y)} must be divisible by {block_size}"
        )

    ut_x = x // block_size
    ut_y = y // block_size

    pore_blocks = onlypores_cropped.reshape(z, ut_x, block_size, ut_y, block_size)
    mask_blocks = mask_cropped.reshape(z, ut_x, block_size, ut_y, block_size)

    depth_rois = calculate_depth_rois(mask_cropped)
    volfrac_maps = calculate_depth_volfrac_maps(
        pore_blocks, mask_blocks, depth_rois,
        material_threshold=material_threshold)

    sum_mask = np.sum(mask_blocks, axis=(0, 2, 4))

    proj_onlypores = np.max(pore_blocks, axis=0)
    proj_mask = np.max(mask_blocks, axis=0)

    sum_onlypores_area = np.sum(proj_onlypores, axis=(1, 3))
    sum_mask_area = np.sum(proj_mask, axis=(1, 3))

    areafrac_image = np.full(sum_mask_area.shape, -1, dtype=np.float32)
    valid_area = sum_mask_area > 0
    areafrac_image[valid_area] = (
        sum_onlypores_area[valid_area] / sum_mask_area[valid_area]
    )

    full_material = np.max(sum_mask)
    if full_material > 0:
        material_percentage = sum_mask / full_material
        invalid_material = material_percentage < material_threshold
        areafrac_image[invalid_material] = -1

    return volfrac_maps, areafrac_image, depth_rois


def calculate_volume_vvf(onlypores, mask, xct_resolution=0.025,
                         material_threshold=0.8, metadata=False):
    """
    Compute per-pixel front/middle/back VVF maps and whole-volume VVF for a
    standalone XCT pore/mask pair (no registered UT pair required).

    Unlike `calculate_fraction_maps` as used elsewhere in the pipeline, this
    does not bin pixels into UT-sized blocks: onlypores/mask are kept at
    their native XCT x-y resolution, with no cropping or reduction. This is
    done by calling `calculate_fraction_maps` with a block size of 1 pixel,
    which reduces to the identity (every XCT pixel is its own output pixel)
    while still reusing its depth-region/material-threshold logic exactly.
    Whole-volume and per-region scalar VVF (voxel sums, no material
    threshold) are computed on the same (untouched) volumes.

    Args:
        onlypores (np.ndarray): 3D XCT pore image (z, x, y), boolean/binary.
        mask (np.ndarray): 3D XCT material mask (z, x, y), boolean/binary.
        xct_resolution (float, optional): XCT pixel resolution in mm, recorded in
            `vvf_row` metadata only (not used in the computation). Defaults to 0.025.
        material_threshold (float, optional): Minimum material content for a valid
            map pixel, passed to `calculate_fraction_maps`. Defaults to 0.8.
        metadata (bool, optional): If True, `vvf_row` also includes shape/ROI/voxel
            bookkeeping fields. Defaults to False (only the vvf_* scalars).

    Returns:
        tuple: (volfrac_maps, areafrac_image, vvf_row)
            - volfrac_maps: shape (3, x, y) float32 at native XCT resolution,
              ordered front/middle/back
            - areafrac_image: shape (x, y) float32 at native XCT resolution
            - vvf_row: dict with 'vvf_total', 'vvf_front', 'vvf_middle', 'vvf_back'
              (nan where the corresponding region/volume has no material), plus
              extended bookkeeping fields when metadata=True

    Raises:
        ValueError: If `onlypores` and `mask` shapes differ.

    Example:
        >>> volfrac_maps, areafrac_image, vvf_row = calculate_volume_vvf(pores, mask)
        >>> vvf_row['vvf_total']
    """
    if onlypores.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: onlypores {onlypores.shape} vs mask {mask.shape}"
        )

    volfrac_maps, areafrac_image, depth_rois = calculate_fraction_maps(
        onlypores, mask, xct_pixels_per_ut_pixel=1,
        material_threshold=material_threshold)

    total_pore_voxels = int(np.sum(onlypores))
    total_material_voxels = int(np.sum(mask))
    vvf_total = (
        total_pore_voxels / total_material_voxels
        if total_material_voxels > 0 else float('nan')
    )

    vvf_row = {'vvf_total': vvf_total}

    region_voxels = {}
    for roi in depth_rois:
        region = roi['region']
        start_z, end_z = roi['start_z'], roi['end_z']
        pore_voxels = int(np.sum(onlypores[start_z:end_z]))
        material_voxels = int(np.sum(mask[start_z:end_z]))
        vvf_row[f'vvf_{region}'] = (
            pore_voxels / material_voxels if material_voxels > 0 else float('nan')
        )
        region_voxels[region] = (pore_voxels, material_voxels)

    if metadata:
        vvf_row['volume_shape'] = tuple(onlypores.shape)
        vvf_row['volfrac_maps_shape'] = tuple(volfrac_maps.shape)
        vvf_row['target_map_shape'] = tuple(volfrac_maps.shape[1:])
        vvf_row['volfrac_region_names'] = tuple(roi['region'] for roi in depth_rois)
        vvf_row['depth_roi_start_z'] = tuple(roi['start_z'] for roi in depth_rois)
        vvf_row['depth_roi_end_exclusive_z'] = tuple(roi['end_exclusive_z'] for roi in depth_rois)
        vvf_row['depth_roi_end_inclusive_z'] = tuple(roi['end_inclusive_z'] for roi in depth_rois)
        vvf_row['depth_roi_start_from_frontwall'] = tuple(
            roi['start_from_frontwall'] for roi in depth_rois)
        vvf_row['depth_roi_end_exclusive_from_frontwall'] = tuple(
            roi['end_exclusive_from_frontwall'] for roi in depth_rois)
        vvf_row['depth_roi_end_inclusive_from_frontwall'] = tuple(
            roi['end_inclusive_from_frontwall'] for roi in depth_rois)
        vvf_row['material_frontwall_z'] = depth_rois[0]['material_frontwall_z']
        vvf_row['material_backwall_z'] = depth_rois[0]['material_backwall_z']
        vvf_row['pore_voxels_total'] = total_pore_voxels
        vvf_row['material_voxels_total'] = total_material_voxels
        for region, (pore_voxels, material_voxels) in region_voxels.items():
            vvf_row[f'pore_voxels_{region}'] = pore_voxels
            vvf_row[f'material_voxels_{region}'] = material_voxels
        vvf_row['material_threshold'] = material_threshold
        vvf_row['xct_resolution'] = xct_resolution

    return volfrac_maps, areafrac_image, vvf_row


def create_image_maps(onlypores_cropped, mask_cropped, ut_rf_cropped,
                      xct_resolution=0.025, ut_resolution=1.0,
                      material_threshold=0.8,
                      conical=False,
                      radius_front_mm=None,
                      radius_back_mm=None):
    """
    Create a plain UT volume and XCT fraction maps on the same x-y grid.

    Args:
        onlypores_cropped (np.ndarray): Cropped XCT pore image (z, x, y)
        mask_cropped (np.ndarray): Cropped XCT mask (z, x, y)
        ut_rf_cropped (np.ndarray): Cropped UT RF data (z, x, y)
        xct_resolution (float, optional): XCT pixel resolution in mm. Defaults to 0.025.
        ut_resolution (float, optional): UT pixel resolution in mm. Defaults to 1.0.
        material_threshold (float, optional): Minimum material content for a valid
            output pixel. Defaults to 0.8.
        conical (bool, optional): If True, use conical beam extraction via FFT
            convolution instead of fixed block reduction. Defaults to False.
        radius_front_mm (float, optional): Beam radius at the frontwall in mm.
            Required when conical=True.
        radius_back_mm (float, optional): Beam radius at the backwall in mm.
            Required when conical=True.

    Returns:
        tuple: (ut_rf_grid, volfrac_maps, areafrac_image, depth_rois)
               - volfrac_maps has shape (3, x, y), ordered as
                 front, middle, and back.
               - depth_rois stores depth ROI bounds relative to the material frontwall.

    Raises:
        ValueError: If conical=True and radius_front_mm or radius_back_mm are not provided.
    """
    xct_pixels = calculate_pixels(ut_resolution, xct_resolution, 1)
    xct_pixels_per_ut_pixel = int(np.round(xct_pixels))

    if xct_pixels_per_ut_pixel <= 0:
        raise ValueError("xct_pixels_per_ut_pixel must be greater than zero")

    if not np.isclose(xct_pixels, xct_pixels_per_ut_pixel):
        raise ValueError(
            "The ratio between UT and XCT resolution must be an integer to "
            f"create fraction maps without interpolation. Got {xct_pixels}."
        )

    onlypores_grid, mask_grid, ut_rf_grid = align_to_ut_grid(
        onlypores_cropped, mask_cropped, ut_rf_cropped, xct_pixels_per_ut_pixel)

    if conical:
        if radius_front_mm is None or radius_back_mm is None:
            raise ValueError(
                "radius_front_mm and radius_back_mm are required when conical=True"
            )
        volfrac_maps, areafrac_image, depth_rois = calculate_conical_fraction_maps(
            onlypores_grid, mask_grid, xct_pixels_per_ut_pixel,
            radius_front_mm=radius_front_mm,
            radius_back_mm=radius_back_mm,
            xct_resolution=xct_resolution,
            material_threshold=material_threshold,
        )
    else:
        volfrac_maps, areafrac_image, depth_rois = calculate_fraction_maps(
            onlypores_grid, mask_grid, xct_pixels_per_ut_pixel,
            material_threshold=material_threshold)

    return ut_rf_grid, volfrac_maps, areafrac_image, depth_rois


def main(onlypores, mask, ut_rf, xct_resolution=0.025, ut_resolution=1.0, 
         ut_patch_size=3, ut_step_size=1, xct_patch_reduced_size=None):
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
        xct_patch_reduced_size (int, optional): Optional reduced size for XCT pore patches.
            Passed to `patch` to center-crop XCT pore patches while keeping UT patches at
            their original size. Defaults to None.
        
    Returns:
        tuple: (patch_grid_shape, dataframe)
               - patch_grid_shape: (num_patches_h, num_patches_w) for reconstructing volfrac image
               - dataframe: The generated DataFrame with UT features and targets
               
    Example:
        >>> grid_shape, df = main(onlypores, mask, ut_rf)
        >>> volfrac_flat = df['volfrac'].values
        >>> volfrac_image = volfrac_flat.reshape(grid_shape)
    """
    
    print('Preprocessing and patching the images...')
    # Step 1: Preprocess images to align coordinate systems and crop ROI
    onlypores_cropped, mask_cropped, ut_rf_cropped = preprocess(
        onlypores, mask, ut_rf, xct_resolution, ut_resolution)    
    
    print('Patching the images...')
    # Step 3: Divide images into patches for analysis
    patches_onlypores, patches_mask, patches_ut, patch_grid_shape = patch(
        onlypores_cropped, mask_cropped, ut_rf_cropped, 
        ut_patch_size, ut_step_size, xct_resolution, ut_resolution,xct_patch_reduced_size)
    
    print('Cleaning the pores...')
    # Step 4: Optional pore cleaning (currently commented out)
    # patches_onlypores = clean_pores_3D(patches_onlypores)
    
    print('Creating the datasets...')
    # Step 5: Create final dataset with features and targets
    df = create_dataset(patches_onlypores, patches_mask, patches_ut)

    return patch_grid_shape, df


def main_images(onlypores, mask, ut_rf, xct_resolution=0.025, ut_resolution=1.0,
                material_threshold=0.8,
                conical=False,
                radius_front_mm=None,
                radius_back_mm=None):
    """
    Main pipeline for creating image-shaped UT vs XCT datasets without UT patches.

    This function mirrors `main` up to preprocessing, but it does not divide UT
    into patches. It returns the cropped UT volume and XCT pore fraction maps
    rescaled to the UT x-y grid.

    Args:
        onlypores (np.ndarray): 3D XCT image showing only pores (z, x, y)
        mask (np.ndarray): 3D XCT mask defining sample boundaries (z, x, y)
        ut_rf (np.ndarray): 3D UT radiofrequency data (z, x, y)
        xct_resolution (float, optional): XCT pixel resolution in mm. Defaults to 0.025.
        ut_resolution (float, optional): UT pixel resolution in mm. Defaults to 1.
        material_threshold (float, optional): Minimum material content for valid
            target pixels. Defaults to 0.8.
        conical (bool, optional): If True, use conical beam extraction via FFT
            convolution instead of fixed block reduction. Defaults to False.
        radius_front_mm (float, optional): Beam radius at the frontwall in mm.
            Required when conical=True.
        radius_back_mm (float, optional): Beam radius at the backwall in mm.
            Required when conical=True.

    Returns:
        tuple: (ut_rf_grid, volfrac_maps, areafrac_image, depth_rois)
               - ut_rf_grid: cropped UT volume with shape (z, x, y)
               - volfrac_maps: XCT volume fraction maps with shape (3, x, y),
                 ordered as front, middle, and back
               - areafrac_image: XCT area fraction map with shape (x, y)
               - depth_rois: list of front/middle/back ROI bounds relative to
                 the material frontwall

    Raises:
        ValueError: If conical=True and radius_front_mm or radius_back_mm are not provided.

    Example:
        >>> ut_image, volfrac_maps, areafrac_image, depth_rois = main_images(
        ...     onlypores, mask, ut_rf,
        ...     conical=True, radius_front_mm=1.615, radius_back_mm=1.40)
    """

    print('Preprocessing the images...')
    # Step 1: Preprocess images to align coordinate systems and crop ROI
    onlypores_cropped, mask_cropped, ut_rf_cropped = preprocess(
        onlypores, mask, ut_rf, xct_resolution, ut_resolution)

    print('Cleaning the pores...')
    # Step 2: Optional pore cleaning (currently commented out)
    # onlypores_cropped = layer_cleaning(mask_cropped, onlypores_cropped)

    print('Creating the image maps...')
    # Step 3: Reduce XCT to UT resolution (block or conical) and keep UT as plain volume
    ut_rf_grid, volfrac_maps, areafrac_image, depth_rois = create_image_maps(
        onlypores_cropped, mask_cropped, ut_rf_cropped,
        xct_resolution=xct_resolution,
        ut_resolution=ut_resolution,
        material_threshold=material_threshold,
        conical=conical,
        radius_front_mm=radius_front_mm,
        radius_back_mm=radius_back_mm,
    )

    return ut_rf_grid, volfrac_maps, areafrac_image, depth_rois


# ---------------------------------------------------------------------------
# Pore-level targets (material voxels, full-depth VVF, per-pore features)
# ---------------------------------------------------------------------------

PORE_FEATURE_COLUMNS = ('volume_mm3', 'eq_diameter_mm', 'area_mm2', 'sphericity',
                        'cenital_angle_deg', 'flatness', 'elongation')


def pore_shape(obj, voxel_mm=0.025):
    """
    Shape features of ONE pore.

    Args:
        obj (np.ndarray): (dz, dy, dx) bool crop containing a single pore
        voxel_mm (float, optional): Isotropic voxel size in mm. Defaults to 0.025.

    Returns:
        dict: volume_vox, volume_mm3, area_mm2 (marching-cubes surface),
            eq_diameter_mm, sphericity (clipped to 1), cenital_angle_deg (angle between
            the main inertia axis and z, folded to [0, 90]), flatness = sqrt(l3/l1) and
            elongation = sqrt(l2/l1) with l1 >= l2 >= l3 the covariance eigenvalues.
            PCA features are NaN for single-voxel pores.
    """
    n = int(obj.sum())
    vol = n * voxel_mm ** 3
    padded = np.pad(obj, 1).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=(voxel_mm,) * 3)
    area = measure.mesh_surface_area(verts, faces)
    out = {'volume_vox': n, 'volume_mm3': vol, 'area_mm2': area,
           'eq_diameter_mm': (6 * vol / np.pi) ** (1 / 3),
           'sphericity': min(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area, 1.0),
           'cenital_angle_deg': np.nan, 'flatness': np.nan, 'elongation': np.nan}
    if n < 2:
        return out
    coords = np.argwhere(obj).astype(np.float64) * voxel_mm
    w, v = np.linalg.eigh(np.cov(coords.T))
    if w[2] <= 0:
        return out
    out['cenital_angle_deg'] = np.degrees(np.arccos(np.clip(abs(v[0, 2]), 0, 1)))
    out['flatness'] = np.sqrt(max(w[0], 0) / w[2])
    out['elongation'] = np.sqrt(max(w[1], 0) / w[2])
    return out


def compute_pore_features(pores_blk, voxel_mm=0.025, min_voxels=8, min_axis=2):
    """
    Per-pore features of a block (3D connected components, 26-connectivity).

    A pore is removed if it has fewer than `min_voxels` voxels or a bounding-box
    extent smaller than `min_axis` voxels along any axis.

    Args:
        pores_blk (np.ndarray): (z, x, y) bool pore block
        voxel_mm (float, optional): Voxel size in mm. Defaults to 0.025.
        min_voxels (int, optional): Minimum voxel count. Defaults to 8.
        min_axis (int, optional): Minimum bounding-box extent per axis. Defaults to 2.

    Returns:
        tuple: (DataFrame with one row per kept pore, n_removed_pores, n_removed_voxels).
            Columns: pore_id, touches_xy_border, touches_z_border, centroid_z/x/y_vox
            (block-local voxel coordinates) and the `pore_shape` features.
    """
    lab, n = scipy.ndimage.label(pores_blk, structure=scipy.ndimage.generate_binary_structure(3, 3))
    Zb, Xb, Yb = pores_blk.shape
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    rows, n_rm, vox_rm = [], 0, 0
    for k, sl in enumerate(scipy.ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        if counts[k] < min_voxels or min(s.stop - s.start for s in sl) < min_axis:
            n_rm += 1
            vox_rm += int(counts[k])
            continue
        obj = lab[sl] == k
        centroid = np.argwhere(obj).mean(axis=0) + np.array([s.start for s in sl])
        row = {'pore_id': k,
               'touches_xy_border': sl[1].start == 0 or sl[1].stop == Xb or sl[2].start == 0 or sl[2].stop == Yb,
               'touches_z_border': sl[0].start == 0 or sl[0].stop == Zb,
               'centroid_z_vox': centroid[0], 'centroid_x_vox': centroid[1], 'centroid_y_vox': centroid[2]}
        row.update(pore_shape(obj, voxel_mm))
        rows.append(row)
    return pd.DataFrame(rows), n_rm, vox_rm


def calculate_material_voxels_map(mask_grid, xct_pixels_per_ut_pixel):
    """
    Material voxel count per UT pixel (full depth, no threshold masking).

    Args:
        mask_grid (np.ndarray): XCT material mask aligned to the UT grid (z, x, y)
        xct_pixels_per_ut_pixel (int): XCT pixels per UT pixel in x-y

    Returns:
        np.ndarray: (ut_x, ut_y) uint32 map
    """
    b = xct_pixels_per_ut_pixel
    z, x, y = mask_grid.shape
    return mask_grid.reshape(z, x // b, b, y // b, b).sum(axis=(0, 2, 4)).astype(np.uint32)


def calculate_thickness_map(mask_grid, xct_pixels_per_ut_pixel, xct_resolution=0.025):
    """
    Material thickness per UT pixel in mm (full depth, no threshold masking).

    Mean over the XCT columns inside each UT pixel of the number of material voxels
    along z, times the voxel size. Pixels without material are 0.

    Args:
        mask_grid (np.ndarray): XCT material mask aligned to the UT grid (z, x, y)
        xct_pixels_per_ut_pixel (int): XCT pixels per UT pixel in x-y
        xct_resolution (float, optional): Voxel size in mm. Defaults to 0.025.

    Returns:
        np.ndarray: (ut_x, ut_y) float32 map in mm
    """
    b = xct_pixels_per_ut_pixel
    voxels = calculate_material_voxels_map(mask_grid, b).astype(np.float64)
    return (voxels / (b * b) * xct_resolution).astype(np.float32)


def calculate_full_vvf_map(onlypores_grid, mask_grid, xct_pixels_per_ut_pixel, material_threshold=0.8):
    """
    Full-depth void volume fraction per UT pixel (-1 where invalid).

    Invalid = no material or material below `material_threshold` times the maximum
    material count, the same rule used by `calculate_depth_volfrac_maps`.

    Returns:
        np.ndarray: (ut_x, ut_y) float32 map
    """
    b = xct_pixels_per_ut_pixel
    z, x, y = mask_grid.shape
    pore_sum = onlypores_grid.reshape(z, x // b, b, y // b, b).sum(axis=(0, 2, 4))
    mask_sum = mask_grid.reshape(z, x // b, b, y // b, b).sum(axis=(0, 2, 4))
    vvf = np.full(mask_sum.shape, -1, dtype=np.float32)
    valid = mask_sum > 0
    vvf[valid] = pore_sum[valid] / mask_sum[valid]
    if material_threshold is not None and mask_sum.max() > 0:
        vvf[mask_sum / mask_sum.max() < material_threshold] = -1
    return vvf


def calculate_pore_patch_targets(onlypores_grid, mask_grid, xct_pixels_per_ut_pixel, patch_ut,
                                 xct_resolution=0.025, frontwall_z=0, material_threshold=0.8,
                                 min_voxels=8, min_axis=2):
    """
    Pore count and per-pore features on non-overlapping patches of `patch_ut` x `patch_ut` UT pixels.

    The patch grid is anchored at UT pixel (0, 0); trailing UT pixels that do not
    fill a whole patch get no pore targets. A patch is valid if its material voxel
    count is at least `material_threshold` times the maximum over patches. Pores are
    only computed for valid patches. Pores cut by a patch border appear in each patch.

    Args:
        onlypores_grid, mask_grid (np.ndarray): XCT pores / mask aligned to the UT grid (z, x, y)
        xct_pixels_per_ut_pixel (int): XCT pixels per UT pixel
        patch_ut (int): Patch side in UT pixels
        xct_resolution (float, optional): Voxel size in mm. Defaults to 0.025.
        frontwall_z (int, optional): Material frontwall slice, origin of the pore depth.
        material_threshold (float, optional): Patch validity threshold. Defaults to 0.8.
        min_voxels, min_axis (int, optional): Pore filter, see `compute_pore_features`.

    Returns:
        tuple: (pores_df, pore_patches_df, pore_count_map)
            - pores_df: one row per kept pore with patch indices, the patch UT range,
              the UT pixel (ut_x, ut_y) containing the pore centroid, the centroid depth
              from the frontwall in mm and the shape features.
            - pore_patches_df: one row per patch with material voxels, validity,
              pore_count and filtered pore/voxel counts.
            - pore_count_map: (patch_x, patch_y) float32, -1 for invalid patches.
    """
    px = xct_pixels_per_ut_pixel
    P = patch_ut * px
    _, X, Y = mask_grid.shape
    npx, npy = X // P, Y // P
    if npx == 0 or npy == 0:
        raise ValueError(f"Patch of {patch_ut} UT pixels does not fit in the UT grid {(X // px, Y // px)}")

    mat = np.zeros((npx, npy), dtype=np.int64)
    for i in range(npx):
        for j in range(npy):
            mat[i, j] = mask_grid[:, i * P:(i + 1) * P, j * P:(j + 1) * P].sum()
    mat_fraction = mat / mat.max() if mat.max() > 0 else np.zeros(mat.shape)
    valid = mat_fraction >= material_threshold

    pore_parts, patch_rows = [], []
    pore_count_map = np.full((npx, npy), -1, dtype=np.float32)
    for i in range(npx):
        for j in range(npy):
            row = {'patch_i': i, 'patch_j': j,
                   'ut_x0': i * patch_ut, 'ut_x1': (i + 1) * patch_ut,
                   'ut_y0': j * patch_ut, 'ut_y1': (j + 1) * patch_ut,
                   'material_voxels': int(mat[i, j]), 'material_fraction': float(mat_fraction[i, j]),
                   'valid': bool(valid[i, j]), 'pore_count': np.nan,
                   'removed_pores': np.nan, 'removed_voxels': np.nan}
            if valid[i, j]:
                blk = onlypores_grid[:, i * P:(i + 1) * P, j * P:(j + 1) * P]
                feats, n_rm, vox_rm = compute_pore_features(blk, xct_resolution, min_voxels, min_axis)
                kept_vox = int(feats.volume_vox.sum()) if len(feats) else 0
                assert kept_vox + vox_rm == int(blk.sum()), 'kept + removed pore voxels != patch pore voxels'
                row.update(pore_count=len(feats), removed_pores=n_rm, removed_voxels=vox_rm)
                pore_count_map[i, j] = len(feats)
                if len(feats):
                    ut_x = i * patch_ut + np.floor(feats.centroid_x_vox / px).astype(int)
                    ut_y = j * patch_ut + np.floor(feats.centroid_y_vox / px).astype(int)
                    feats.insert(0, 'patch_j', j)
                    feats.insert(0, 'patch_i', i)
                    feats.insert(3, 'ut_x0', i * patch_ut)
                    feats.insert(4, 'ut_x1', (i + 1) * patch_ut)
                    feats.insert(5, 'ut_y0', j * patch_ut)
                    feats.insert(6, 'ut_y1', (j + 1) * patch_ut)
                    feats.insert(7, 'ut_x', ut_x.to_numpy())
                    feats.insert(8, 'ut_y', ut_y.to_numpy())
                    feats['centroid_z_mm_from_frontwall'] = (feats.centroid_z_vox - frontwall_z) * xct_resolution
                    pore_parts.append(feats)
            patch_rows.append(row)

    pores_df = pd.concat(pore_parts, ignore_index=True) if pore_parts else pd.DataFrame()
    return pores_df, pd.DataFrame(patch_rows), pore_count_map


def main_images_pores(onlypores, mask, ut_rf, xct_resolution=0.025, ut_resolution=1.0,
                      material_threshold=0.8, pore_patch_ut=5,
                      min_pore_voxels=8, min_pore_axis_voxels=2):
    """
    Image-shaped UT vs XCT pipeline with pore-level targets (block reduction only).

    Same preprocessing and maps as `main_images`, plus:
    - per-UT-pixel material voxel count, thickness (mm) and full-depth VVF maps;
    - pore count and per-pore features on `pore_patch_ut` x `pore_patch_ut` UT-pixel patches.

    Args:
        onlypores, mask, ut_rf (np.ndarray): XCT pores, XCT material mask, UT volume (z, x, y)
        xct_resolution, ut_resolution (float): Resolutions in mm
        material_threshold (float, optional): Valid-pixel/patch material threshold. Defaults to 0.8.
        pore_patch_ut (int, optional): Pore patch side in UT pixels. Defaults to 5.
        min_pore_voxels, min_pore_axis_voxels (int, optional): Pore filter.

    Returns:
        dict: ut_image (z, x, y); volfrac_maps (3, x, y); areafrac_image (x, y);
            depth_rois; vvf_full_map (x, y); material_voxels_map (x, y);
            thickness_map (x, y, mm);
            pores_df; pore_patches_df; pore_count_map (x//patch, y//patch)
    """
    xct_pixels = calculate_pixels(ut_resolution, xct_resolution, 1)
    px = int(np.round(xct_pixels))
    if px <= 0 or not np.isclose(xct_pixels, px):
        raise ValueError(f"UT/XCT resolution ratio must be an integer. Got {xct_pixels}.")

    onlypores_cropped, mask_cropped, ut_rf_cropped = preprocess(
        onlypores, mask, ut_rf, xct_resolution, ut_resolution)
    onlypores_grid, mask_grid, ut_image = align_to_ut_grid(
        onlypores_cropped, mask_cropped, ut_rf_cropped, px)

    volfrac_maps, areafrac_image, depth_rois = calculate_fraction_maps(
        onlypores_grid, mask_grid, px, material_threshold=material_threshold)
    material_voxels_map = calculate_material_voxels_map(mask_grid, px)
    thickness_map = calculate_thickness_map(mask_grid, px, xct_resolution)
    vvf_full_map = calculate_full_vvf_map(onlypores_grid, mask_grid, px, material_threshold)

    pores_df, pore_patches_df, pore_count_map = calculate_pore_patch_targets(
        onlypores_grid, mask_grid, px, pore_patch_ut,
        xct_resolution=xct_resolution,
        frontwall_z=depth_rois[0]['material_frontwall_z'],
        material_threshold=material_threshold,
        min_voxels=min_pore_voxels, min_axis=min_pore_axis_voxels)

    return {
        'ut_image': ut_image, 'volfrac_maps': volfrac_maps, 'areafrac_image': areafrac_image,
        'depth_rois': depth_rois, 'vvf_full_map': vvf_full_map,
        'material_voxels_map': material_voxels_map, 'thickness_map': thickness_map, 'pores_df': pores_df,
        'pore_patches_df': pore_patches_df, 'pore_count_map': pore_count_map,
    }
