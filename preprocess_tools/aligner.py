# Import necessary libraries
import numpy as np                      # For numerical operations and array manipulations
from . import onlypores                # Custom module for material segmentation
from scipy.ndimage import affine_transform, rotate  # For geometric transformations
from skimage.restoration import estimate_sigma      # For noise estimation

def align_volume_xyz(volume, mask):
    """
    Aligns a 3D volume using Principal Component Analysis (PCA) such that the 
    principal axes of the object align with the coordinate axes (X, Y, Z).
    
    Args:
        volume (numpy.ndarray): 3D int8 volume with axes (x,y,z).
        mask (numpy.ndarray): Binary mask identifying the object in the volume.
        
    Returns:
        numpy.ndarray: Aligned volume (int8)
    """
    # 1. Get object voxel coordinates - these are the points where mask == 1
    points_xyz = np.argwhere(mask)  # Returns coordinates of non-zero points in the mask
    
    # Check if we have enough points to perform PCA
    if points_xyz.shape[0] < 3:
        print("Not enough object points found for PCA.")
        return volume, mask
    
    # 2. Compute centroid (center of mass) and center the points
    centroid_xyz = points_xyz.mean(axis=0)
    centered_points_xyz = points_xyz - centroid_xyz
    
    # 3. Calculate covariance matrix efficiently
    # The covariance matrix describes how coordinates vary together
    n_points = centered_points_xyz.shape[0]
    cov_matrix_xyz = np.dot(centered_points_xyz.T, centered_points_xyz) / (n_points - 1)
    
    # Add numerical stability for very small objects
    if n_points <= 3:  
        cov_matrix_xyz += np.eye(3) * 1e-9
    
    # 4. Find principal axes via eigendecomposition
    # eigvals are the variances along the principal axes
    # eigvecs are the directions of the principal axes
    try:
        eigvals_xyz, eigvecs_xyz = np.linalg.eigh(cov_matrix_xyz)
    except np.linalg.LinAlgError:
        print("LinAlgError during eigendecomposition. Object might be too small or degenerate.")
        return volume, mask
    
    # 5. Sort axes by importance (largest eigenvalue first)
    idx = eigvals_xyz.argsort()[::-1]
    eigvals_xyz = eigvals_xyz[idx]
    eigvecs_xyz = eigvecs_xyz[:, idx]
    
    # 6. Create transformation matrix to align principal axes with coordinate axes
    # Sort array dimensions by size to match principal axes to appropriate dimensions
    array_dims = np.array(volume.shape)
    dims_idx = np.argsort(array_dims)[::-1]
    
    # Build transformation matrix by mapping principal axes to sorted dimensions
    matrix_candidate = np.zeros((3, 3))
    for i in range(3):
        matrix_candidate[:, dims_idx[i]] = eigvecs_xyz[:, i]
    
    # 7. Ensure consistent orientation by making diagonal elements positive
    for i in range(3):
        if matrix_candidate[i, i] < 0:
            matrix_candidate[:, i] *= -1
    
    # 8. Ensure proper rotation matrix (determinant should be 1, not -1)
    if np.linalg.det(matrix_candidate) < 0:
        matrix_candidate[:, dims_idx[2]] *= -1  # Flip the shortest dimension

    # 9. Calculate offset and output shape to avoid clipping
    # Get all 8 corners of the original volume
    shape = np.array(volume.shape)
    corners = np.array(np.meshgrid([0, shape[0]-1], [0, shape[1]-1], [0, shape[2]-1])).T.reshape(-1,3)
    # Transform corners
    transformed_corners = np.dot(matrix_candidate, (corners - centroid_xyz).T).T + centroid_xyz
    min_corner = np.floor(transformed_corners.min(axis=0)).astype(int)
    max_corner = np.ceil(transformed_corners.max(axis=0)).astype(int)
    new_shape = (max_corner - min_corner + 1)
    # Calculate new center and offset
    output_center_xyz = (new_shape - 1) / 2.0
    offset_vector = centroid_xyz - matrix_candidate @ output_center_xyz

    print('Transforming')
    
    # 10. Apply the affine transformation to the volume
    aligned_volume = affine_transform(
        volume.astype(float),
        matrix=matrix_candidate,   # Rotation matrix
        offset=offset_vector,      # Translation vector
        output_shape=tuple(new_shape), # Expanded shape
        order=1,                   # Cubic interpolation for smooth results
        cval=40                    # Fill value for regions outside input volume
    )
    
    return aligned_volume.astype(np.uint8)

def measure_noise_skimage(image):
    """
    Estimate the noise level in an image using skimage's wavelet-based method.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        float: Estimated noise standard deviation
    """
    return estimate_sigma(image, average_sigmas=True)

def count_material(image):
    """
    Calculate the fraction of non-zero voxels in an image (material density).
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        float: Fraction of non-zero voxels (between 0 and 1)
    """
    return np.count_nonzero(image) / image.size

def find_flat_region_start(noise_values, pixels_values, window_size=5, threshold=0.01):
    """
    Find the index where both noise and material density curves stabilize,
    indicating the start of the consistent material region.
    
    Args:
        noise_values: Array of noise measurements across slices
        pixels_values: Array of material density measurements across slices
        window_size: Size of the moving window for gradient analysis
        threshold: Maximum gradient allowed to consider a region as "flat"
        
    Returns:
        int: Index where the flat region begins (-1 if not found)
    """
    # Calculate absolute gradients (rate of change) for both metrics
    noise_gradients = np.abs(np.gradient(noise_values))
    pixels_gradients = np.abs(np.gradient(pixels_values))
    
    # Smooth the gradients using moving average for more stable detection
    noise_smooth_gradients = np.convolve(noise_gradients, np.ones(window_size)/window_size, mode='valid')
    pixels_smooth_gradients = np.convolve(pixels_gradients, np.ones(window_size)/window_size, mode='valid')
    
    noise_index = -1
    pixels_index = -1

    # Find index where noise gradient stays below threshold for the entire window
    for i in range(len(noise_smooth_gradients) - window_size + 1):
        # Skip regions with too little material (< 30% density)
        if pixels_values[i] < 0.3:
            continue
        # Check if all gradients in window are below threshold
        if np.all(noise_smooth_gradients[i:i+window_size] < threshold):
            noise_index = (i + window_size//2)
            break
    
    # Find index where material density gradient stays below threshold
    for i in range(len(pixels_smooth_gradients) - window_size + 1):
        # Skip regions with too little material (< 30% density)
        if pixels_values[i] < 0.3:
            continue
        # Check if all gradients in window are below threshold
        if np.all(pixels_smooth_gradients[i:i+window_size] < threshold):
            pixels_index = (i + window_size//2)
            break
    
    # Return the later of the two indices (ensuring both metrics are stable)
    if noise_index >= 0 and pixels_index >= 0:
        return max(noise_index, pixels_index)
    
    # If no flat region found
    return -1

def find_frontwall(mask, max_slice=50):
    """
    Locate the front wall of the sample by analyzing noise and material density
    in consecutive slices until a stable region is found.
    
    Args:
        mask (numpy.ndarray): 3D binary mask of the object with axes (x,y,z).
        max_slice (int): Maximum number of slices to check from the beginning.
        
    Returns:
        int: Index of the front wall slice.
    """
    # Limit analysis to the first max_slice slices
    cropped_mask = mask[:,:, :max_slice]

    # Measure noise in each slice
    noise = np.zeros(cropped_mask.shape[2])
    for i in range(cropped_mask.shape[2]):
        noise[i] = measure_noise_skimage(cropped_mask[:,:,i])

    # Measure material density in each slice
    pixels = np.zeros(cropped_mask.shape[2])
    for i in range(cropped_mask.shape[2]):
        pixels[i] = count_material(cropped_mask[:,:,i])

    # Find where both noise and material density stabilize
    flat_start_idx = find_flat_region_start(noise, pixels, 5)

    return flat_start_idx

def centering(volume,mask):
    """
    Center the volume based on the material mask.
    
    Args:
        volume (numpy.ndarray): 3D uint8 volume with axes (x,y,z).
        mask (numpy.ndarray): Binary mask identifying the object in the volume.
        
    Returns:
        numpy.ndarray: Centered volume (uint8)
    """
    # Find the bounding box of the object in the mask
    coords = np.where(mask)
    zmin, zmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    xmin, xmax = coords[2].min(), coords[2].max()

    print(f"Bounding box coordinates: zmin={zmin}, zmax={zmax}, ymin={ymin}, ymax={ymax}, xmin={xmin}, xmax={xmax}")

    # Crop both arrays
    return volume[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1], mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]

def crop_volume(volume, mask):
    """
    Crop the volume to start at the front wall.

    Args:
        volume (numpy.ndarray): 3D uint8 volume with axes (x,y,z).
        mask (numpy.ndarray): Binary mask identifying the object in the volume.

    Returns:
        numpy.ndarray: Cropped volume (uint8)
    """

    # Find the front wall (first stable slice with material)
    front_wall_index = find_frontwall(mask)
    
    # Return the aligned volume cropped to start at the front wall
    return volume[:,:,front_wall_index:]

def main(volume,crop = False):
    """
    Main function that processes a 3D volume:
    1. Generates a material mask
    2. Aligns the volume to principal axes
    3. Finds the front wall
    4. Returns the aligned volume cropped (if crop = True) to start at the front wall
    
    Args:
        volume (numpy.ndarray): 3D int8 volume with axes (x,y,z).
        crop (bool): If True, crop the volume to start at the front wall.
        
    Returns:
        volume (numpy.ndarray): Aligned and cropped volume.
    """
    # Generate mask identifying material voxels in the original volume
    mask = onlypores.material_mask_parallel(volume)

    # Align the volume so principal axes match coordinate axes
    volume = align_volume_xyz(volume, mask)

    # Generate a new mask for the aligned volume
    mask = onlypores.material_mask_parallel(volume)

    volume,mask = centering(volume,mask)

    if not crop:

        return volume
    
    return crop_volume(volume, mask)