import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects

def flat_filter(labeled_volume, min_size = 2):
    """
    Filters pores in a binary volume based on their size in each dimension.
    
    This function removes small artifacts from the volume that don't meet the minimum
    size criteria in all three dimensions (x, y, z). Only objects with a bounding box
    larger than or equal to min_size in each dimension are kept.
    
    Parameters:
    -----------

    labeled_volume : numpy.ndarray
        A labeled 3D numpy array (dtype int) where each connected component
        (pore) is assigned a unique label. This can be obtained using `skimage.measure.label`.
    
    min_size : int, optional (default=2)
        The minimum size (in pixels) that a pore must have in each dimension
        (width, height, depth) to be kept in the filtered volume. Objects smaller
        than this in any dimension will be considered artifacts and removed.
    
    Returns:
    --------
    numpy.ndarray
        A filtered binary volume with the same shape as the input, where
        small artifacts have been removed.
    """
    
    props = regionprops(labeled_volume)

    # Create a mask to keep only the objects that are 2 pixels wide in each axis
    mask = np.zeros_like(labeled_volume, dtype=bool)

    for prop in tqdm(props):
        # Get the bounding box of the object
        min_row, min_col, min_depth, max_row, max_col, max_depth = prop.bbox
        # Check if the object is 2 pixels wide in each axis
        if (max_row - min_row >= min_size) and (max_col - min_col >= min_size) and (max_depth - min_depth >= min_size):
            mask[prop.slice] = labeled_volume[prop.slice] == prop.label

    # Apply the mask to the labeled volume
    filtered_pores = labeled_volume * mask

    return filtered_pores

def complete_filtering(volume,small_objects_size=8, flat_size=2):
    """
    Complete filtering of a 3D volume to remove small artifacts.
    
    This functions applies a remove small objects function with size 8 and then a flat filtering

    to remove small artifacts based on their size in each dimension.
    Parameters:
    -----------
    volume : numpy.ndarray
        A binarized 3D numpy array (dtype uint8) where:
        - Pores/objects are represented by the value 255
        - Background is represented by the value 0
    min_size : int, optional (default=2)
        The minimum size (in pixels) that a pore must have in each dimension
        (width, height, depth) to be kept in the filtered volume. Objects smaller
        than this in any dimension will be considered artifacts and removed.
    Returns:
    --------
    numpy.ndarray
        A filtered binary volume with the same shape as the input, where
        small artifacts have been removed.
    """

    # Label the pores in the volume
    labeled_pores = label(volume, connectivity=1)

    #filter
    filtered_pores = remove_small_objects(labeled_pores, min_size=small_objects_size)
    
    # Then, apply the flat filter to remove artifacts based on size in each dimension
    filtered_pores = flat_filter(filtered_pores, min_size=flat_size)

    return ((filtered_pores > 0) * 255).astype(np.uint8)

