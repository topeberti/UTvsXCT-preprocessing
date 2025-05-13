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

    Parameters:
    - volume: 3D numpy array
    - window_size: size of the local window for Sauvola thresholding
    - k: parameter for Sauvola thresholding
    - min_size: minimum size of objects to keep

    Returns:
    - binary_volume: 3D binary numpy array after Sauvola thresholding and small object removal
    """

    def sauvola_slice(slice_):
        sauvola_threshold = threshold_sauvola(slice_, window_size=49, k=0.05)
        return slice_ < sauvola_threshold

    def remove_small_objects_slice(slice_):
        labeled_slice = label(slice_)
        return remove_small_objects(labeled_slice, min_size=100) <= 0

    # Apply Sauvola thresholding to each slice in parallel
    sauvola_noisy = np.array(Parallel(n_jobs=-1)(delayed(sauvola_slice)(volume[i]) for i in range(volume.shape[0])))

    # Remove small objects from each slice in parallel
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

def onlypores(xct):
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

def onlypores_parallel(xct):

    # Function to apply the mask and compress the data
    def mask_and_compress(xct_chunk):
        masked_data = np.ma.masked_equal(xct_chunk, 0)
        return masked_data.compressed()
    
    print('masking')
    
    # Number of chunks (adjust depending on the size of your array and available cores)
    num_chunks = 16  # You can increase or decrease this based on testing

    # Assuming xct is your large array, divide it into chunks
    chunks = np.array_split(xct, num_chunks)

    # Use joblib to apply the function in parallel
    compressed_chunks = Parallel(n_jobs=-1, backend='loky')(delayed(mask_and_compress)(chunk) for chunk in chunks)

    # Combine the results back into a single array
    unmasked_data = np.concatenate(compressed_chunks)

    print('Thresholding')

    binary = sauvola_thresholding(unmasked_data)

    max_proj = np.max(binary, axis=0)

    labels = measure.label(max_proj)

    props = regionprops(labels)

    minr, minc, maxr, maxc = props[0].bbox

    #crop the volume

    binary_cropped = binary[:, minr:maxr, minc:maxc]

    sample_mask_cropped = fill_voids.fill(binary_cropped, in_place=False)

    sample_mask = np.zeros_like(binary)
    sample_mask[:, minr:maxr, minc:maxc] = sample_mask_cropped

    #invert binary
    binary_inverted = np.invert(binary)
    onlypores = np.logical_and(binary_inverted, sample_mask)

    return onlypores, sample_mask, binary

def material_mask_parallel(xct):

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