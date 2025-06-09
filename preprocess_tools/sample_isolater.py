import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes
from joblib import Parallel, delayed
from skimage.measure import label, regionprops

def isolate_samples(volume, n_samples):
    """
    Process the volume to extract sample volumes.

    Args:
    volume (numpy.ndarray): The 3D volume to process.
    n_samples (int): The number of samples to extract.

    Returns:
    list: A list of sample volumes.
    """
    # Step 1: Apply Otsu's threshold to segment the volume into foreground (samples) and background
    thresh = threshold_otsu(volume)
    binary = volume > thresh

    # Step 2: Fill holes in each 2D slice to ensure samples are solid regions
    filled = np.zeros_like(binary)

    def process_slice(i):
        # Fill holes in a single 2D slice
        return binary_fill_holes(binary[i])

    # Step 3: Use parallel processing to fill holes in all slices for efficiency
    filled_slices = Parallel(n_jobs=-1)(
        delayed(process_slice)(i) for i in range(binary.shape[0])
    )

    # Step 4: Combine the processed slices back into a 3D volume
    for i in range(binary.shape[0]):
        filled[i] = filled_slices[i]

    # Step 5: Label connected regions in the 3D binary volume (each sample gets a unique label)
    label_image = label(filled)

    # Step 6: Extract properties (such as area and bounding box) of each labeled region
    props = regionprops(label_image)

    # Step 7: Sort the regions by area in descending order (largest samples first)
    props.sort(key=lambda x: x.area, reverse=True)

    def process_sample(volume, label_image, label, bbox):
        # Extract a single sample from the original volume using its label and bounding box
        sample = volume.copy()
        # Set all voxels outside the current label to the minimum value in the center slice (background)
        value = sample[sample.shape[0] // 2, sample.shape[1] // 2].min()
        sample[label_image != label] = value
        # Crop the sample to its bounding box
        sample = sample[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        return sample

    # Step 8: Extract the n_samples largest samples in parallel
    volumes = Parallel(n_jobs=-1)(
        delayed(process_sample)(volume, label_image, props[i].label, props[i].bbox)
        for i in range(n_samples)
    )

    # Step 9: Collect bounding boxes for sorting
    bboxes = [props[i].bbox for i in range(n_samples)]

    # Step 10: Order the samples by their position along the z-axis (bbox[2])
    volumes = [v for _, v in sorted(zip(bboxes, volumes), key=lambda pair: pair[0][2])]

    # Step 12: Replace all the pixels with value 0 with the minimum value of the center slice of the original volume
    center_value = volume[volume.shape[0] // 2, volume.shape[1] // 2].min()
    for i in range(len(volumes)):
        volumes[i][volumes[i] == 0] = center_value

    # Step 13: Return the list of extracted sample volumes
    return volumes
