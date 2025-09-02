import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes
from joblib import Parallel, delayed
from skimage.measure import label, regionprops
from . import signal

def isolate_samples(volume, n_samples):
    """
    Process the volume to extract sample volumes.

    Args:
    volume (numpy.ndarray): The 3D volume to process.
    n_samples (int): The number of samples to extract.

    Returns:
    list: A list of sample volumes.
    """
    print('Segmenting with Otsu')
    # Step 1: Apply Otsu's threshold to segment the volume into foreground (samples) and background
    thresh = threshold_otsu(volume)
    binary = volume > thresh

    print("Filling holes")
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

    print("Labeling connected regions")
    # Step 5: Label connected regions in the 3D binary volume (each sample gets a unique label)
    label_image = label(filled)

    # Step 6: Extract properties (such as area and bounding box) of each labeled region
    props = regionprops(label_image)

    # Step 7: Sort the regions by area in descending order (largest samples first)
    props.sort(key=lambda x: x.area, reverse=True)

    minimum_value = volume[volume.shape[0] // 2, volume.shape[1] // 2].min()

    def process_sample(volume, label_image, label, bbox):
        # Extract a single sample from the original volume using its label and bounding box
        sample = volume.copy()
        # Set all voxels outside the current label to the minimum value of the middle of the volume (background)
        sample[label_image != label] = minimum_value
        # Crop the sample to its bounding box
        sample = sample[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        return sample
    
    print("Extracting samples")
    # Step 8: Extract the n_samples largest samples in parallel
    volumes = Parallel(n_jobs=-1)(
        delayed(process_sample)(volume, label_image, props[i].label, props[i].bbox)
        for i in range(n_samples)
    )
    
    # Step 9: Collect bounding boxes for sorting
    bboxes = [props[i].bbox for i in range(n_samples)]

    # Step 10: Order the samples by their position along the z-axis (bbox[2])
    volumes = [v for _, v in sorted(zip(bboxes, volumes), key=lambda pair: pair[0][2])]

    # Step 12: Return the list of extracted sample volumes
    return volumes

def isolate_samples_ut(volume, n_samples):
    """
    Isolate UT samples from a 3D volume.

    Args:
    volume (numpy.ndarray): The 3D ut volume to process with axes (z,y,x) and RF signal.
    n_samples (int): The number of UT samples to extract.

    Returns:
    list: A list of isolated UT sample volumes.
    """

    #get envelope of the RF signal

    volume = volume.swapaxes(2,0) 

    envelope = signal.envelope(volume)

    envelope = envelope.swapaxes(2,0)
    volume = volume.swapaxes(2,0)  # Swap back to (z,y,x) for processing

    #get the maximum value of the envelope
    max_image = envelope.max(axis=0)

    # Step 1: Apply Otsu's threshold to segment the volume into foreground (samples) and background
    thresh = threshold_otsu(max_image)
    binary = max_image > thresh

    # Step 2: Fill holes in each 2D slice to ensure samples are solid regions
    filled = binary_fill_holes(binary)

    # Step 3: Label connected regions in the 2D binary volume (each sample gets a unique label)
    label_image = label(filled)

    # Step 4: Extract properties (such as area and bounding box) of each labeled region
    props = regionprops(label_image)

    # Step 5: Sort the regions by area in descending order (largest samples first)
    props.sort(key=lambda x: x.area, reverse=True)

    def process_sample(volume, label_image, label, bbox):
        """
        Extract a single UT sample from the original volume using its label and bounding box.
        
        Args:
        volume (numpy.ndarray): The 3D volume to extract the sample from.
        label_image (numpy.ndarray): The labeled image indicating sample regions.
        label (int): The label of the sample to extract.
        bbox (tuple): The bounding box of the sample.

        Returns:
        numpy.ndarray: The extracted UT sample volume.
        """
        sample = volume.copy()

        # Crop the sample to its bounding box
        sample = sample[:,bbox[0]-10:bbox[2]+10, bbox[1]-10:bbox[3]+10]
        return sample

    # Step 6: Extract the n_samples largest samples sequentially
    volumes = [
        process_sample(volume, label_image, props[i].label, props[i].bbox)
        for i in range(n_samples)
    ]

    # Step 7: Collect bounding boxes for sorting
    bboxes = [props[i].bbox for i in range(n_samples)]

    # Step 8: Order the samples by their position along the z-axis (bbox[2])
    volumes = [v for _, v in sorted(zip(bboxes, volumes), key=lambda pair: pair[0][2])]

    # Step 9: Return the list of extracted UT sample volumes
    return volumes



    