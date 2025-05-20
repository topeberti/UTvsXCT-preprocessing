import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import measure
from skimage.measure import regionprops
import pandas as pd
from skimage.util import view_as_windows
from skimage.measure import label, regionprops
import scipy.ndimage
from joblib import Parallel, delayed


def divide_into_patches(image, patch_size, step_size):
    patches = view_as_windows(image, (image.shape[0], patch_size, patch_size), step=(image.shape[0], step_size, step_size))
    return patches.reshape(-1, image.shape[0], patch_size, patch_size)

def generate_patches(image, patch_size, step_size):
    _, height, width = image.shape
    for i in range(0, height - patch_size + 1, step_size):
        for j in range(0, width - patch_size + 1, step_size):
            patch = image[:, i:i + patch_size, j:j + patch_size]
            yield patch

def calculate_patch_shape(image_shape, patch_size, step_size):
    # Calculate the number of patches along each dimension
    num_patches_h = ((image_shape[1] - patch_size) // step_size) + 1
    num_patches_w = ((image_shape[2] - patch_size) // step_size) + 1

    # The outgoing shape would be (num_patches_h, num_patches_w, image_shape[0], patch_size, patch_size)
    return (num_patches_h * num_patches_w, image_shape[0], patch_size, patch_size)


def calculate_pixels(ut_resolution, xct_resolution, ut_pixels):
    # Calculate the ratio of the resolutions
    resolution_ratio = ut_resolution /xct_resolution 
    
    # Calculate the equivalent number of pixels in the xct resolution
    xct_pixels = ut_pixels * resolution_ratio
    
    return xct_pixels

def crop_image_to_patch_size(image, patch_size):
    z, x, y = image.shape

    crop_x = x % patch_size
    crop_y = y % patch_size

    crop_x_before = crop_x // 2
    crop_x_after = crop_x - crop_x_before

    crop_y_before = crop_y // 2
    crop_y_after = crop_y - crop_y_before

    cropped_image = image[:, crop_x_before:-crop_x_after or None, crop_y_before:-crop_y_after or None]

    return cropped_image

def nearest_lower_divisible(num, divisor):
	if num % divisor == 0:
		return num
	else:
		remainder = num % divisor
		return int(num - remainder)

def nearest_higher_divisible(num, divisor):
    if num % divisor == 0:
        return num
    else:
        remainder = num % divisor
        return int(num + divisor - remainder)
    
def nearest_bounding_box(minr, minc, maxr, maxc, divisor):
    minr = nearest_higher_divisible(minr, divisor)
    minc = nearest_higher_divisible(minc, divisor)
    maxr = nearest_lower_divisible(maxr, divisor)
    maxc = nearest_lower_divisible(maxc, divisor)

    return minr, minc, maxr, maxc

def preprocess(onlypores,mask,ut_rf, xct_resolution = 0.025, ut_resolution = 1):

    #Cropeamos onlypores y la mascara para quitarnos el fondo y le aplicamos la misma bounding box a los datos de UT

    # Calculate scaling factor
    scaling_factor = xct_resolution / ut_resolution

    #XCT 

    max_proj = np.max(mask, axis=0)

    labels = measure.label(max_proj)

    props = regionprops(labels)

    minr_xct, minc_xct, maxr_xct, maxc_xct = props[0].bbox

    minr_xct, minc_xct, maxr_xct, maxc_xct = nearest_bounding_box(minr_xct, minc_xct, maxr_xct, maxc_xct, 1/scaling_factor)

    #crop the volume

    mask_cropped = mask[:, minr_xct:maxr_xct, minc_xct:maxc_xct]
    onlypores_cropped = onlypores[:, minr_xct:maxr_xct, minc_xct:maxc_xct]

    #UT

    # Convert bounding box to UT resolution
    minr_ut = int(minr_xct * scaling_factor)
    minc_ut = int(minc_xct * scaling_factor)
    maxr_ut = int(maxr_xct * scaling_factor)
    maxc_ut = int(maxc_xct * scaling_factor)

    ut_rf_cropped = ut_rf[:,minr_ut:maxr_ut, minc_ut:maxc_ut]

    return onlypores_cropped, mask_cropped, ut_rf_cropped

def patch(onlypores_cropped, mask_cropped, ut_rf_cropped, ut_patch_size = 3, ut_step_size =1, xct_resolution = 0.025, ut_resolution = 1):

    #compute xct patch size
    xct_patch_size = int(np.round(calculate_pixels(ut_resolution, xct_resolution, ut_patch_size)))
    xct_step_size = int(np.round(calculate_pixels(ut_resolution, xct_resolution, ut_step_size)))

    #crop volumes to fit patch size division
    ut_rf_cropped = crop_image_to_patch_size(ut_rf_cropped, ut_patch_size)
    onlypores_cropped = crop_image_to_patch_size(onlypores_cropped, xct_patch_size)
    mask_cropped = crop_image_to_patch_size(mask_cropped, xct_patch_size)

    #ensure patches are the same
    ut_shape = calculate_patch_shape(ut_rf_cropped.shape, ut_patch_size, ut_step_size)
    xct_shape = calculate_patch_shape(onlypores_cropped.shape, xct_patch_size, xct_step_size)

    if not (ut_shape[0] == xct_shape[0]):
        print('Patches are not the same')
        return 0,0,0,0
    
    #divide into patches
    patches_ut = divide_into_patches(ut_rf_cropped, ut_patch_size, ut_step_size)

    patches_onlypores = divide_into_patches(onlypores_cropped, xct_patch_size, xct_step_size)
    # center_size = int(patches_onlypores.shape[2] / ut_patch_size)
    # patches_onlypores = patches_onlypores[:, :, center_size:-center_size, center_size:-center_size]
    patches_mask = divide_into_patches(mask_cropped,xct_patch_size, xct_step_size)
    # patches_mask = patches_mask[:, :, center_size:-center_size, center_size:-center_size]

    return patches_onlypores, patches_mask, patches_ut, ut_rf_cropped.shape

def patch_2(onlypores_cropped, mask_cropped, ut_rf_cropped, ut_patch_size = 3, ut_step_size = 1, beam_size = 3, xct_resolution = 0.025, ut_resolution = 1):
    # Compute XCT patch size
    xct_patch_size = int(np.round(calculate_pixels(ut_resolution, xct_resolution, ut_patch_size)))
    xct_beam_size = int(np.round(calculate_pixels(ut_resolution, xct_resolution, beam_size)))
    xct_step_size = int(np.round(calculate_pixels(ut_resolution, xct_resolution, ut_step_size)))

    #crop volumes to fit patch size division
    ut_rf_cropped = crop_image_to_patch_size(ut_rf_cropped, ut_patch_size)
    onlypores_cropped = crop_image_to_patch_size(onlypores_cropped, xct_patch_size)
    mask_cropped = crop_image_to_patch_size(mask_cropped, xct_patch_size)

    #pad onlypores and mask to fit beam size division
    pad = (xct_beam_size - xct_patch_size) // 2
    onlypores_cropped = np.pad(onlypores_cropped, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    mask_cropped = np.pad(mask_cropped, ((0, 0), (pad, pad), (pad, pad)), mode='constant')

    #ensure patches are the same
    ut_shape = calculate_patch_shape(ut_rf_cropped.shape, ut_patch_size, ut_step_size)
    xct_shape = calculate_patch_shape(onlypores_cropped.shape, xct_beam_size, xct_step_size)

    if not (ut_shape[0] == xct_shape[0]):
        print('Patches are not the same')
        return 0,0,0,0
    
    #divide into patches
    patches_ut = divide_into_patches(ut_rf_cropped, ut_patch_size, ut_step_size)

    patches_onlypores = divide_into_patches(onlypores_cropped, xct_beam_size, xct_step_size)
    patches_mask = divide_into_patches(mask_cropped,xct_beam_size, xct_step_size)

    return patches_onlypores, patches_mask, patches_ut

def clean_material(sum_mask, volfrac, areafrac):

    #get the value of a patch that is full material
    full_material = np.max(sum_mask)
    #get the percentage of material in each patch
    mat_percentage = sum_mask/full_material
    #the patches with less tahn 80% materail will be discarded
    indexes = np.where(mat_percentage < 0.8)[0]
    volfrac[indexes] = -1
    areafrac[indexes] = -1

    return volfrac,areafrac

def clean_pores_3D(patches_onlypores):

    # Function to process each slice
    def process_slice(i, patches_onlypores): 
        dilated_image = scipy.ndimage.binary_dilation(patches_onlypores[i], iterations=2) #dilate to group near pores
        labeled_image = label(dilated_image) #label the image
        #Now we get the indexes of the pixels that are not pores in the original image
        indexes = np.where(patches_onlypores[i] == 0) 
        #we delete the pixels that are not pores in the labeled image, because due to the dilation they are now labeled as pores
        labeled_image[indexes] = 0
        
        properties = []  # To store properties for this slice
        
        # Loop over the labels and get the needed properties
        for l in np.unique(labeled_image):
            if l == 0 and (np.unique(label) != 1):
                continue

            # Get the mask of the label so we only process the region
            indexes = np.where(labeled_image == l)
            mask = np.zeros_like(labeled_image)
            mask[indexes] = 1

            #we do the projections in xy to get the major and minor axis and in xz to get the length in z
            xyproj = np.max(mask, axis=0)
            xzproj = np.max(mask, axis=1)

            #now we get the first and last pixel in z to now the length of the pore in z 

            # Find the indices of all non-zero elements
            non_zero_indices = np.nonzero(xzproj)

            if non_zero_indices[0].size == 0:  # Handle case where there are no non-zero elements
                continue

            # Extract the row indices
            row_indices = non_zero_indices[0]

            # Determine the first and last row with a labeled pixel
            first_row = np.min(row_indices)
            last_row = np.max(row_indices)

            # Get properties in each projection
            xyprops = regionprops(xyproj)
            xzprops = [{'Z Length': last_row - first_row}]

            #save the properties in a dictionary
            for xy, xz in zip(xyprops, xzprops):
                if l == 0:
                    l = 1
                properties.append({
                    'Image Index': i,
                    'Label': l,
                    'xy Major Axis Length': xy.major_axis_length,
                    'xy Minor Axis Length': xy.minor_axis_length,
                    'Z Length': xz['Z Length'],
                    'Solidity': xy.solidity
                })
            
            if l == 0:
                labeled_image[labeled_image == 0] = 1
        
        return labeled_image, properties

    # Assuming proj_onlypores is already loaded as a numpy array
    # proj_onlypores = ...

    # Parallel processing of slices
    results = Parallel(n_jobs=-1, backend='loky')(delayed(process_slice)(i, patches_onlypores) for i in range(patches_onlypores.shape[0]))

    # Collect results
    labeled_volume_onlypores, properties_list = zip(*results)

    labeled_volume_onlypores = np.array(labeled_volume_onlypores)

    # Convert the properties list to a pandas DataFrame
    properties_df_3D = pd.DataFrame([item for sublist in properties_list for item in sublist])

    #now clean the pores that are too small
    #we dont want the pores that are smaller than 12 pixels in their major axis in xy

    z_lengths = properties_df_3D['Z Length'].values

    indexes = np.where(z_lengths < 0 )[0]

    cleaned_labeled_patches_onlypores = np.copy(labeled_volume_onlypores)

    for i in indexes:
        row = properties_df_3D.iloc[i]
        img_idx = int(row['Image Index'])
        delete_indexes = np.where(labeled_volume_onlypores[img_idx] == row['Label'])
        cleaned_labeled_patches_onlypores[img_idx][delete_indexes] = 0

    cleaned_patches_onlypores = cleaned_labeled_patches_onlypores > 0

    return cleaned_patches_onlypores

def layer_cleaning(mask_cropped, onlypores_cropped):

    layer_thickness = 0.508 #mm

    layer_thickness = int(np.round(layer_thickness / 0.025))

    indices = np.where(mask_cropped == 1)[0]

    frontwall = indices[0]

    backwall = indices[-1]

    edges = [frontwall]

    while True:

        edge = edges[-1]

        if edge + layer_thickness > backwall:

            break

        edges.append(edge + layer_thickness)

    onlypores_cleaned = np.copy(onlypores_cropped)

    #cleaning the last two layers because ut can't see them

    onlypores_cleaned[edges[-3]:] = 0
    
    return onlypores_cleaned

     

def datasets1(patches_onlypores, patches_mask, patches_ut, folder, ut_patch_size):

    #compute the sum of onlypores and mask
    sum_onlypores_patches = np.sum(patches_onlypores, axis = 1)
    sum_mask_patches = np.sum(patches_mask, axis = 1)

    proj_onlypores = np.max(patches_onlypores, axis = 1)
    proj_mask = np.max(patches_mask, axis = 1)

    #######volfrac for patch vs volfrac dataset

    sum_onlypores = np.sum(sum_onlypores_patches, axis = (1,2))
    sum_mask = np.sum(sum_mask_patches, axis = (1,2))

    #the points that are zero in the mask are not material, so we set them to -1 in volfrac to know that they are not material

    zero_indices = np.where(sum_mask == 0)

    volfrac = sum_onlypores / (sum_mask + 1e-6)

    volfrac[zero_indices] = -1

    #areafrac
    sum_onlypores_area = np.sum(proj_onlypores, axis = (1,2)).astype(np.int16)
    sum_mask_area = np.sum(proj_mask, axis = (1,2)).astype(np.int16)

    areafrac = sum_onlypores_area / (sum_mask_area + 1e-6)

    zero_indices = np.where(sum_mask_area == 0)

    areafrac[zero_indices] = -1

    #clean volfrac and areafrac
    volfrac, areafrac = clean_material(sum_mask, volfrac, areafrac)

    #prepare ut for dataframe

    ut_patches_reshaped = patches_ut.transpose(0, 2, 3, 1)

    ut_patches_reshaped = ut_patches_reshaped.reshape(ut_patches_reshaped.shape[0], -1)

    #create both dataframes

    #column names for ut

    columns_ut = []

    for i in range(ut_patches_reshaped.shape[1]):
        columns_ut.append(f'ut_rf_{i}')

    columns_ut = np.array(columns_ut)

    #dataframe for patch vs volfrac dataset

    patch_vs_volfrac = np.hstack((ut_patches_reshaped, volfrac.reshape(-1,1), areafrac.reshape(-1,1)))

    df_patch_vs_volfrac = pd.DataFrame(patch_vs_volfrac, columns = np.append(columns_ut, ['volfrac','areafrac']))

    #save the dataframes

    save_path = folder / ('patch_vs_volfrac_' + str(ut_patch_size) + '.csv')

    df_patch_vs_volfrac.to_csv(save_path, index = False)

    return df_patch_vs_volfrac, save_path

def main(onlypores,mask,ut_rf,folder, xct_resolution = 0.025, ut_resolution = 1,ut_patch_size = 3, ut_step_size =1):
    from time import time
    print('Preprocessing and patching the images...')
    #preprocess the images
    onlypores_cropped, mask_cropped, ut_rf_cropped = preprocess(onlypores,mask,ut_rf, xct_resolution, ut_resolution)    
    print('Layer cleaning')
    # onlypores_cropped = layer_cleaning(mask_cropped, onlypores_cropped)
    print('Patching the images...')
    #patch the images
    patches_onlypores, patches_mask, patches_ut, shape = patch(onlypores_cropped, mask_cropped, ut_rf_cropped, ut_patch_size, ut_step_size, xct_resolution, ut_resolution)
    print('Cleaning the pores...')
    # patches_onlypores = clean_pores_3D(patches_onlypores)
    print('Creating the datasets...')
    #create the datasets
    df, save_path = datasets1(patches_onlypores, patches_mask, patches_ut, folder,ut_patch_size)

    return shape, len(df), save_path