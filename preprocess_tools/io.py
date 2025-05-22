from tqdm import tqdm
from pathlib import Path
import tifffile
import os
import numpy as np

def read_sequence(folder_path):
    """
    Read a sequence of TIFF files in a folder as a 3D volume.
    
    Args:
    folder_path (str): Path to the folder containing TIFF files.

    Returns:
    numpy.ndarray: A 3D array where each slice corresponds to a TIFF file.
    """

    # List and sort the TIFF files
    tiff_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if (f.endswith('.tiff') or f.endswith('.tif'))])
    
    # Get the total number of TIFF files
    total_files = len(tiff_files)
    
    # Read each TIFF file and update progress
    volume = []
    with tqdm(total=total_files, desc="Progress") as pbar:
        for i, file_path in enumerate(tiff_files):
            slice_data = tifffile.imread(file_path)
            volume.append(slice_data)
            
            # Update progress
            pbar.update(1)
    
    return np.array(volume)

def write_sequence(folder_path, name, volume):
    """
    Save a 3D volume as a sequence of TIFF files in a folder.
    
    Args:
    folder_path (str): Path to the folder where TIFF files will be saved.
    name (str): Name of the TIFF files.
    volume (numpy.ndarray): A 3D array where each slice corresponds to an image.
    """

    folder_path = folder_path / name

    # Create the folder if it doesn't exist
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # Save each slice as a TIFF file with progress bar
    with tqdm(total=volume.shape[0], desc="Saving") as pbar:
        for i in range(volume.shape[0]):
            tifffile.imwrite(f"{folder_path}/{name}_{i:04d}.tif", volume[i])
            pbar.update(1)
    
    print("Saving complete.")

def to_matrix(string):

    """
    Convert a database string representation of an affine transform matrix into a numpy array.
    
    Args:
    string (str): A database string representation of the affine transform matrix.

    Returns:
    numpy.ndarray: A 3x3 numpy array representing the affine transform matrix.
    """

    matrix1 = float(string[2:17])

    matrix2 = float(string[17:33])

    matrix3 = float(string[33:49])

    matrix4 = float(string[53:68])

    matrix5 = float(string[68:84])

    matrix6 = float(string[84:100])

    matrix7 = float(string[105:121])

    matrix8 = float(string[121:137])

    matrix9 = float(string[137:151])

    matrix = np.array([[matrix1,matrix2,matrix3],[matrix4,matrix5,matrix6],[matrix7,matrix8,matrix9]])

    return matrix

def load_tif(path):
    """
    Load a 3D volume from a TIF/TIFF file or a TIF/TIFF folder.

    Args:
    path (str): Path to the TIFF file/folder.

    Returns:
    numpy.ndarray: A 3D array representing the volume.
    """
    
    # Check if the path is a directory or a file
    if os.path.isdir(path):
        # If it's a directory, read all TIFF files in the directory
        volume = read_sequence(path)
    elif os.path.isfile(path):
        # If it's a file, read the single TIFF file
        volume = tifffile.imread(path)
    else:
        raise ValueError("Invalid path: must be a directory or a TIFF file.")
    
    return volume

def save_tif(path, volume):
    """
    Save a 3D volume as a TIF/TIFF file or a sequence of TIF/TIFF files.

    Args:
    path (str): Path to the output TIFF file/folder.
    volume (numpy.ndarray): A 3D array representing the volume.
    """
    
    # Check if the path is a directory or a file
    if os.path.isdir(path):
        # If it's a directory, save the volume as a sequence of TIFF files
        write_sequence(path, "output", volume)
    else:
        # If it's a file, save the single TIFF file
        tifffile.imwrite(path, volume)