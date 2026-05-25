from tqdm import tqdm
from pathlib import Path
import tifffile
import os
import numpy as np
import re

RAW_DIMENSIONS_PATTERN = re.compile(r"(?<!\d)(\d+)\s*x\s*(\d+)\s*x\s*(\d+)(?!\d)", re.IGNORECASE)

def infer_raw_shape(path):
    """
    Infer RAW volume shape from a filename containing widthxheightxdepth.

    Args:
    path (str): Path to the RAW file. The filename must contain dimensions like
        sample_300x295x448.raw, where values are width, height, depth.

    Returns:
    tuple: Shape in numpy order (depth, height, width).
    """

    matches = RAW_DIMENSIONS_PATTERN.findall(Path(path).stem)
    if not matches:
        raise ValueError(
            "Could not infer dimensions from the RAW filename. "
            "The filename must contain widthxheightxdepth, "
            "for example sample_300x295x448.raw."
        )

    width, height, depth = (int(value) for value in matches[-1])
    return (depth, height, width)

def dtype_with_byte_order(dtype, byte_order):
    """
    Return a numpy dtype with the requested byte order.

    Args:
    dtype (str or numpy.dtype): RAW data type.
    byte_order (str): One of "little", "big", or "native".

    Returns:
    numpy.dtype: Dtype with byte order applied.
    """

    dtype = np.dtype(dtype)
    if dtype.itemsize == 1:
        return dtype

    byte_orders = {
        "little": "<",
        "big": ">",
        "native": "=",
    }
    if byte_order not in byte_orders:
        raise ValueError('byte_order must be "little", "big", or "native".')

    return dtype.newbyteorder(byte_orders[byte_order])

def load_raw(path, dtype="uint8", shape=None, byte_order="little", use_memmap=True):
    """
    Load a RAW volume as a 3D numpy array.

    Args:
    path (str): Path to the RAW file.
    dtype (str or numpy.dtype, optional): RAW data type. Defaults to "uint8".
    shape (tuple, optional): Shape in numpy order (depth, height, width). If None,
        dimensions are inferred from the filename as widthxheightxdepth.
    byte_order (str, optional): One of "little", "big", or "native". Defaults to "little".
    use_memmap (bool, optional): If True, memory-map the RAW file instead of loading
        it fully into RAM. Defaults to True.

    Returns:
    numpy.ndarray: A 3D array representing the volume.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if shape is None:
        shape = infer_raw_shape(path)
    else:
        shape = tuple(int(axis) for axis in shape)

    dtype = dtype_with_byte_order(dtype, byte_order)
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    actual_bytes = path.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size does not match the requested shape and dtype.\n"
            f"Expected: {expected_bytes:,} bytes for shape {shape} and dtype {dtype}.\n"
            f"Actual:   {actual_bytes:,} bytes."
        )

    if use_memmap:
        return np.memmap(path, mode="r", dtype=dtype, shape=shape)

    return np.fromfile(path, dtype=dtype).reshape(shape)

def raw_to_tif(raw_path, tif_path, dtype="uint8", shape=None, byte_order="little", use_memmap=True):
    """
    Convert a RAW volume to a TIF/TIFF file.

    Args:
    raw_path (str): Path to the RAW file.
    tif_path (str): Path to the output TIF/TIFF file.
    dtype (str or numpy.dtype, optional): RAW data type. Defaults to "uint8".
    shape (tuple, optional): Shape in numpy order (depth, height, width). If None,
        dimensions are inferred from the RAW filename as widthxheightxdepth.
    byte_order (str, optional): One of "little", "big", or "native". Defaults to "little".
    use_memmap (bool, optional): If True, memory-map the RAW file. Defaults to True.

    Returns:
    pathlib.Path: Path to the saved TIF/TIFF file.
    """

    volume = load_raw(raw_path, dtype=dtype, shape=shape, byte_order=byte_order, use_memmap=use_memmap)
    tif_path = Path(tif_path)
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(tif_path, volume, bigtiff=True, photometric="minisblack", metadata={"axes": "ZYX"})
    return tif_path

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

def write_sequence(folder_path, volume, ignore_not_empty=False):
    """
    Save a 3D volume as a sequence of TIFF files in a folder.
    
    Args:
    folder_path (str): Path to the folder where TIFF files will be saved.
    volume (numpy.ndarray): A 3D array where each slice corresponds to an image.
    ignore_not_empty (bool, optional): If False and .tif files exists in folder_path, raise an error. Defaults to False.
    """
    
    # Check if folder contains .tif or .tiff files and ignore_not_empty is False
    if not ignore_not_empty:
        existing_tiff_files = [f for f in os.listdir(folder_path) if (f.endswith('.tiff') or f.endswith('.tif'))]
        if existing_tiff_files:
            raise ValueError(f"Folder {folder_path} already contains TIFF files. Set ignore_not_empty=True to override.")

    # Save each slice as a TIFF file with progress bar
    with tqdm(total=volume.shape[0], desc="Saving") as pbar:
        for i in range(volume.shape[0]):
            tifffile.imwrite(f"{folder_path}/{i:04d}.tif", volume[i])
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

def save_tif(path, volume, ignore_not_empty=False):
    """
    Save a 3D volume as a TIF/TIFF file or a sequence of TIF/TIFF files.

    Args:
    path (str): Path to the output TIFF file/folder.
    volume (numpy.ndarray): A 3D array representing the volume.
    ignore_not_empty (bool, optional): If False and .tif files exists in folder_path, raise an error. Defaults to False.
    """
    
    # Check if the path is a directory or a file
    if os.path.isdir(path):
        # If it's a directory, save the volume as a sequence of TIFF files
        write_sequence(path, volume, ignore_not_empty)
    else:
        # If it's a file, save the single TIFF file
        tifffile.imwrite(path, volume)
