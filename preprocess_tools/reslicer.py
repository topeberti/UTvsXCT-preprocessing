"""
Reslicer module for 3D volume manipulation.

This module provides functions to reslice and rotate 3D volumes,
which is useful for viewing data from different orientations.
"""
import numpy as np
    
def reslice(volume, name):
    """
    Reslice a 3D volume according to the specified orientation.
    
    Transforms the input volume to present different viewing perspectives
    (Top, Left, Right, Bottom) by applying appropriate transpositions
    and axis flips.
    
    Parameters:
    -----------
    volume : numpy.ndarray
        3D numpy array representing the input volume
    name : str
        Orientation name ('Top', 'Left', 'Right', 'Bottom')
        
    Returns:
    --------
    numpy.ndarray
        Resliced volume according to the specified orientation
    """
    if name == 'Top':
        resliced = np.transpose(volume, (1, 0, 2))[::-1, :, :]
        resliced = np.flip(resliced, axis=1)
    elif name == 'Left':
        resliced = np.transpose(volume, (2, 1, 0))
        resliced = np.flip(resliced, axis=1)
        resliced = np.flip(resliced, axis=2)
    elif name == 'Right':
        resliced = np.transpose(volume, (2, 1, 0))[::-1, :, :]
        resliced = np.flip(resliced, axis=1)
    elif name == 'Bottom':
        resliced = np.transpose(volume, (1, 0, 2))
    
    return resliced

def rotate_auto(volume):
    """
    Automatically rotate the input volume by -90 degrees (counterclockwise) around axes (1, 2).
    
    Parameters:
    -----------
    volume : numpy.ndarray
        3D numpy array representing the input volume
        
    Returns:
    --------
    numpy.ndarray
        Rotated volume with a -90 degree rotation applied to the second and third axes
    """
    return np.rot90(volume, -1, (1, 2))