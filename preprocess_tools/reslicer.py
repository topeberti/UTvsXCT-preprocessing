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
        resliced = np.transpose(volume, (1, 0, 2))
        resliced = np.flip(resliced, axis=1)
        resliced = np.flip(resliced, axis=2)
    elif name == 'Left':
        resliced = np.transpose(volume, (2, 1, 0))
        resliced = np.flip(resliced, axis=1)
        resliced = np.flip(resliced, axis=2)
    elif name == 'Right':
        resliced = np.transpose(volume, (2, 1, 0))
        resliced = np.flip(resliced, axis=0)
        resliced = np.flip(resliced, axis=1)
    elif name == 'Bottom':
        resliced = np.transpose(volume, (1, 0, 2))
        resliced = np.flip(resliced, axis=0)
        resliced = np.flip(resliced, axis=2)
    
    return resliced

def rotate_90(volume, clockwise=True):
    """
    Rotate the input volume by 90 degrees around axes (1, 2).
    
    Parameters:
    -----------
    volume : numpy.ndarray
        3D numpy array representing the input volume
    clockwise : bool, optional
        If True, rotates 90 degrees clockwise; if False, rotates 90 degrees counterclockwise.
        Default is True (clockwise).
        
    Returns:
    --------
    numpy.ndarray
        Rotated volume with a rotation of 90 degrees around axes (1, 2) in the specified direction.
    """
    if clockwise:
        rotated = np.rot90(volume, k=-1, axes=(1, 2))
    else:
        rotated = np.rot90(volume, k=1, axes=(1, 2))
    return rotated