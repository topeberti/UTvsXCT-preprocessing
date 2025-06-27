#Signal processing tools for the UT vs XCT project.
"""
Signal Processing Module for Ultrasonic Testing (UT) vs X-ray Computed Tomography (XCT) Project

This module provides signal processing utilities for analyzing and manipulating 
ultrasonic testing data. It contains functions for calculating amplitude envelopes 
using Hilbert transform and aligning signals based on frontwall peaks.

Functions:
    envelope(volume): Applies Hilbert transform to calculate amplitude envelope
    align_signals_optimized(ut_patches, point): Aligns signals so frontwall peaks are at a specified point
    convert_to_IQ(signal): Converts a real-valued signal to its I/Q components
    get_IQ_patch(ut_patch): Converts a single UT patch to its I/Q components
    get_IQ_patches(ut_patches): Converts a batch of UT patches to their I/Q components
"""
from scipy.signal import hilbert
import numpy as np
from tqdm import tqdm

def envelope(volume):
    """
    Calculates the amplitude envelope of a 3D volume using Hilbert transform.
    
    This function applies the Hilbert transform along the last dimension of the input
    volume (which is assumed to be the signal propagation axis). It first converts the
    volume to int16 type, centers it by subtracting the mean, applies the Hilbert
    transform, and then calculates the absolute value to get the amplitude envelope.
    
    Parameters:
    ----------
    volume (numpy.ndarray): 
        3D array representing the volume of an image where the last dimension 
        is the signal propagation axis.
    
    Returns:
    -------
    numpy.ndarray: 
        Amplitude envelope of the Hilbert transform with the same shape as the input volume.
    
    Notes:
    -----
    The Hilbert transform is useful for extracting the envelope of oscillatory signals,
    which can highlight features like peaks in ultrasonic testing data.
    """

    # Convert to int16 to center in 0
    volume = volume.astype(np.int16)

    # Center the data by subtracting the mean to center in 0
    volume = volume - volume.mean()

    # Apply Hilbert transform along the last dimension (signal propagation axis)
    data_hilbert = hilbert(volume, axis=-1)
    
    # Calculate the amplitude envelope as the absolute value of the Hilbert transform
    amplitude_envelope = np.abs(data_hilbert)

    return amplitude_envelope

def align_signals_optimized(ut_patches, point=50):
    """
    Aligns the signals in UT patches so the frontwall peak is at a specified point.
    
    This function processes ultrasonic testing (UT) patches to align their frontwall 
    peaks at a specified position. The alignment is performed by:
    1. Calculating the amplitude envelope using the Hilbert transform
    2. Finding the position of the maximum amplitude (frontwall peak)
    3. Shifting each signal so that the peak aligns with the specified point
    
    The function processes the patches in batches to optimize memory usage.
    
    Parameters:
    ----------
    ut_patches (numpy.ndarray): 
        3D array of shape (n_samples, n_signals, signal_len) containing the UT patches.
        - n_samples: Number of samples/patches
        - n_signals: Number of signals per sample
        - signal_len: Length of each signal
    
    point (int, optional): 
        The index where the frontwall peak should be aligned. Default is 50.
    
    Returns:
    -------
    numpy.ndarray: 
        Aligned patches with the frontwall peak at the specified point.
        Same shape as the input ut_patches.
    
    Notes:
    -----
    This implementation uses batch processing to balance memory usage and
    processing speed, making it suitable for large datasets.
    """

    # Extract dimensions of input data
    n_samples, n_signals, signal_len = ut_patches.shape
    
    # Process in batches to balance memory usage and speed
    batch_size = min(100, n_samples)  # Adjust based on available memory
    aligned_patches = np.zeros_like(ut_patches)
    
    # Process each batch with progress bar
    for batch_idx in tqdm(range(0, n_samples, batch_size), desc="Aligning signal batches"):
        # Get current batch (handling the case where the last batch might be smaller)
        batch_end = min(batch_idx + batch_size, n_samples)
        batch = ut_patches[batch_idx:batch_end]
        
        # Reshape to process all signals in batch at once (flattening the first two dimensions)
        batch_flat = batch.reshape(-1, signal_len)
        
        # Calculate amplitude envelopes for all signals in the batch
        amp_envelopes = envelope(batch_flat)
        
        # Find frontwall positions (index of maximum amplitude for each signal)
        frontwall_positions = np.argmax(amp_envelopes, axis=1).reshape(batch.shape[0], n_signals)
        
        # Calculate shift amounts for each signal (how much each signal needs to be shifted)
        shifts = point - frontwall_positions
        
        # Apply the shifts to align the signals
        for i, sample_shifts in enumerate(shifts):
            for j, shift in enumerate(sample_shifts):
                # Use numpy's roll function to shift the signal
                aligned_patches[batch_idx + i, j] = np.roll(batch[i, j], shift)
    
    return aligned_patches

def convert_to_IQ(signal):
    """
    Convert a real-valued signal to its In-phase (I) and Quadrature (Q) components using the Hilbert transform.
    
    This function applies the Hilbert transform to obtain the analytic signal representation
    of the input real-valued signal. The analytic signal is a complex signal whose real part
    is the original signal (I component) and whose imaginary part is the Hilbert transform
    of the original signal (Q component).
    
    Parameters:
    ----------
    signal (numpy.ndarray): 
        The input real-valued signal. This can be a 1D array representing a single signal
        or a multi-dimensional array where the transform is applied along the last axis.
    
    Returns:
    -------
    I (numpy.ndarray): 
        The in-phase component of the signal (real part of the analytic signal),
        same shape as the input signal.
    Q (numpy.ndarray): 
        The quadrature component of the signal (imaginary part of the analytic signal),
        same shape as the input signal.
    
    Notes:
    -----
    The I/Q representation is commonly used in signal processing for demodulation, 
    analysis of signal phase and amplitude, and in various communication systems.
    """

    # Apply Hilbert transform to get the analytic signal (complex representation I + jQ)
    analytic_signal = hilbert(signal)
    
    # Extract the in-phase component (real part)
    I = np.real(analytic_signal)
    
    # Extract the quadrature component (imaginary part)
    Q = np.imag(analytic_signal)

    return I, Q

def get_IQ_patch(ut_patch):
    """
    Convert a UT (Ultrasonic Testing) patch to its I/Q components.
    
    This function takes an ultrasonic testing patch and converts it to its
    In-phase (I) and Quadrature (Q) components using the Hilbert transform.
    The resulting I and Q components are stacked along a new axis to create
    a representation suitable for further processing or analysis.
    
    Parameters:
    ----------
    ut_patch (numpy.ndarray): 
        The input UT patch. Can be a 1D array representing a single signal 
        or a multi-dimensional array where the transform is applied along the last axis.
    
    Returns:
    -------
    numpy.ndarray: 
        The I/Q components of the patch stacked along a new last axis.
        Shape is original_shape + (2,), where the last dimension contains
        I component at index 0 and Q component at index 1.
    
    Notes:
    -----
    This function uses the convert_to_IQ function internally to perform the
    I/Q decomposition. The I/Q representation provides both amplitude and phase
    information which can be valuable in ultrasonic signal analysis.
    """
    
    # Convert the patch to I/Q components using the Hilbert transform
    I, Q = convert_to_IQ(ut_patch)

    # Stack I and Q components along a new axis (last dimension)
    # This creates a single array where the last dimension has size 2:
    # iq_patch[..., 0] is I component and iq_patch[..., 1] is Q component
    iq_patch = np.stack((I, Q), axis=-1)

    return iq_patch

def get_IQ_patches(ut_patches):
    """
    Convert a set of UT (Ultrasonic Testing) patches to their I/Q components.
    
    This function processes a batch of ultrasonic testing patches, converting each
    to its In-phase (I) and Quadrature (Q) components. It first centers the data by 
    subtracting the mean value, then applies the I/Q conversion to each patch in the 
    input array.
    
    Parameters:
    ----------
    ut_patches (numpy.ndarray): 
        The input UT patches. Expected to be a batch of patches, typically with shape
        (n_patches, n_signals, signal_len) where n_patches is the number of patches and the remaining
        dimensions describe each patch.
    
    Returns:
    -------
    numpy.ndarray: 
        The I/Q components of all patches. The shape is the same as the input with
        an additional dimension of size 2 appended at the end for the I and Q components.
    
    Notes:
    -----
    This function first centers the data by subtracting the mean value across all patches,
    which can help reduce DC bias. It then processes each patch individually using the
    get_IQ_patch function and combines the results into a single array.
    """
    # Initialize an empty list to store the IQ patches
    iq_patches = []

    # Center the data by subtracting the mean to reduce DC bias
    ut_patches = ut_patches - ut_patches.mean()

    # Loop through each patch and convert it to I/Q components
    for ut_patch in ut_patches:
        # Process each patch individually
        iq_patch = get_IQ_patch(ut_patch)
        iq_patches.append(iq_patch)

    # Convert the list of IQ patches to a numpy array
    # This creates an array with shape (n_patches, ..., 2) where 2 is for I/Q components
    iq_patches = np.array(iq_patches)

    return iq_patches