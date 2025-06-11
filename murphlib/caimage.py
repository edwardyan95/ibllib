import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import tifffile
from scipy.signal import butter, filtfilt

from scipy.ndimage import percentile_filter

def find_frame_indices(dataframe, frame_rate, time_window, column_keyword):
    # Step 1: Identify the correct column
    for col in dataframe.columns:
        if column_keyword in col:
            target_column = col
            break
    else:  # If no column contains the keyword
        raise ValueError(f"No column found containing the keyword '{column_keyword}'")
    
    # Step 2: Calculate frame indices for each element
    indices_list = []
    for frame_number in dataframe[target_column]:
        if np.isnan(frame_number):
            continue
        # Calculate the number of frames for the given time window
        frames_before = int(round(frame_rate * time_window[0]))
        frames_after = int(round(frame_rate * time_window[1]))
        
        # Calculate the start and end frame
        start_frame = frame_number + frames_before
        end_frame = frame_number + frames_after
        
        # Generate the frame indices and add to the list
        indices = np.arange(start_frame, end_frame + 1)  # +1 because end is exclusive in np.arange
        indices_list.append(indices)
    
    return np.array(indices_list)

def get_trial_PSTH(ca_imaging_data, frame_idx, zscore=True):
    # ca_imaging_data is a 2D NumPy array with shape (num_cells, frames)
    # frame_idx is 2D numpy array with shape (trials, num frames to be extracted for a trial(center frame is when event happens))
    # return zscored psth (num_trial, num_cells, num_frames_extracted)
    num_trials, num_frames_extracted = frame_idx.shape
    num_cells = ca_imaging_data.shape[0]
    psth = np.zeros((num_trials, num_cells, num_frames_extracted))
    z_psth = np.zeros_like(psth)  # Array to hold Z-score normalized PSTH

    for trial in range(num_trials):
        idx = frame_idx[trial, :]
        trial_psth = ca_imaging_data[:, idx]
        psth[trial, :, :] = trial_psth
        if zscore:
            # Calculate mean and std for Z-score normalization, across frames for each cell
            mean_psth = np.mean(trial_psth, axis=1, keepdims=True)
            std_psth = np.std(trial_psth, axis=1, keepdims=True)
            
            # Avoid division by zero by setting std to 1 where it's 0 (or very close to 0)
            std_psth[std_psth == 0] = 1
            
            # Calculate Z-score
            z_trial_psth = (trial_psth - mean_psth) / std_psth
            
            # Store the Z-score normalized data
            z_psth[trial, :, :] = z_trial_psth

    #return psth
    if zscore:
        return z_psth
    else:
        return psth

def sort_psth_by_average_response(mean_psth, post, frame_rate):
    num_cells, num_frames = mean_psth.shape
    midpoint = num_frames//2
    window_start = midpoint
    window_end = window_start + int(post*frame_rate)

    # Calculate average response in the defined window
    average_responses = np.mean(mean_psth[:, window_start:window_end], axis=1)

    # Get sorted indices, from highest to lowest average response
    sorted_indices_average = np.argsort(-average_responses)

    # Sort the PSTH array
    sorted_psth_average = mean_psth[sorted_indices_average]
    
    return sorted_indices_average, sorted_psth_average



def load_tiff(filepath):
    # Open the TIFF file
    with tifffile.TiffFile(filepath) as tif:
        # Initialize an empty list to hold the data from each page
        pages = []
        
        # Iterate over each page in the TIFF file
        for page in tif.pages:
            # Read the page into a NumPy array and append it to the list
            pages.append(page.asarray())
        
        # Stack the pages along a new first dimension
        tif_array = np.stack(pages, axis=0)
        
    return tif_array

def calculate_dff_with_moving_median(ca_imaging_data, frame_rate):
    # ca_imaging_data is a 2D NumPy array with shape (num_cells, frames)
    # frame_rate is the number of frames per second
    # Shift all fluoescence values to ensure they are above zero
    min_fluo = np.min(ca_imaging_data)
    if min_fluo <= 0:
        ca_imaging_data += (-min_fluo + 0.1)  # Shift fluoescence to slightly above zero

    window_size = int(20 * frame_rate)  # 20 seconds window
    num_cells, frames = ca_imaging_data.shape
    dff = np.zeros_like(ca_imaging_data)

    # Apply median filter to the entire dataset for each cell
    median_filtered = np.zeros_like(ca_imaging_data)
    for cell in range(num_cells):
        # Note: medfilt with kernel_size applied to 1D array per cell
        median_filtered[cell, :] = scipy.signal.medfilt(ca_imaging_data[cell, :], kernel_size=window_size)

    # Calculate ΔF/F for each cell using the filtered data as baseline
    for cell in range(num_cells):
        # Avoid division by zero
        baseline = median_filtered[cell, :]
        baseline[baseline == 0] = np.min(baseline[baseline > 0])  # replace 0 with the smallest non-zero baseline value

        dff[cell, :] = (ca_imaging_data[cell, :] - baseline) / baseline

    return dff

def calculate_dff_with_percentile(ca_imaging_data, percentile):
    # ca_imaging_data is a 2D NumPy array with shape (cell#, frames)
    # Shift all fluoescence values to ensure they are above zero
    min_fluo = np.min(ca_imaging_data, axis=1, keepdims=True)
    if np.any(min_fluo <= 0):
        ca_imaging_data += (-min_fluo + 0.1)  # Shift fluoescence to slightly above zero

    # Calculate baseline using the specified percentile
    baseline = np.percentile(ca_imaging_data, percentile, axis=1, keepdims=True)
    
    # Calculate df/f
    dff = (ca_imaging_data - baseline) / baseline
   
    return dff


def calculate_dff_with_moving_percentile(ca_imaging_data, frame_rate, moving_window=30, percentile=15):
    num_cells, num_frames = ca_imaging_data.shape
    
    # Shift all fluorescence values to ensure they are above zero
    min_fluo = np.min(ca_imaging_data)
    if min_fluo <= 0:
        ca_imaging_data += (-min_fluo + 0.1)  # Shift fluorescence to slightly above zero

    # Calculate the window length in frames
    window_length = int(frame_rate * moving_window)
    
    # Initialize the percentile filtered baseline array
    percentile_filtered = np.zeros((num_cells, num_frames))
    
    # Calculate the moving percentile
    for cell in range(num_cells):
        for frame in range(num_frames):
            start = max(0, frame - window_length // 2)
            end = min(num_frames, frame + window_length // 2)
            percentile_filtered[cell, frame] = np.percentile(ca_imaging_data[cell, start:end], percentile)
    
    # Calculate dF/F
    dff = np.zeros_like(ca_imaging_data)
    for cell in range(num_cells):
        baseline = percentile_filtered[cell, :]
        # Replace zero baselines with the smallest non-zero baseline value to prevent division by zero
        baseline[baseline == 0] = np.min(baseline[baseline > 0]) if np.any(baseline > 0) else np.min(baseline) + 0.1
        dff[cell, :] = (ca_imaging_data[cell, :] - baseline) / baseline

    return dff

def calculate_dff_with_moving_percentile_vectorized(ca_imaging_data, frame_rate, moving_window=30, percentile=15):
    """
    Vectorized version of calculate_dff_with_moving_percentile using scipy.ndimage.percentile_filter
    for much faster performance.
    
    Parameters:
    ca_imaging_data (np.ndarray): 2D array of calcium traces with shape (num_cells, num_frames).
    frame_rate (float): The frame rate (sampling frequency) of the data (in Hz).
    moving_window (float): Window size in seconds for percentile calculation.
    percentile (float): Percentile value to use as baseline (0-100).
    
    Returns:
    np.ndarray: dF/F values with same shape as input.
    """
    num_cells, num_frames = ca_imaging_data.shape
    
    # Shift all fluorescence values to ensure they are above zero
    min_fluo = np.min(ca_imaging_data)
    if min_fluo <= 0:
        ca_imaging_data = ca_imaging_data + (-min_fluo + 0.1)  # Shift fluorescence to slightly above zero
    
    # Calculate the window length in frames (ensure it's odd for centered window)
    window_length = int(frame_rate * moving_window)
    if window_length % 2 == 0:
        window_length += 1
    
    # Pad data to handle edge effects
    pad_width = window_length // 2
    padded_data = np.pad(ca_imaging_data, ((0, 0), (pad_width, pad_width)), mode='reflect')
    
    # Calculate the moving percentile for all cells using percentile_filter
    percentile_filtered = np.zeros_like(ca_imaging_data)
    for cell in range(num_cells):
        percentile_filtered[cell] = percentile_filter(
            padded_data[cell], 
            percentile=percentile, 
            size=window_length,
            mode='constant'
        )[pad_width:pad_width+num_frames]  # Remove padding
    
    # Handle zero or very small baselines to prevent division issues
    percentile_filtered[percentile_filtered < 1e-6] = 1e-6
    
    # Calculate dF/F in one vectorized operation
    dff = (ca_imaging_data - percentile_filtered) / percentile_filtered
    
    return dff

def butter_filter(data, cutoff, fs, filter_type='low', order=5):
    """
    Apply a Butterworth filter to calcium traces.
    
    Parameters:
    data (np.ndarray): 2D array of calcium traces with shape (num_neurons, num_timepoints).
    cutoff (float): The cutoff frequency for the filter (in Hz).
    fs (float): The frame rate (sampling frequency) of the data (in Hz).
    filter_type (str): 'low' for low-pass, 'high' for high-pass. Default is 'low'.
    order (int): The order of the filter. Default is 5.
    
    Returns:
    np.ndarray: Filtered calcium traces of the same shape as the input.
    """
    # Normalize the frequency
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency

    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    
    # Apply the filter along the timepoints (axis=1 for 2D array)
    filtered_data = filtfilt(b, a, data, axis=1)
    
    return filtered_data

def calculate_zstack_fluo(neuron_stats, zstack_mean):
    """
    Calculates the mean fluorescence of each neuron in each z-plane.

    Parameters:
    neuron_stats (list of dict): A list where each dict contains neuron information,
                                 including 'xpix' and 'ypix' for pixel coordinates.
    zstack_mean (np.ndarray): A 3D array of shape (num_planes, num_ypixels_image, num_xpixels_image)
                              representing the mean image of each z-plane.

    Returns:
    np.ndarray: An array of shape (num_neurons, num_planes) containing the mean
                fluorescence of each neuron in each z-plane.
    """
    num_neurons = len(neuron_stats)
    num_planes = zstack_mean.shape[0]

    zstack_fluo = np.zeros((num_neurons, num_planes))

    for i in range(num_neurons):
        neuron_ypix = neuron_stats[i]['ypix'] # y-coordinates are typically rows
        neuron_xpix = neuron_stats[i]['xpix'] # x-coordinates are typically columns

        for j in range(num_planes):
            plane_image = zstack_mean[j, :, :] 
            # Extract fluorescence values for the neuron's pixels in the current plane
            # Ensure coordinates are within image bounds
            # Note: In image indexing, y comes before x: image[y, x]
            neuron_pixel_fluorescence = plane_image[neuron_ypix, neuron_xpix]
            
            # Calculate the mean fluorescence for the neuron in this plane
            if neuron_pixel_fluorescence.size > 0:
                zstack_fluo[i, j] = np.mean(neuron_pixel_fluorescence)
            else:
                zstack_fluo[i, j] = np.nan # Or 0, if preferred for empty pixel sets
                
    return zstack_fluo


def calculate_integrated_zshift_dff_vectorized(raw_fluo, zstack_fluo, frame_zplanes, frame_rate, temporal_window=30, percentile=15):
    """
    Vectorized calculation of dF/F with integrated z-shift correction and temporal baseline.
    
    Parameters:
    raw_fluo: array of shape (num_cells, num_frames) - Raw fluoescence values
    zstack_fluo: array of shape (num_cells, num_zplanes) - Baseline fluoescence at each z-plane
    frame_zplanes: array of shape (num_frames,) - Z-plane index for each frame
    frame_rate: Imaging frame rate (Hz)
    temporal_window: Window size in seconds for temporal baseline calculation
    percentile: Percentile for temporal baseline calculation
    
    Returns:
    final_dff: dF/F values corrected for z-shift and temporal baseline
    """
    num_cells, num_frames = raw_fluo.shape
    
    # Step 1: Create a reference z-plane baseline for each cell (using middle z-plane)
    mid_zplane = zstack_fluo.shape[1] // 2
    reference_baseline = zstack_fluo[:, mid_zplane].reshape(-1, 1)  # Shape: (num_cells, 1)
    
    # Step 2: Calculate z-plane correction factors for all frames at once
    # Extract baseline values for each cell at each frame's z-plane
    frame_baselines = zstack_fluo[:, frame_zplanes]  # Shape: (num_cells, num_frames)
    
    # Calculate correction factors (ratio of each frame's z-plane baseline to reference baseline)
    z_correction_factors = frame_baselines / reference_baseline  # Broadcasting handles this
    
    # Handle division by zero or very small values
    z_correction_factors[~np.isfinite(z_correction_factors) | (z_correction_factors < 1e-6)] = 1.0
    
    # Step 3: Apply z-correction to raw fluoescence
    z_corrected_fluo = raw_fluo / z_correction_factors
    
    # Step 4: Calculate temporal baseline using sliding window
    window_size_frames = int(frame_rate * temporal_window)
    
    # Calculate temporal baseline using percentile filter (much faster than looping)
    # Pad the data to handle edge effects
    pad_width = window_size_frames // 2
    padded_data = np.pad(z_corrected_fluo, ((0, 0), (pad_width, pad_width)), mode='reflect')
    
    # Apply percentile filter to each cell's time series
    temp_baseline = np.zeros_like(z_corrected_fluo)
    for cell in range(num_cells):
        # Use percentile filter for each cell (scipy's percentile_filter is faster than manual window)
        temp_baseline[cell] = percentile_filter(
            padded_data[cell], 
            percentile=percentile, 
            size=window_size_frames,
            mode='constant'
        )[pad_width:pad_width+num_frames]  # Remove padding
    
    # Ensure no zero baselines
    temp_baseline[temp_baseline <= 0] = 0.1
    
    # Step 5: Calculate final dF/F
    final_dff = (z_corrected_fluo - temp_baseline) / temp_baseline
    
    return final_dff



def calculate_zshift_corrected_fluo(raw_fluo, zstack_fluo, frame_zplanes):
    """
    Vectorized calculation of dF/F with integrated z-shift correction and temporal baseline.
    
    Parameters:
    raw_fluo: array of shape (num_cells, num_frames) - Raw fluoescence values
    zstack_fluo: array of shape (num_cells, num_zplanes) - Baseline fluoescence at each z-plane
    frame_zplanes: array of shape (num_frames,) - Z-plane index for each frame
    frame_rate: Imaging frame rate (Hz)
    temporal_window: Window size in seconds for temporal baseline calculation
    percentile: Percentile for temporal baseline calculation
    
    Returns:
    final_dff: dF/F values corrected for z-shift and temporal baseline
    """
    num_cells, num_frames = raw_fluo.shape
    
    # Step 1: Create a reference z-plane baseline for each cell (using middle z-plane)
    mid_zplane = zstack_fluo.shape[1] // 2
    reference_baseline = zstack_fluo[:, mid_zplane].reshape(-1, 1)  # Shape: (num_cells, 1)
    
    # Step 2: Calculate z-plane correction factors for all frames at once
    # Extract baseline values for each cell at each frame's z-plane
    frame_baselines = zstack_fluo[:, frame_zplanes]  # Shape: (num_cells, num_frames)
    
    # Calculate correction factors (ratio of each frame's z-plane baseline to reference baseline)
    z_correction_factors = frame_baselines / reference_baseline  # Broadcasting handles this
    
    # Handle division by zero or very small values
    z_correction_factors[~np.isfinite(z_correction_factors) | (z_correction_factors < 1e-6)] = 1.0
    
    # Step 3: Apply z-correction to raw fluoescence
    z_corrected_fluo = raw_fluo / z_correction_factors

    return z_corrected_fluo

