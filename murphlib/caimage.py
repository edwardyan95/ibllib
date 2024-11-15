import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import tifffile
from scipy.signal import butter, filtfilt

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
    # Shift all fluorescence values to ensure they are above zero
    min_fluo = np.min(ca_imaging_data)
    if min_fluo <= 0:
        ca_imaging_data += (-min_fluo + 0.1)  # Shift fluorescence to slightly above zero

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
    # Shift all fluorescence values to ensure they are above zero
    min_fluo = np.min(ca_imaging_data, axis=1, keepdims=True)
    if np.any(min_fluo <= 0):
        ca_imaging_data += (-min_fluo + 0.1)  # Shift fluorescence to slightly above zero

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

