import json
import os
import numpy as np
from sklearn.decomposition import PCA
import scipy.io
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import tifffile
from pathlib import Path
def find_tif_file(directory_path):
    # List all files in the directory
    files = os.listdir(directory_path)
    
    # Filter for .tif files
    tif_files = [f for f in files if f.endswith('.tif')]
    
    # Check if there is only one .tif file
    if len(tif_files) == 1:
        # Return the full path to the .tif file
        return os.path.join(directory_path, tif_files[0])
    elif len(tif_files) == 0:
        return "No .tif files found in the directory."
    else:
        # Return a list of full paths if multiple .tif files are found
        return [os.path.join(directory_path, f) for f in tif_files]

def load_tif_file(tif_file):
    with tifffile.TiffFile(tif_file) as tif:
        substack = []
        for ind, page in enumerate(tif.pages):
            image = page.asarray()
            substack.append(image)
        return substack

def parse_text_to_dict(text):
    # Split the text into lines and initialize an empty dictionary
    lines = text.split('\n')
    config_dict = {}
    json_str = ''
    json_started = False

    # Process each line
    for line in lines:
        # Handle JSON-like structures
        if line.strip().startswith('{'):
            json_started = True

        if json_started:
            json_str += line + '\n'
            if line.strip().endswith('}'):
                try:
                    json_dict = json.loads(json_str)
                    config_dict.update(json_dict)
                    json_started = False
                    json_str = ''
                except json.JSONDecodeError:
                    pass  # Continue accumulating lines for JSON
            continue

        # Process as key-value pairs
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip("'")

            # Convert boolean and numeric values from string
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.replace('.', '', 1).isdigit():
                value = float(value) if '.' in value else int(value)

            config_dict[key] = value

    return config_dict

def zscore(arr, axis=0):
    """
    Computes the Z-score of each value in the array along a specified axis.
    
    Parameters:
    - arr: Input array (NumPy array).
    - axis: The axis along which to compute the Z-score. Default is 0.
    
    Returns:
    - Z-score array.
    """
    # Calculate the mean along the specified axis
    mean = np.mean(arr, axis=axis, keepdims=True)
    
    # Calculate the standard deviation along the specified axis
    std = np.std(arr, axis=axis, ddof=0, keepdims=True)
    
    # Compute the Z-scores
    z_scores = (arr - mean) / std
    
    return z_scores

def downsample_array(arr, desired_length, axis=0, method='simple'):
    """
    Downsamples a NumPy array along a specified axis to a desired length.
    
    Parameters:
    - arr: The input array (NumPy array).
    - desired_length: The target length after downsampling.
    - axis: The axis along which to downsample.
    - method: 'simple' for simple downsampling, 'interpolate' for linear interpolation.
    
    Returns:
    - downsampled array.
    """
    if method not in ['simple', 'interpolate']:
        raise ValueError("Method must be either 'simple' or 'interpolate'.")

    # Get the shape of the input array
    original_length = arr.shape[axis]

    if method == 'simple':
        # Calculate the step size for downsampling
        step_size = original_length // desired_length

        # Generate indices for simple downsampling
        indices = np.arange(0, original_length, step_size)
        
        # If too many indices, truncate to desired length
        indices = indices[:desired_length]

        # Use np.take along the specified axis
        downsampled = np.take(arr, indices, axis=axis)

    elif method == 'interpolate':
        # Generate the original indices
        original_indices = np.linspace(0, original_length - 1, num=original_length)
        
        # Generate the new indices for the desired length
        new_indices = np.linspace(0, original_length - 1, num=desired_length)

        # Perform interpolation along the specified axis
        downsampled = np.apply_along_axis(
            lambda x: np.interp(new_indices, original_indices, x), 
            axis, 
            arr
        )
    
    return downsampled

def perform_pca(df_f_data, n_components=5):
    """
    Perform PCA on df/f traces and return the top components, explained variance, and scores.

    Parameters:
    df_f_data (numpy array): Matrix of shape (num_neurons, timepoints) representing df/f traces.
    n_components (int): Number of top PCA components to return.

    Returns:
    pca_components (numpy array): Principal components of shape (n_components, num_neurons).
    explained_variance (numpy array): Explained variance for each component.
    pca_scores (numpy array): Projection of the original data onto the principal components.
    """
    # Initialize PCA model
    pca = PCA(n_components=n_components)

    # Fit PCA on the data (transpose to shape (timepoints, num_neurons) for PCA)
    pca.fit(df_f_data.T)

    # Get the principal components (shape will be (n_components, num_neurons))
    pca_components = pca.components_

    # Get the explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    # Project the original data onto the principal components (scores)
    pca_scores = pca.transform(df_f_data.T)

    return pca_components.T, explained_variance, pca_scores.T

def fill_nan_with_previous(arr):
    """
    Fill NaN values in the array with the immediate previous non-NaN value.
    
    Parameters:
    arr (np.ndarray): Input array (1D or 2D) with NaN values.
    
    Returns:
    np.ndarray: Array with NaN values filled.
    """
    if arr.ndim == 1:  # For 1D arrays
        mask = np.isnan(arr)
        # Fill NaN with the last valid value
        for i in range(1, len(arr)):
            if mask[i]:
                arr[i] = arr[i - 1]
    elif arr.ndim == 2:  # For 2D arrays
        # Process each row independently
        for row in range(arr.shape[0]):
            mask = np.isnan(arr[row])
            for i in range(1, arr.shape[1]):
                if mask[i]:
                    arr[row, i] = arr[row, i - 1]
    else:
        raise ValueError("Input array must be 1D or 2D.")
    
    return arr

def get_valid_suite2p_stats(path):
    """
    from a suite2p path: ...\\suite2p\\plane0, load stat.npy
    pick cells that are valid cells, attach suite2p id
    return the stats
    """
    stat = np.load(os.path.join(path, 'stat.npy'), allow_pickle=True)
    iscell = np.load(path+'\\iscell.npy', allow_pickle=True)
    iscell = iscell[:,0]==1
    new_stat = []
    for i, s in enumerate(stat):
        if iscell[i]:
            s['id'] = i
            new_stat.append(s)
    return new_stat

def get_stat_with_coord(path):
    suite2p_path = os.path.join(path, 'suite2p', 'plane0')
    xcoord_atlas = scipy.io.loadmat(os.path.join(path, 'tform_xcoord_atlas.mat'))['tform_xcoord_atlas'][0]
    ycoord_atlas = scipy.io.loadmat(os.path.join(path, 'tform_ycoord_atlas.mat'))['tform_ycoord_atlas'][0]
    xcoord_lin = scipy.io.loadmat(os.path.join(path, 'tform_xcoord_lin.mat'))['tform_xcoord_lin'][0]
    ycoord_lin = scipy.io.loadmat(os.path.join(path, 'tform_ycoord_lin.mat'))['tform_ycoord_lin'][0]
    stat = get_valid_suite2p_stats(os.path.join(suite2p_path))
    aspect_ratio = np.array([stat[i]['aspect_ratio'] for i in range(len(stat))])
    npix = np.array([stat[i]['npix'] for i in range(len(stat))])
    dd_idx = (aspect_ratio > 1.1) | (npix > 120) | (npix < 100)
    pc_idx = (aspect_ratio<1.1) & (npix<120) & (npix>100)
    xcoord_atlas = np.array([sub_arr.flatten() for sub_arr in xcoord_atlas if sub_arr.size > 0], dtype=object)
    ycoord_atlas = np.array([sub_arr.flatten() for sub_arr in ycoord_atlas if sub_arr.size > 0], dtype=object)
    xcoord_lin = np.array([sub_arr.flatten() for sub_arr in xcoord_lin if sub_arr.size > 0], dtype=object)
    ycoord_lin = np.array([sub_arr.flatten() for sub_arr in ycoord_lin if sub_arr.size > 0], dtype=object)
    for i, s in enumerate(stat):
        s['xcoord_atlas'] = xcoord_atlas[i]
        s['ycoord_atlas'] = ycoord_atlas[i]
        s['xcoord_lin'] = xcoord_lin[i]
        s['ycoord_lin'] = ycoord_lin[i]
        s['celltype'] = 'dendrite' if dd_idx[i] else 'soma'
        
    return stat

def attach_reg_model_to_stat(stat, model):
    
    assert(len(stat)==len(model))
    for i,s in enumerate(stat):
        s['beta'] = model[i]['beta']
        s['intercepts'] = model[i]['intercepts']
        try:
            s['explained_variance'] = model[i]['explained_variance']
        except:
            continue
        try:
            s['unique_explained_variance'] = model[i]['unique_explained_variance']
        except:
            continue
        try:
            s['f_stat'] = model[i]['f_stat']
        except:
            continue
        try:
            s['bootstrap_p_value'] = model[i]['bootstrap_p_value']
        except:
            continue
        try:
            s['bootstrap_f_stat'] = model[i]['bootstrap_f_stat']
        except:
            continue
    return stat

def evaluate_clusters(data, max_clusters=10):
    silhouette_scores = []
    calinski_harabasz_scores = []
    wcss = []  # Within-cluster sum of squares

    # Try different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        
        # Calculate metrics
        silhouette_scores.append(silhouette_score(data, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(data, labels))
        wcss.append(kmeans.inertia_)
    
    return silhouette_scores, calinski_harabasz_scores, wcss

def holm_bonferroni_correction(p_values):
    """
    Apply Holm-Bonferroni correction across tests
    
    Parameters:
    p_values: array of shape (num_tests,) for a single neuron
              or shape (num_neurons, num_tests) for multiple neurons
    
    Returns:
    corrected p-values of same shape as input
    """
    # Check if input is 1D or 2D
    input_is_1d = p_values.ndim == 1
    
    # If 1D, convert to 2D temporarily
    if input_is_1d:
        p_values = p_values.reshape(1, -1)
    
    num_neurons, num_tests = p_values.shape
    p_corrected = np.zeros_like(p_values)
    
    # Apply correction separately for each neuron
    for i in range(num_neurons):
        # Get p-values for this neuron
        p_neuron = p_values[i]
        
        # Get sorting indices and ranks
        sorted_indices = np.argsort(p_neuron)
        ranks = np.argsort(sorted_indices)  # to map back to original order
        
        # Apply Holm's correction
        p_sorted = p_neuron[sorted_indices]
        p_corrected_sorted = np.minimum(1, p_sorted * (num_tests - np.arange(num_tests)))
        
        # Ensure monotonicity (each element should be >= previous)
        for j in range(1, len(p_corrected_sorted)):
            p_corrected_sorted[j] = max(p_corrected_sorted[j], p_corrected_sorted[j-1])
        
        # Map back to original order
        p_corrected[i] = p_corrected_sorted[ranks]
    
    # If input was 1D, convert output back to 1D
    if input_is_1d:
        p_corrected = p_corrected.flatten()
    
    return p_corrected

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

def detect_licking_events(motion_energy, threshold=2.0, distance=5, detect_negative=True):
    """
    Detects discrete licking events from a motion energy signal by finding peaks
    in the z-scored signal.

    This function first z-scores the motion energy signal, then identifies peaks
    that are above/below a certain threshold (in standard deviations) and separated by a
    minimum distance.

    Parameters:
    motion_energy (np.ndarray): 1D array representing motion energy over time.
    threshold (float): The z-score threshold for detecting a peak. Only peaks with
                       a z-score higher/lower than this value will be considered.
                       Default is 2.0 (2 standard deviations above/below the mean).
    distance (int): The minimum required horizontal distance (in frames/samples)
                    between neighboring peaks. Default is 5 frames.
    detect_negative (bool): Whether to also detect negative peaks. Default is True.

    Returns:
    np.ndarray: An array of indices (timepoints/frames) where licking events are detected,
                sorted in chronological order.
    """
    
    # Z-score the motion energy signal
    z_scored_energy = (motion_energy - np.mean(motion_energy)) / np.std(motion_energy)
    
    # Find positive peaks
    positive_peaks, _ = scipy.signal.find_peaks(z_scored_energy, height=threshold, distance=distance)
    
    if detect_negative:
        # Find negative peaks by inverting the signal and finding peaks
        negative_peaks, _ = scipy.signal.find_peaks(-z_scored_energy, height=threshold, distance=distance)
        
        # Combine positive and negative peaks
        all_peaks = np.concatenate([positive_peaks, negative_peaks])
        
        # Sort chronologically
        all_peaks = np.sort(all_peaks)
        
        return all_peaks
    else:
        return positive_peaks

def find_first_event_after(event_times, target_times):
    """
    For each target time, finds the timestamp of the first event that occurs at or after it.

    Parameters:
    event_times (np.ndarray): A sorted 1D array of event timestamps.
    target_times (np.ndarray): A 1D array of target timestamps to search from.

    Returns:
    np.ndarray: An array of the same shape as target_times, containing the timestamp
                of the first corresponding event. If no event is found after a
                target time, the value is np.nan.
    """
    # Find the insertion indices for each target time in the event_times array.
    # This gives us the index of the first event >= the target time.
    indices = np.searchsorted(event_times, target_times, side='left')

    # Create an output array filled with NaNs by default.
    result_times = np.full(target_times.shape, np.nan)

    # Identify valid indices (i.e., not pointing past the end of event_times).
    valid_mask = indices < len(event_times)

    # For valid indices, get the corresponding event time from the original array.
    result_times[valid_mask] = event_times[indices[valid_mask]]

    return result_times

import math

def grid_shape(n):
    # rows*cols >= n and rows ≈ cols
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def calculate_lick_rate(lick_times_behavior, video_frame_rate, bin_size_s=None, n_bins=None, total_duration_s=None, total_frames=None):
    """
    Convert lick frame times to lick rate (Hz) by binning.
    
    Parameters:
    -----------
    lick_times_behavior : array-like
        Frame indices where lick events were detected
    video_frame_rate : float
        Frame rate of the behavior video (frames per second)
    bin_size_s : float, optional
        Size of each bin in seconds (mutually exclusive with n_bins)
    n_bins : int, optional
        Number of bins to create (mutually exclusive with bin_size_s)
    total_duration_s : float, optional
        Total duration of the session in seconds
    total_frames : int, optional
        Total number of frames in the video (alternative to total_duration_s)
        
    Returns:
    --------
    lick_rate : array
        Lick rate in Hz for each time bin
    bin_centers : array
        Time points (in seconds) for the center of each bin
    bin_edges : array
        Time points (in seconds) for the edges of each bin
    """
    import numpy as np
    
    # Check that exactly one of bin_size_s or n_bins is provided
    if (bin_size_s is None and n_bins is None) or (bin_size_s is not None and n_bins is not None):
        raise ValueError("Must specify exactly one of 'bin_size_s' or 'n_bins'")
    
    # Convert lick frames to times in seconds
    lick_times_s = np.array(lick_times_behavior) / video_frame_rate
    
    # Determine total duration
    if total_duration_s is not None:
        duration = total_duration_s
    elif total_frames is not None:
        duration = total_frames / video_frame_rate
    else:
        # Use the last lick time plus some buffer
        if n_bins is not None:
            duration = np.max(lick_times_s) * 1.1  # Add 10% buffer
        else:
            duration = np.max(lick_times_s) + bin_size_s
    
    # Create time bins
    if n_bins is not None:
        # Create specified number of bins
        bin_edges = np.linspace(0, duration, n_bins + 1)
        bin_size_s = duration / n_bins
    else:
        # Use specified bin size
        bin_edges = np.arange(0, duration + bin_size_s, bin_size_s)
    
    bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    
    # Count licks in each bin
    lick_counts, _ = np.histogram(lick_times_s, bins=bin_edges)
    
    # Convert counts to rate (Hz)
    lick_rate = lick_counts / bin_size_s
    
    return lick_rate, bin_centers, bin_edges

def is_under(file_path, dir_path):
    """
    Check if file_path exists and is located under dir_path.
    
    Args:
        file_path (str or Path): Path to file
        dir_path (str or Path): Path to directory

    Returns:
        bool
    """
    file_path = Path(file_path).resolve()
    dir_path = Path(dir_path).resolve()

    try:
        # Check both existence and ancestry
        return file_path.exists() and dir_path in file_path.parents
    except RuntimeError:
        # In rare cases (bad symlinks, permissions), just fail safe
        return False