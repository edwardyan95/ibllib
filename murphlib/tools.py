import json
import os
import numpy as np
from sklearn.decomposition import PCA
import scipy.io
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

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
        return "Multiple .tif files found in the directory."

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
    xcoord_atlas = np.array([sub_arr.flatten() for sub_arr in xcoord_atlas if sub_arr.size > 0], dtype=object)
    ycoord_atlas = np.array([sub_arr.flatten() for sub_arr in ycoord_atlas if sub_arr.size > 0], dtype=object)
    xcoord_lin = np.array([sub_arr.flatten() for sub_arr in xcoord_lin if sub_arr.size > 0], dtype=object)
    ycoord_lin = np.array([sub_arr.flatten() for sub_arr in ycoord_lin if sub_arr.size > 0], dtype=object)
    for i, s in enumerate(stat):
        s['xcoord_atlas'] = xcoord_atlas[i]
        s['ycoord_atlas'] = ycoord_atlas[i]
        s['xcoord_lin'] = xcoord_lin[i]
        s['ycoord_lin'] = ycoord_lin[i]
    return stat

def attach_reg_model_to_stat(stat, model):
    assert(len(stat)==len(model))
    for i,s in enumerate(stat):
        s['beta'] = model[i]['beta']
        s['intercepts'] = model[i]['intercepts']
        s['explained_variance'] = model[i]['explained_variance']
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
