import json
import os
import numpy as np
from sklearn.decomposition import PCA

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