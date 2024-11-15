import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
def generate_event_windows(df, column_names, event_type):
    """
    Generates a list of NumPy arrays where 0 indicates no event and 1 indicates event occurrence in the specified window.
    One array per event type, combining across all trials.

    Parameters:
    df (pd.DataFrame): Input DataFrame where rows are trials and columns specify frame indices of events.
    column_names (list of str): List of column names for which to generate output arrays.
    event_types (list of str): List of event types ('whole', 'peri', 'post') corresponding to column names.
    frame_rate (float): Frame rate to convert time (in seconds) to frames.

    Returns:
    list of np.ndarray: List of NumPy arrays where each array represents the event window for all trials combined.
    """
    
    # List to hold output arrays for all events combined across trials
    output_arrays = []
    
    # Get the length for the output arrays from the last row of 'intervals_1'
    output_length = int(df.loc[df.index[-1], 'intervals_1'])
    
    # Loop through each event column and type
    for col in column_names:
        # Initialize a single array for this event (combined across trials)
        event_array = np.zeros(output_length)
        
        # Loop through each trial in the DataFrame
        for i, row in df.iterrows():
            if pd.isna(row[col]):
                # If the event_frame is NaN, skip this row
                continue
            event_frame = row[col]
            intervals_start = row['intervals_0']
            intervals_end = row['intervals_1']
            if i==len(df)-1:
                next_intervals_start = intervals_end
            else:
                next_intervals_start = df.loc[i+1,'intervals_0']
            
            
            
            if event_type == 'whole':
                # Set frames from intervals_0 to intervals_1 to event_frame
                if col == 'reward_history':
                    event_array[int(intervals_start):int(next_intervals_start)] = event_frame
                else:   
                    event_array[int(intervals_start):int(intervals_end)] = event_frame
            
            elif event_type == 'peri' or event_type == 'post':
                event_array[int(event_frame)] = 1
                
        
        # Append this event array (combined across trials) to the output list
        output_arrays.append(event_array)
    
    return output_arrays


def transform_trial_table(df):
    """
    Transforms the input trial DataFrame by adding columns for left/right choices, reward/punish times,
    and trial modality (visual/auditory).

    Parameters:
    df (pd.DataFrame): Input trial DataFrame with 'choice', 'feedbackType', 'feedback_times', and 'modality' columns.

    Returns:
    pd.DataFrame: Transformed DataFrame with added columns.
    """
    df_transformed = df.copy()
    # Add 'leftchoice' column: 1 when 'choice' is -1, NaN otherwise
    df_transformed['leftchoice'] = np.where(df_transformed['choice'] == -1, 1, 0)

    # Add 'rightchoice' column: 1 when 'choice' is 1, NaN otherwise
    df_transformed['rightchoice'] = np.where(df_transformed['choice'] == 1, 1, 0)

    # Add 'reward_times' column: 'feedback_times' where 'feedbackType' is 1, NaN otherwise
    df_transformed['reward_times'] = np.where(df_transformed['feedbackType'] == 1, df_transformed['feedback_times'], np.nan)

    # Add 'punish_times' column: 'feedback_times' where 'feedbackType' is -1, NaN otherwise
    df_transformed['punish_times'] = np.where(df_transformed['feedbackType'] == -1, df_transformed['feedback_times'], np.nan)

    # Add 'visual_trial' column: 1 where 'modality' is 0, 0 elsewhere
    df_transformed['visual_trial'] = np.where(df_transformed['modality'] == 0, 1, 0)

    # Add 'aud_trial' column: 1 where 'modality' is 1, 0 elsewhere
    df_transformed['aud_trial'] = np.where(df_transformed['modality'] == 1, 1, 0)

    # Add 'vis_stimOn' column: 'stimOnTrigger_times' where 'modality' is 0, NaN otherwise
    df_transformed['left_vis_stimOn_times'] = np.where(df_transformed['contrastLeft'] > 0, df_transformed['vis_stimOn_times'], np.nan)
    df_transformed['right_vis_stimOn_times'] = np.where(df_transformed['contrastRight'] > 0, df_transformed['vis_stimOn_times'], np.nan)
    # Add 'aud_stimOn' column: 'stimOnTrigger_times' where 'modality' is 1, NaN otherwise
    df_transformed['left_aud_stimOn_times'] = np.where(df_transformed['contrastLeft'] == 0.0, df_transformed['aud_stimOn_times'], np.nan)
    df_transformed['right_aud_stimOn_times'] = np.where(df_transformed['contrastRight'] == 0.0, df_transformed['aud_stimOn_times'], np.nan)

    # Add 'previous_feedbackType' column
    previous_feedbackType = np.roll(np.array(df_transformed['feedbackType']),1)
    previous_feedbackType[0] = 0
    df_transformed['previous_feedbackType'] = previous_feedbackType

    reward_history_window = 20 # rolling window of 20
    feedbackType = df_transformed['feedbackType'].copy()
    feedbackType.iloc[feedbackType==-1]=0
    reward_history = feedbackType.rolling(window=reward_history_window).mean().to_numpy()
    reward_history[:reward_history_window] = reward_history[reward_history_window]
    df_transformed['reward_history'] = reward_history

    return df_transformed




def generate_spline_basis_functions(num_basis, window_length, frame_rate):
    """
    Generate B-spline basis functions.
    
    Parameters:
    num_basis (int): Number of spline basis functions.
    window_length (float): Window length in seconds for the basis functions.
    frame_rate (float): Frame rate of the data (frames per second).
    
    Returns:
    np.ndarray: Matrix of basis functions (num_basis, T).
    """
    # Convert window length to number of frames
    T = int(window_length * frame_rate)
    
    # Generate the x values (time points)
    x = np.linspace(0, window_length, T)
    
    # Degree of the spline (cubic splines)
    degree = 3
    
    # Generate the number of knots
    # We need enough knots to satisfy the spline degree
    num_knots = num_basis + degree + 1
    internal_knots = np.linspace(0, window_length, num_knots - 2 * degree)
    
    # Pad the knot vector by repeating the boundary knots
    knots = np.pad(internal_knots, (degree, degree), mode='edge')
    
    # Ensure that the knots cover the entire range of x
    if (x.min() < knots[degree]) or (x.max() > knots[-degree-1]):
        raise ValueError(f"x values are out of bounds for the given knots.")
    
    # Generate B-spline basis functions
    basis_functions = interpolate.BSpline.design_matrix(x, knots, degree).toarray().T
    
    return basis_functions

def convolve_event_with_splines(event_traces, basis_functions, names):
    """
    Convolve each event binary trace in a list with spline basis functions and return corresponding names.
    
    Parameters:
    event_traces (list): List of 1D binary trace arrays of shape (T,).
    basis_functions (np.ndarray): Array of spline basis functions of shape (num_basis, window_length).
    names (list): List of names (str) corresponding to each event trace. Should have the same length as event_traces.
    
    Returns:
    tuple: (list of 1D convolved predictors, list of names with basis function index appended)
    """
    num_basis, window_length = basis_functions.shape
    num_traces = len(event_traces)
    
    # Ensure the names list is the same length as event_traces
    if len(names) != num_traces:
        raise ValueError("Length of names must be the same as length of event_traces.")
    
    # Initialize lists to store the convolved predictors and updated names
    convolved_predictors_list = []
    updated_names = []
    
    # Loop over each event trace in the list
    for t, trace in enumerate(event_traces):
        for i in range(num_basis):
            # Perform the convolution
            convolved = np.convolve(trace, basis_functions[i], mode='full')[:len(trace)]
            
            # Add the convolved 1D array to the list
            convolved_predictors_list.append(convolved)
            
            # Append the corresponding name with the basis function index
            updated_names.append(f"{names[t]}_bf_{i}")
    
    return convolved_predictors_list, updated_names

def shift_predictors(predictors_list, predictor_names, window, frame_rate):
    """
    Shift a list of 1D predictors forward and backward by a given window (in seconds).
    
    Parameters:
    predictors_list (list): List of original 1D predictors arrays, each of shape (T,).
    predictor_names (list): List of predictor names corresponding to each array in predictors_list.
    window (tuple): Tuple (x, y) representing the time window in seconds. 
                    x is the backward shift window (positive delay), 
                    y is the forward shift window (negative delay).
    frame_rate (float): Frame rate of the data (frames per second).
    
    Returns:
    tuple: (shifted_predictors_list, shifted_predictor_names)
        - shifted_predictors_list: List of shifted predictors arrays, with shape (T, num_total_shifts).
        - shifted_predictor_names: List of names corresponding to each shifted predictor.
    """
    # Unpack the window into backward and forward shift
    backward_window, forward_window = window
    
    # Convert window from seconds to number of frames
    backward_shift_frames = int(backward_window * frame_rate)
    forward_shift_frames = int(forward_window * frame_rate)
    
    # Initialize lists to store shifted predictors and corresponding names
    shifted_predictors_list = []
    shifted_predictor_names = []
    
    # Loop over each 1D predictor array in the predictors_list
    for predictor, predictor_name in zip(predictors_list, predictor_names):
        # Number of time steps
        T = predictor.shape[0]
        
        # Shift predictors backward (positive delay, use np.roll with negative shift)
        for shift in np.arange(1,backward_shift_frames+1)[::-1]:
            shifted = np.roll(predictor, -shift)
            # Zero out the shifted part that has no actual data (at the end)
            shifted[-shift:] = 0
            shifted_predictors_list.append(shifted)
            # Add names for backward shifts
            shifted_predictor_names.append(f'backward_{shift}frame_{predictor_name}')
        
        # Add the original (unshifted) predictor
        shifted_predictors_list.append(predictor)
        shifted_predictor_names.append(f'{predictor_name}')
        
        # Shift predictors forward (negative delay, use np.roll with positive shift)
        for shift in range(1, forward_shift_frames + 1):
            shifted = np.roll(predictor, shift)
            # Zero out the shifted part that has no actual data (at the beginning)
            shifted[:shift] = 0
            shifted_predictors_list.append(shifted)
            # Add names for forward shifts
            shifted_predictor_names.append(f'forward_{shift}frame_{predictor_name}')
    
    
    
    return shifted_predictors_list, shifted_predictor_names


def generate_polynomial_predictors(continuous_variable, max_degree):
    """
    Generate polynomial terms for a continuous variable up to a specified degree.
    
    Parameters:
    continuous_variable (np.ndarray): Continuous variable of shape (T,).
    max_degree (int): Maximum degree of polynomial.
    
    Returns:
    np.ndarray: Array of shape (T, max_degree) containing polynomial predictors.
    """
    predictors = np.zeros((len(continuous_variable), max_degree))
    for degree in range(1, max_degree + 1):
        predictors[:, degree - 1] = continuous_variable ** degree
    return predictors

def transform_continuous_variable(components, name):
    """
    Transform a 2D array into a list of 1D arrays, with corresponding names.
    
    Parameters:
    components (np.ndarray): 2D array of shape (num_components, timepoints).
    name (str): Base name for the components.
    
    Returns:
    tuple: (list of 1D arrays, list of names corresponding to each component)
    """
    num_components, _ = components.shape
    
    # Initialize lists to store the 1D arrays and corresponding names
    component_list = []
    name_list = []
    
    # Loop through each row (component) of the 2D array
    for i in range(num_components):
        # Extract the 1D array (row)
        component_list.append(components[i])
        
        # Generate the corresponding name
        name_list.append(f"{name}_comp_{i}")
    
    return component_list, name_list

def zscore_predictors(predictors):
    """
    Z-score the predictors along the time axis.
    
    Parameters:
    predictors (np.ndarray): Predictors array.
    
    Returns:
    np.ndarray: Z-scored predictors.
    """
    return zscore(predictors, axis=0, ddof=1)

def construct_design_matrix(event_predictors_list, event_predictor_names, 
                            whole_trial_predictors, whole_trial_predictor_names, 
                            continuous_predictors_list, continuous_predictor_names):
    """
    Construct the design matrix from all 1D predictors, z-score each predictor, and generate a corresponding list of names.
    
    Parameters:
    event_predictors_list (list): List of 1D event predictors arrays.
    event_predictor_names (list): List of event predictor names.
    whole_trial_predictors (list): List of 1D whole-trial predictors arrays.
    whole_trial_predictor_names (list): List of whole-trial predictor names.
    continuous_predictors_list (list): List of 1D continuous predictors arrays.
    continuous_predictor_names (list): List of continuous predictor names.
    
    Returns:
    tuple: (design_matrix, predictor_names)
        - design_matrix: The constructed design matrix as a 2D numpy array.
        - predictor_names: List of predictor names corresponding to the columns of the design matrix.
    """
    predictors = []
    predictor_names = []
    
    # Z-score and add event predictors and their names
    for event_predictors, event_name in zip(event_predictors_list, event_predictor_names):
        zscored_event = zscore(event_predictors)  # Z-score each event predictor
        predictors.append(zscored_event)
        predictor_names.append(event_name)
    
    # Z-score and add whole-trial predictors and their names
    if whole_trial_predictors is not None:
        for predictor, name in zip(whole_trial_predictors, whole_trial_predictor_names):
            zscored_predictor = zscore(predictor)  # Z-score each whole-trial predictor
            predictors.append(zscored_predictor)
            predictor_names.append(name)
    
    # Z-score and add continuous predictors and their names
    for continuous_predictors, continuous_name in zip(continuous_predictors_list, continuous_predictor_names):
        zscored_continuous = zscore(continuous_predictors)  # Z-score each continuous predictor
        predictors.append(zscored_continuous)
        predictor_names.append(continuous_name)
    
    # Stack all predictors as columns in the design matrix
    design_matrix = np.column_stack(predictors)
    
    return design_matrix, predictor_names

def estimate_beta(F, design_matrix, regression_type='linear', alpha=1.0):
    """
    Estimate beta coefficients for each neuron using least squares, Lasso, or Ridge regression.
    
    Parameters:
    F (np.ndarray): Neural data array of shape (num_neurons, timepoints).
    design_matrix (np.ndarray): Design matrix of shape (timepoints, num_predictors).
    regression_type (str): Type of regression ('linear', 'lasso', 'ridge').
    alpha (float): Regularization strength (used only for Lasso or Ridge).
    
    Returns:
    tuple: (Beta coefficients array, intercepts array)
        - Beta coefficients array of shape (num_neurons, num_predictors).
        - Intercepts array of shape (num_neurons,).
    """
    num_neurons, _ = F.shape
    num_predictors = design_matrix.shape[1]
    
    # Initialize arrays to store the beta coefficients and intercepts for each neuron
    beta_matrix = np.zeros((num_neurons, num_predictors))
    intercepts = np.zeros(num_neurons)
    
    # Loop over each neuron and fit the regression model
    for neuron_idx in range(num_neurons):
        if regression_type == 'linear':
            reg = LinearRegression(fit_intercept=True)
        elif regression_type == 'lasso':
            reg = Lasso(alpha=alpha, fit_intercept=True)
        elif regression_type == 'ridge':
            reg = Ridge(alpha=alpha, fit_intercept=True)
        else:
            raise ValueError("Invalid regression type. Choose 'linear', 'lasso', or 'ridge'.")
        
        # Fit the model for the current neuron's data
        reg.fit(design_matrix, F[neuron_idx])
        
        # Store the beta coefficients and intercept
        beta_matrix[neuron_idx, :] = reg.coef_
        intercepts[neuron_idx] = reg.intercept_
    
    return beta_matrix, intercepts

def encoding_model(F, design_matrix, regression_type='linear', alpha=1.0, n_splits=5):
    """
    Implement the encoding model with pre-constructed design matrix and return cross-validated explained variance.
    
    Parameters:
    F (np.ndarray): Neural data array of shape (num_neurons, timepoints).
    design_matrix (np.ndarray): Design matrix of shape (timepoints, num_predictors).
    regression_type (str): Type of regression ('linear', 'lasso', 'ridge').
    alpha (float): Regularization strength (used only for Lasso or Ridge).
    n_splits (int): Number of splits for cross-validation.
    
    Returns:
    tuple: (beta coefficients array, intercepts array, cross-validated explained variance array)
        - beta coefficients array of shape (num_neurons, num_predictors)
        - intercepts array of shape (num_neurons,)
        - explained variance array of shape (num_neurons,)
    """
    num_neurons, T = F.shape  # F is now of shape (num_neurons, timepoints)
    
    # Estimate beta coefficients and intercepts for each neuron using all data
    beta_matrix, intercepts = estimate_beta(F, design_matrix, regression_type=regression_type, alpha=alpha)
    
    # Initialize an array to store cross-validated explained variance for each neuron
    explained_variances = np.zeros(num_neurons)
    
    # Cross-validation setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Calculate cross-validated explained variance for each neuron
    for neuron_idx in range(num_neurons):
        fold_variances = []
        
        for train_index, test_index in kf.split(design_matrix):
            X_train, X_test = design_matrix[train_index], design_matrix[test_index]
            y_train, y_test = F[neuron_idx, train_index], F[neuron_idx, test_index]
            
            # Choose the regression model
            if regression_type == 'linear':
                reg = LinearRegression(fit_intercept=True)
            elif regression_type == 'lasso':
                reg = Lasso(alpha=alpha, fit_intercept=True)
            elif regression_type == 'ridge':
                reg = Ridge(alpha=alpha, fit_intercept=True)
            else:
                raise ValueError("Invalid regression type. Choose 'linear', 'lasso', or 'ridge'.")
            
            # Fit the model on the training set
            reg.fit(X_train, y_train)
            
            # Predict on the test set
            y_pred = reg.predict(X_test)
            
            # Calculate explained variance for this fold and append
            fold_variances.append(explained_variance_score(y_test, y_pred))
        
        # Average explained variance across folds for the current neuron
        explained_variances[neuron_idx] = np.mean(fold_variances)
    
    return beta_matrix, intercepts, explained_variances

def encoding_model_with_significance(F, design_matrix, regression_type='linear', alpha=1.0):
    """
    Implement the encoding model with significance testing for each predictor and return reduced explained variance.
    
    Parameters:
    F (np.ndarray): Neural data array of shape (num_neurons, timepoints).
    design_matrix (np.ndarray): Design matrix of shape (timepoints, num_predictors).
    regression_type (str): Type of regression ('linear', 'lasso', 'ridge').
    alpha (float): Regularization strength (used only for Lasso or Ridge).
    
    Returns:
    tuple: 
        - beta_matrix: Beta coefficients array of shape (num_neurons, num_predictors).
        - intercepts: Intercepts array of shape (num_neurons,).
        - explained_variances: Explained variance array of shape (num_neurons,).
        - reduced_explained_variances: Reduced explained variance array of shape (num_neurons, num_predictors).
        - F_statistics: F-statistics array of shape (num_neurons, num_predictors).
    """
    num_neurons, T = F.shape  # F is now of shape (num_neurons, timepoints)
    num_predictors = design_matrix.shape[1]
    
    # Estimate beta coefficients and intercepts for each neuron (full model)
    beta_matrix, intercepts = estimate_beta(F, design_matrix, regression_type=regression_type, alpha=alpha)
    
    # Initialize arrays to store explained variance, reduced explained variance, and F-statistics
    explained_variances = np.zeros(num_neurons)
    reduced_explained_variances = np.zeros((num_neurons, num_predictors))
    F_statistics = np.zeros((num_neurons, num_predictors))
    
    # Full model residual sum of squares (RSS)
    RSS_full = np.zeros(num_neurons)
    
    # Calculate explained variance for the full model and RSS
    for neuron_idx in range(num_neurons):
        F_pred = design_matrix @ beta_matrix[neuron_idx, :] + intercepts[neuron_idx]
        RSS_full[neuron_idx] = np.sum((F[neuron_idx] - F_pred) ** 2)
        explained_variances[neuron_idx] = explained_variance_score(F[neuron_idx], F_pred)
    
    # Fit reduced models by removing each predictor and calculate F-statistics and reduced explained variance
    for predictor_idx in range(num_predictors):
        # Create a reduced design matrix by removing the current predictor
        reduced_design_matrix = np.delete(design_matrix, predictor_idx, axis=1)
        
        # Estimate beta coefficients and intercepts for the reduced model
        reduced_beta_matrix, reduced_intercepts = estimate_beta(F, reduced_design_matrix, regression_type=regression_type, alpha=alpha)
        
        # Calculate the explained variance and RSS for the reduced model
        for neuron_idx in range(num_neurons):
            F_pred_reduced = reduced_design_matrix @ reduced_beta_matrix[neuron_idx, :] + reduced_intercepts[neuron_idx]
            RSS_reduced = np.sum((F[neuron_idx] - F_pred_reduced) ** 2)
            reduced_explained_variances[neuron_idx, predictor_idx] = explained_variance_score(F[neuron_idx], F_pred_reduced)
            
            # Calculate F-statistic for this predictor
            numerator = (RSS_reduced - RSS_full[neuron_idx]) / 1  # Degrees of freedom lost = 1
            denominator = RSS_full[neuron_idx] / (T - num_predictors)
            F_statistics[neuron_idx, predictor_idx] = numerator / denominator
    
    return beta_matrix, intercepts, explained_variances, reduced_explained_variances, F_statistics