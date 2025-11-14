import pandas as pd
import numpy as np
from scipy import interpolate, stats
from scipy.stats import zscore
from sklearn.metrics import explained_variance_score
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, MultiTaskLasso, MultiTaskElasticNet
from joblib import Parallel, delayed
from multiprocessing import Pool
import warnings
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


def transform_trial_table(df, valve_silent = False, punish_silent = False):
    """
    Transforms the input trial DataFrame by adding columns for left/right choices, reward/punish times,
    and trial modality (visual/auditory).

    Parameters:
    df (pd.DataFrame): Input trial DataFrame with 'choice', 'feedbackType', 'feedback_times', and 'modality' columns.

    Returns:
    pd.DataFrame: Transformed DataFrame with added columns.
    """
    df_transformed = df.copy()

    df_transformed.loc[np.isnan(df_transformed['omission']),'omission'] = 0
    

    # Add 'reward_times' column: 'feedback_times' where 'feedbackType' is 1, NaN otherwise
    if valve_silent:
        df_transformed['reward_times'] = np.where((df_transformed['feedbackType'] == 1) & (df_transformed['omission'] == 0), df_transformed['reward_consume_times'], np.nan)
    else:
        df_transformed['reward_times'] = np.where((df_transformed['feedbackType'] == 1) & (df_transformed['omission'] == 0), df_transformed['feedback_times'], np.nan)

    # Add 'punish_times' column: 'feedback_times' where 'feedbackType' is -1, NaN otherwise
    if punish_silent:
        df_transformed['punish_times'] = np.where((df_transformed['feedbackType'] == -1) & (df_transformed['omission'] == 0), df_transformed['punish_consume_times'], np.nan)
    else:
        df_transformed['punish_times'] = np.where((df_transformed['feedbackType'] == -1) & (df_transformed['omission'] == 0), df_transformed['feedback_times'], np.nan)

    df_transformed['omission_times'] = np.where((df_transformed['feedbackType']==1) & (df_transformed['omission'] == 1), df_transformed['feedback_times'], np.nan)
    # Add 'visual_trial' column: 1 where 'modality' is 0, 0 elsewhere
    df_transformed['visual_trial'] = np.where(df_transformed['modality'] == 0, 1, 0)

    # Add 'aud_trial' column: 1 where 'modality' is 1, 0 elsewhere
    df_transformed['aud_trial'] = np.where(df_transformed['modality'] == 1, 1, 0)
    
    if valve_silent:
        df_transformed['vis_reward_times'] = np.where((df_transformed['modality'] == 0) & (df_transformed['feedbackType'] == 1) & (df_transformed['omission'] == 0), df_transformed['reward_consume_times'], np.nan)
        df_transformed['aud_reward_times'] = np.where((df_transformed['modality'] == 1) & (df_transformed['feedbackType'] == 1) & (df_transformed['omission'] == 0), df_transformed['reward_consume_times'], np.nan)
    else:
        df_transformed['vis_reward_times'] = np.where((df_transformed['modality'] == 0) & (df_transformed['feedbackType'] == 1) & (df_transformed['omission'] == 0), df_transformed['feedback_times'], np.nan)
        df_transformed['aud_reward_times'] = np.where((df_transformed['modality'] == 1) & (df_transformed['feedbackType'] == 1) & (df_transformed['omission'] == 0), df_transformed['feedback_times'], np.nan)

    if punish_silent:
        df_transformed['vis_punish_times'] = np.where((df_transformed['modality'] == 0) & (df_transformed['feedbackType'] == -1) & (df_transformed['omission'] == 0), df_transformed['punish_consume_times'], np.nan)
        df_transformed['aud_punish_times'] = np.where((df_transformed['modality'] == 1) & (df_transformed['feedbackType'] == -1) & (df_transformed['omission'] == 0), df_transformed['punish_consume_times'], np.nan)
    else:
        df_transformed['vis_punish_times'] = np.where((df_transformed['modality'] == 0) & (df_transformed['feedbackType'] == -1) & (df_transformed['omission'] == 0), df_transformed['feedback_times'], np.nan)
        df_transformed['aud_punish_times'] = np.where((df_transformed['modality'] == 1) & (df_transformed['feedbackType'] == -1) & (df_transformed['omission'] == 0), df_transformed['feedback_times'], np.nan)


    df_transformed['vis_ruleCue_times'] = np.where(df_transformed['modality'] == 0, df_transformed['ruleCue_times'], np.nan)
    df_transformed['aud_ruleCue_times'] = np.where(df_transformed['modality'] == 1, df_transformed['ruleCue_times'], np.nan)
    # Add 'vis_stimOn' column: 'stimOnTrigger_times' where 'modality' is 0, NaN otherwise
    df_transformed['left_vis_stimOn_times'] = np.where(df_transformed['contrastLeft'] > 0, df_transformed['vis_stimOn_times'], np.nan)
    df_transformed['right_vis_stimOn_times'] = np.where(df_transformed['contrastRight'] > 0, df_transformed['vis_stimOn_times'], np.nan)
    # Add 'aud_stimOn' column: 'stimOnTrigger_times' where 'modality' is 1, NaN otherwise
    df_transformed['left_aud_stimOn_times'] = np.where(df_transformed['contrastLeft'] == 0.0, df_transformed['aud_stimOn_times'], np.nan)
    df_transformed['right_aud_stimOn_times'] = np.where(df_transformed['contrastRight'] == 0.0, df_transformed['aud_stimOn_times'], np.nan)

    # df_transformed['vis_left_choice_times'] = np.where((df_transformed['choice'] == -1) & (df_transformed['modality'] == 0), df_transformed['firstMovement_times']-0.2*15, np.nan)
    # df_transformed['vis_right_choice_times'] = np.where((df_transformed['choice'] == 1) & (df_transformed['modality'] == 0), df_transformed['firstMovement_times']-0.2*15, np.nan)
    # df_transformed['aud_left_choice_times'] = np.where((df_transformed['choice'] == -1) & (df_transformed['modality'] == 1), df_transformed['firstMovement_times']-0.2*15, np.nan)
    # df_transformed['aud_right_choice_times'] = np.where((df_transformed['choice'] == 1) & (df_transformed['modality'] == 1), df_transformed['firstMovement_times']-0.2*15, np.nan)
    df_transformed['vis_left_choice_times'] = np.where((df_transformed['choice'] == -1) & (df_transformed['modality'] == 0), df_transformed['lastMovement_times']-0.2*15, np.nan)
    df_transformed['vis_right_choice_times'] = np.where((df_transformed['choice'] == 1) & (df_transformed['modality'] == 0), df_transformed['lastMovement_times']-0.2*15, np.nan)
    df_transformed['aud_left_choice_times'] = np.where((df_transformed['choice'] == -1) & (df_transformed['modality'] == 1), df_transformed['lastMovement_times']-0.2*15, np.nan)
    df_transformed['aud_right_choice_times'] = np.where((df_transformed['choice'] == 1) & (df_transformed['modality'] == 1), df_transformed['lastMovement_times']-0.2*15, np.nan)

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
        if np.sum(event_predictors==0) == len(event_predictors): # all zero arrays cannot be zscored
            pass
        else:
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

def block_shuffle(data, block_size, random_state=None):
    """
    Shuffle time series data in blocks, preserving the last incomplete block.
    
    Parameters:
    data: array of shape (num_neurons, timepoints)
    block_size: number of timepoints per block
    
    Returns:
    shuffled data of same shape
    """
    rng = np.random.RandomState(random_state)
    n_neurons, n_timepoints = data.shape
    n_blocks = n_timepoints // block_size
    remainder = n_timepoints % block_size
    
    if remainder == 0:
        # If data length is exactly divisible by block_size
        blocks = data.reshape(n_neurons, n_blocks, block_size)
        block_indices = rng.permutation(n_blocks)
        shuffled_data = blocks[:, block_indices, :].reshape(n_neurons, n_timepoints)
    else:
        # Handle the main blocks
        main_data = data[:, :(n_blocks * block_size)]
        blocks = main_data.reshape(n_neurons, n_blocks, block_size)
        block_indices = rng.permutation(n_blocks)
        shuffled_main = blocks[:, block_indices, :].reshape(n_neurons, -1)
        
        # Preserve the remainder block at the end
        remainder_data = data[:, (n_blocks * block_size):]
        shuffled_data = np.concatenate([shuffled_main, remainder_data], axis=1)
    
    return shuffled_data

def compute_cv_F_stats(F, design_matrix, kf, unique_predictors, full_predictors, reg_model):
    """Helper function to compute cross-validated F-statistics for one dataset"""
    num_neurons = F.shape[0]
    num_unique_predictors = len(unique_predictors)
    n_splits = kf.n_splits
    
    # Arrays to store RSS for each fold
    cv_full_RSS = np.zeros((n_splits, num_neurons))
    cv_reduced_RSS = np.zeros((n_splits, num_neurons, num_unique_predictors))
    
    # print("F shape:", F.shape)
    # print("Design matrix shape:", design_matrix.shape)
    # print("Number of splits:", kf.n_splits)

    for fold_idx, (train_index, test_index) in enumerate(kf.split(design_matrix)):
        # print("Max train index:", max(train_index))
        # print("Max test index:", max(test_index))
        X_train, X_test = design_matrix[train_index], design_matrix[test_index]
        y_train, y_test = F[:, train_index].T, F[:, test_index].T
        
        # Fit full model
        reg = reg_model()
        reg.fit(X_train, y_train)
        y_pred_test = reg.predict(X_test)
        
        # Calculate full model RSS on test data
        cv_full_RSS[fold_idx] = np.sum((y_test - y_pred_test) ** 2, axis=0)
        
        # Calculate reduced model RSS for each predictor
        for unique_idx, unique_name in enumerate(unique_predictors):
            columns_to_remove = [i for i, name in enumerate(full_predictors) if unique_name in name]
            X_train_reduced = np.delete(X_train, columns_to_remove, axis=1)
            X_test_reduced = np.delete(X_test, columns_to_remove, axis=1)
            
            reg_reduced = reg_model()
            reg_reduced.fit(X_train_reduced, y_train)
            y_pred_test_reduced = reg_reduced.predict(X_test_reduced)
            
            cv_reduced_RSS[fold_idx, :, unique_idx] = np.sum(
                (y_test - y_pred_test_reduced) ** 2, axis=0
            )
    
    # Average RSS across folds
    mean_full_RSS = np.mean(cv_full_RSS, axis=0)
    mean_reduced_RSS = np.mean(cv_reduced_RSS, axis=0)
    
    # Store both F-stats and RSS differences for later analysis
    F_stats = np.zeros((num_neurons, num_unique_predictors))
    RSS_differences = np.zeros((num_neurons, num_unique_predictors))
    
    for unique_idx in range(num_unique_predictors):
        delta_p = len([i for i, name in enumerate(full_predictors) 
                      if unique_predictors[unique_idx] in name])
        df1 = delta_p
        df2 = len(test_index) - design_matrix.shape[1]
        
        # Calculate RSS difference and set negative values to 0
        RSS_diff = mean_reduced_RSS[:, unique_idx] - mean_full_RSS
        RSS_differences[:, unique_idx] = RSS_diff  # store original differences
        RSS_diff = np.maximum(RSS_diff, 0)  # force non-negative
        
        numerator = RSS_diff / df1
        denominator = mean_full_RSS / df2
        F_stats[:, unique_idx] = numerator / denominator
    
    return F_stats, RSS_differences, mean_full_RSS, mean_reduced_RSS

def bootstrap_iteration(iteration, F, design_matrix, kf, unique_predictors, full_predictors, 
                       reg_model, block_size):
    """Single bootstrap iteration for parallel processing"""
    F_shuffled = block_shuffle(F, block_size, random_state=iteration)
    F_stats, RSS_diffs, full_RSS, red_RSS = compute_cv_F_stats(
        F_shuffled, design_matrix, kf, unique_predictors, full_predictors, reg_model
    )
    return F_stats, RSS_diffs, full_RSS, red_RSS

def encoding_model_with_significance_cv(
    F, 
    design_matrix, 
    frame_rate,
    regression_type='linear', 
    alpha=1.0,  # renamed from alpha to avoid confusion
    n_splits=5, 
    n_bootstraps=100,
    unique_predictors=None, 
    full_predictors=None,
    n_jobs=-1,
    show_progress=True
):
    """
    Encoding model analysis with cross-validated F-statistics and block bootstrap testing.
    
    Returns beta coefficients, intercepts, explained variances, and significance measures.
    """
    num_neurons, T = F.shape
    num_predictors = design_matrix.shape[1]
    num_unique_predictors = len(unique_predictors)
    
    # Initialize arrays
    beta_matrix = np.zeros((num_neurons, num_predictors))
    intercepts = np.zeros(num_neurons)
    explained_variances = np.zeros(num_neurons)
    unique_explained_variances = np.zeros((num_neurons, num_unique_predictors))
    F_statistics = np.zeros((num_neurons, num_unique_predictors))
    p_values = np.zeros((num_neurons, num_unique_predictors))
    bootstrap_p_values = np.zeros((num_neurons, num_unique_predictors))
    
    # Cross-validation setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Choose regression model type
    if regression_type == 'linear':
        RegModel = lambda: LinearRegression(fit_intercept=True)
    elif regression_type == 'ridge':
        RegModel = lambda: Ridge(alpha=alpha, fit_intercept=True)
    elif regression_type == 'lasso':
        RegModel = lambda: MultiTaskLasso(alpha=alpha, fit_intercept=True)
    else:
        raise ValueError("Invalid regression type. Choose 'linear', 'lasso', or 'ridge'.")
    
    # Check for collinearity using VIF
    _ = check_multicollinearity(design_matrix, full_predictors,
                            vif_thresh=5.0, corr_thresh=0.90, cond_thresh=1e4, verbose=True)
    
    # Fit full model on entire dataset to get beta coefficients and intercepts
    full_model = RegModel()
    full_model.fit(design_matrix, F.T)
    beta_matrix = full_model.coef_
    intercepts = full_model.intercept_
    
    # Calculate total explained variance for each neuron
    y_pred = full_model.predict(design_matrix)
    total_ss = np.sum((F.T - np.mean(F.T, axis=0))**2, axis=0)
    residual_ss = np.sum((F.T - y_pred)**2, axis=0)
    explained_variances = 1 - (residual_ss / total_ss)
    
    # Calculate unique explained variance for each predictor
    for unique_idx, unique_name in enumerate(unique_predictors):
        # Remove columns corresponding to this predictor
        columns_to_remove = [i for i, name in enumerate(full_predictors) if unique_name in name]
        X_reduced = np.delete(design_matrix, columns_to_remove, axis=1)
        
        # Fit reduced model
        reduced_model = RegModel()
        reduced_model.fit(X_reduced, F.T)
        y_pred_reduced = reduced_model.predict(X_reduced)
        
        # Calculate unique explained variance
        residual_ss_reduced = np.sum((F.T - y_pred_reduced)**2, axis=0)
        unique_explained_variances[:, unique_idx] = (residual_ss_reduced - residual_ss) / total_ss
    
    # Compute real cross-validated F-statistics and RSS values
    F_statistics, RSS_differences, real_full_RSS, real_reduced_RSS = compute_cv_F_stats(
        F, design_matrix, kf, unique_predictors, full_predictors, RegModel
    )
    
    # Calculate parametric p-values with correction for multiple comparisons
    for n in range(num_neurons):
        for unique_idx in range(num_unique_predictors):
            delta_p = len([i for i, name in enumerate(full_predictors) 
                          if unique_predictors[unique_idx] in name])
            df1 = delta_p
            # Correct df2 calculation - use actual test set size
            df2 = len(design_matrix) // n_splits - design_matrix.shape[1]
            # Ensure degrees of freedom are positive
            if df2 <= 0:
                p_values[n, unique_idx] = 1.0
                continue
                
            # Calculate raw p-value
            raw_p = 1 - stats.f.cdf(F_statistics[n, unique_idx], df1, df2)
            # Store raw p-value for multiple comparison correction later
            p_values[n, unique_idx] = raw_p
    
    
    # Perform parallel bootstrap iterations
    block_size = int(frame_rate)  # 1 second blocks
    
    iterator = range(n_bootstraps)
    if show_progress:
        iterator = tqdm(iterator, desc='Bootstraps')
    
    bootstrap_results = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_iteration)(
            i, F, design_matrix, kf, unique_predictors, full_predictors, RegModel, block_size
        ) for i in iterator
    )
    
    # Unpack bootstrap results
    bootstrap_F_stats = np.array([res[0] for res in bootstrap_results])
    bootstrap_RSS_diffs = np.array([res[1] for res in bootstrap_results])
    
    # Compute bootstrap p-values
    for n in range(num_neurons):
        for p in range(num_unique_predictors):
            # Add 1 to both numerator and denominator (recommended practice)
            bootstrap_p_values[n, p] = (1 + np.sum(
                bootstrap_F_stats[:, n, p] >= F_statistics[n, p]
            )) / (n_bootstraps + 1)
            
    
    # Compute confidence intervals (95%)
    confidence_intervals = np.zeros((num_neurons, num_unique_predictors, 2))
    for n in range(num_neurons):
        for p in range(num_unique_predictors):
            confidence_intervals[n, p] = np.percentile(
                bootstrap_F_stats[:, n, p], [2.5, 97.5]
            )
    
    # Count negative RSS differences
    negative_RSS_diff_counts = np.sum(RSS_differences < 0, axis=0)
    
    return (
        beta_matrix,
        intercepts,
        explained_variances,
        unique_explained_variances,
        F_statistics,
        p_values,
        bootstrap_p_values,
        bootstrap_F_stats,
        confidence_intervals,
        negative_RSS_diff_counts
    )

def fix_encoding_model_cv_uev(
    F, 
    design_matrix, 
    frame_rate,
    regression_type='linear', 
    alpha=1.0,  # renamed from alpha to avoid confusion
    n_splits=5, 
    unique_predictors=None, 
    full_predictors=None,
    n_jobs=-1
):
    """
    fix cross validated uev
    """
    num_neurons, T = F.shape
    num_predictors = design_matrix.shape[1]
    num_unique_predictors = len(unique_predictors)

    # Initialize arrays
    beta_matrix = np.zeros((num_neurons, num_predictors))
    intercepts = np.zeros(num_neurons)

    # These will now store CROSS-VALIDATED EV and UEV
    explained_variances_cv = np.zeros(num_neurons)
    unique_explained_variances_cv = np.zeros((num_neurons, num_unique_predictors))

    F_statistics = np.zeros((num_neurons, num_unique_predictors))
    p_values = np.zeros((num_neurons, num_unique_predictors))
    bootstrap_p_values = np.zeros((num_neurons, num_unique_predictors))

    # Cross-validation setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Choose regression model type
    if regression_type == 'linear':
        RegModel = lambda: LinearRegression(fit_intercept=True)
    elif regression_type == 'ridge':
        RegModel = lambda: Ridge(alpha=alpha, fit_intercept=True)
    elif regression_type == 'lasso':
        RegModel = lambda: MultiTaskLasso(alpha=alpha, fit_intercept=True)
    else:
        raise ValueError("Invalid regression type. Choose 'linear', 'lasso', or 'ridge'.")

    # Check for collinearity using VIF
    _ = check_multicollinearity(
        design_matrix, full_predictors,
        vif_thresh=5.0, corr_thresh=0.90, cond_thresh=1e4, verbose=True
    )

    # Fit full model on entire dataset to get beta coefficients and intercepts
    full_model = RegModel()
    full_model.fit(design_matrix, F.T)         # multi-output: shape (T, num_predictors) -> (T, num_neurons)
    beta_matrix = full_model.coef_            # shape (num_neurons, num_predictors)
    intercepts = full_model.intercept_        # shape (num_neurons,)

    # --------- CROSS-VALIDATED EV AND UEV ---------

    # Precompute which columns to remove for each unique predictor
    columns_to_remove_list = []
    for unique_name in unique_predictors:
        cols = [i for i, name in enumerate(full_predictors) if unique_name in name]
        columns_to_remove_list.append(cols)

    # Accumulators for CV sums
    RSS_full = np.zeros(num_neurons)                          # residual sum of squares for full model
    TSS = np.zeros(num_neurons)                               # total sum of squares (using train mean)
    RSS_reduced = np.zeros((num_neurons, num_unique_predictors))  # residuals for each reduced model

    X = design_matrix  # (T, num_predictors)

    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]

        # y is (T, num_neurons) for sklearn, so transpose F
        y_train = F[:, train_idx].T    # (len(train_idx), num_neurons)
        y_test = F[:, test_idx].T      # (len(test_idx), num_neurons)

        # Baseline: mean of training data per neuron
        y_train_mean = np.mean(y_train, axis=0, keepdims=True)  # (1, num_neurons)

        # Update total sum of squares (TSS) using train mean
        TSS += np.sum((y_test - y_train_mean) ** 2, axis=0)

        # ----- Full model -----
        full_model_cv = RegModel()
        full_model_cv.fit(X_train, y_train)
        y_pred_full = full_model_cv.predict(X_test)  # (len(test_idx), num_neurons)

        RSS_full += np.sum((y_test - y_pred_full) ** 2, axis=0)

        # ----- Reduced models for each unique predictor -----
        for u_idx, cols_to_remove in enumerate(columns_to_remove_list):
            X_train_red = np.delete(X_train, cols_to_remove, axis=1)
            X_test_red = np.delete(X_test, cols_to_remove, axis=1)

            red_model = RegModel()
            red_model.fit(X_train_red, y_train)
            y_pred_red = red_model.predict(X_test_red)

            RSS_reduced[:, u_idx] += np.sum((y_test - y_pred_red) ** 2, axis=0)

    # Convert sums into CV R² and CV UEV
    with np.errstate(divide='ignore', invalid='ignore'):
        explained_variances_cv = 1.0 - (RSS_full / TSS)
        for u_idx in range(num_unique_predictors):
            unique_explained_variances_cv[:, u_idx] = (RSS_reduced[:, u_idx] - RSS_full) / TSS

    # Handle neurons where TSS == 0 (e.g., flat signals)
    explained_variances_cv[~np.isfinite(explained_variances_cv)] = np.nan
    unique_explained_variances_cv[~np.isfinite(unique_explained_variances_cv)] = np.nan

    return (
        explained_variances_cv,
        unique_explained_variances_cv
    )

def encoding_model_cv_with_reconstruction(
    F, 
    design_matrix, 
    regression_type='linear', 
    alpha=1.0,
    n_splits=5,
    n_jobs=-1
):
    """
    Simplified encoding model with cross-validated reconstruction.
    
    Parameters:
    -----------
    F : array, shape (num_neurons, T)
        Neural activity data
    design_matrix : array, shape (T, num_predictors)
        Design matrix with predictors
    regression_type : str
        Type of regression ('linear', 'ridge', 'lasso')
    alpha : float
        Regularization parameter for ridge/lasso
    n_splits : int
        Number of cross-validation folds
    n_jobs : int
        Number of parallel jobs
        
    Returns:
    --------
    beta_matrix : array, shape (num_neurons, num_predictors)
        Beta coefficients from full model
    intercepts : array, shape (num_neurons,)
        Intercepts from full model
    explained_variances : array, shape (num_neurons,)
        Cross-validated explained variance for each neuron
    reconstructed_activity : array, shape (num_neurons, T)
        Cross-validated reconstructed neural activity
    original_activity_cv : array, shape (num_neurons, T)
        Original neural activity at cross-validated indices (same indices as reconstructed)
    cv_indices : array, shape (T,)
        Boolean array indicating which time points were used for cross-validation
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression, Ridge, MultiTaskLasso
    from sklearn.metrics import r2_score
    
    num_neurons, T = F.shape
    num_predictors = design_matrix.shape[1]
    
    # Initialize arrays
    beta_matrix = np.zeros((num_neurons, num_predictors))
    intercepts = np.zeros(num_neurons)
    explained_variances = np.zeros(num_neurons)
    reconstructed_activity = np.zeros((num_neurons, T))
    original_activity_cv = np.zeros((num_neurons, T))
    cv_indices = np.zeros(T, dtype=bool)  # Track which indices were used for CV
    
    # Initialize with NaN
    reconstructed_activity[:] = np.nan
    original_activity_cv[:] = np.nan
    
    # Cross-validation setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Choose regression model type
    if regression_type == 'linear':
        RegModel = lambda: LinearRegression(fit_intercept=True)
    elif regression_type == 'ridge':
        RegModel = lambda: Ridge(alpha=alpha, fit_intercept=True)
    elif regression_type == 'lasso':
        RegModel = lambda: MultiTaskLasso(alpha=alpha, fit_intercept=True)
    else:
        raise ValueError("Invalid regression type. Choose 'linear', 'lasso', or 'ridge'.")
    
    # Fit full model on entire dataset to get final beta coefficients and intercepts
    full_model = RegModel()
    full_model.fit(design_matrix, F.T)
    beta_matrix = full_model.coef_
    intercepts = full_model.intercept_
    
    # Cross-validated reconstruction and explained variance calculation
    for train_idx, test_idx in kf.split(design_matrix):
        # Split data
        X_train, X_test = design_matrix[train_idx], design_matrix[test_idx]
        y_train, y_test = F[:, train_idx].T, F[:, test_idx].T
        
        # Fit model on training data
        cv_model = RegModel()
        cv_model.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = cv_model.predict(X_test)
        
        # Store predictions and original activity at test indices
        reconstructed_activity[:, test_idx] = y_pred.T
        original_activity_cv[:, test_idx] = F[:, test_idx]
        cv_indices[test_idx] = True
    
    # Calculate cross-validated explained variance for each neuron
    for n in range(num_neurons):
        # Use only the cross-validated indices
        valid_idx = cv_indices
        if np.sum(valid_idx) > 0:
            explained_variances[n] = r2_score(
                original_activity_cv[n, valid_idx], 
                reconstructed_activity[n, valid_idx]
            )
        else:
            explained_variances[n] = 0.0
    
    return (
        beta_matrix,
        intercepts,
        explained_variances,
        reconstructed_activity,
        original_activity_cv,
        cv_indices
    )



def grid_search_encoding_model(F, design_matrix, param_grid, n_splits=5):
    """
    Perform grid search over hyperparameters for the encoding model.
    
    Parameters:
    F (np.ndarray): Neural data array of shape (num_neurons, timepoints).
    design_matrix (np.ndarray): Design matrix of shape (timepoints, num_predictors).
    param_grid (dict): Dictionary containing parameter grid for 'regression_type' and 'alpha'.
        Example: {'regression_type': ['linear', 'lasso', 'ridge'], 'alpha': [0.1, 1.0, 10.0]}.
    n_splits (int): Number of splits for cross-validation.
    
    Returns:
    dict: Best parameters and their corresponding explained variance.
        Example: {'best_params': {'regression_type': 'ridge', 'alpha': 1.0}, 'best_variance': 0.85}
    """
    from itertools import product
    import numpy as np
    
    # Initialize variables to store the best results
    best_params = None
    best_variance = -np.inf  # Start with the lowest possible variance
    
    # Get all combinations of parameters from the param grid
    param_combinations = list(product(param_grid['regression_type'], param_grid['alpha']))
    
    # Iterate over all parameter combinations
    for regression_type, alpha in param_combinations:
        print(f"Testing parameters: regression_type={regression_type}, alpha={alpha}")
        
        # Run the encoding model with the current parameters
        _, _, explained_variances = encoding_model(
            F, 
            design_matrix, 
            regression_type=regression_type, 
            alpha=alpha, 
            n_splits=n_splits
        )
        
        # Compute the average explained variance across neurons
        avg_variance = np.mean(explained_variances)
        
        # Update the best parameters if the current setup is better
        if avg_variance > best_variance:
            best_variance = avg_variance
            best_params = {'regression_type': regression_type, 'alpha': alpha}
    
    return {'best_params': best_params, 'best_variance': best_variance}

def _compute_vif(X):
    # VIF for each column: regress col i on all others -> 1/(1-R^2)
    n, p = X.shape
    vifs = np.empty(p, dtype=float)
    for i in range(p):
        others = [j for j in range(p) if j != i]
        Xi = X[:, others]
        yi = X[:, i]
        reg = LinearRegression()
        reg.fit(Xi, yi)
        r2 = reg.score(Xi, yi)
        vifs[i] = np.inf if r2 >= 0.999999999 else 1.0 / max(1.0 - r2, 1e-12)
    return vifs

def check_multicollinearity(design_matrix, full_predictors,
                            vif_thresh=5.0, corr_thresh=0.95,
                            cond_thresh=1e4, verbose=True,
                            ignore_substr="omission"):
    """
    Checks multicollinearity (VIF, pairwise correlation, duplicates, condition number).
    Completely drops any predictors whose names contain `ignore_substr`.
    """
    import warnings
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # --- filter out ignored predictors ---
    mask = [ignore_substr.lower() not in name.lower() for name in full_predictors]
    X = np.asarray(design_matrix)[:, mask]
    names = [name for keep, name in zip(mask, full_predictors) if keep]

    if X.shape[1] == 0:
        if verbose:
            warnings.warn(f"All predictors dropped due to '{ignore_substr}' filter.")
        return {}

    # --- standardize for stability ---
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)

    # 1) VIF
    vifs = []
    for i in range(Xz.shape[1]):
        others = [j for j in range(Xz.shape[1]) if j != i]
        Xi = Xz[:, others]
        yi = Xz[:, i]
        reg = LinearRegression()
        reg.fit(Xi, yi)
        r2 = reg.score(Xi, yi)
        vif = np.inf if r2 >= 0.999999999 else 1.0 / max(1.0 - r2, 1e-12)
        vifs.append(vif)
    vifs = np.array(vifs)

    high_vif_idx = np.where(vifs > vif_thresh)[0]
    if high_vif_idx.size and verbose:
        for i in high_vif_idx:
            warnings.warn(f"High VIF >{vif_thresh}: {names[i]} (VIF={vifs[i]:.2f})")

    # 2) Pairwise correlation
    corr = np.corrcoef(Xz, rowvar=False)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr[i, j]
            if abs(r) >= corr_thresh and verbose:
                warnings.warn(
                    f"High correlation |r|={abs(r):.3f} ≥ {corr_thresh} "
                    f"between {names[i]} ↔ {names[j]}"
                )

    # 3) Duplicates
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.allclose(Xz[:, i], Xz[:, j], atol=1e-10, rtol=1e-10):
                warnings.warn(f"Duplicate columns: {names[i]} ↔ {names[j]}")

    # 4) Condition number
    s = np.linalg.svd(Xz, full_matrices=False, compute_uv=False)
    cond = s.max() / max(s.min(), 1e-15)
    if cond > cond_thresh and verbose:
        warnings.warn(f"Design matrix ill-conditioned: cond={cond:.2e} (> {cond_thresh})")

    return {
        "predictors_checked": names,
        "vif": vifs,
        "high_vif_idx": high_vif_idx,
        "condition_number": cond,
    }



from scipy.linalg import eigh
from sklearn.linear_model import ElasticNetCV

# ---------- utilities ----------
def explained_variance(y_true, y_pred):
    resid = y_true - y_pred
    denom = np.var(y_true)
    return 0.0 if denom == 0 else 1.0 - (np.var(resid) / denom)

def compute_rrr_basis(P, F, lam_rr=1e-2, r_max=None, method='ridge_svd'):
    """
    Compute reduced-rank regression basis.
    
    Methods:
      'ridge_svd': SVD of ridge solution (current default, stable)
      'classic': Classic RRR via SVD of OLS/ridge solution
      'covariance': Maximize covariance between P*B and F (generalized eigenvalue)
    
    Shapes:
      P: (T, Ppred), F: (T, Nneur)
      B: (Ppred, r), PB: (T, r)
    Returns B (predictor-space basis), PB (temporal basis), and singular values/eigenvalues.
    """
    T, Ppred = P.shape
    _, Nneur = F.shape
    if r_max is None:
        r_max = min(Ppred, Nneur)

    if method == 'ridge_svd':
        # Current approach: Ridge → SVD → basis
        PtP = P.T @ P
        PtF = P.T @ F
        PtP_reg = PtP + lam_rr * np.eye(Ppred, dtype=P.dtype)
        W_ridge = np.linalg.solve(PtP_reg, PtF)  # (Ppred x Nneur)
        U, s, Vt = np.linalg.svd(W_ridge, full_matrices=False)
        r_eff = min(r_max, len(s))
        B = U[:, :r_eff]
        PB = P @ B
        evals = s[:r_eff]
        
    elif method == 'classic':
        # Classic RRR: Find rank-r approximation that minimizes ||F - P*W||²
        # With ridge regularization for stability
        PtP = P.T @ P
        PtF = P.T @ F
        PtP_reg = PtP + lam_rr * np.eye(Ppred, dtype=P.dtype)
        
        # Compute regularized predictor: P_reg = P @ (P'P + lam*I)^-1 P'
        # This is equivalent to ridge regression followed by projection
        W_ridge = np.linalg.solve(PtP_reg, PtF)
        
        # SVD of the cross-covariance weighted by ridge
        C = PtF.T @ np.linalg.solve(PtP_reg, PtF)  # (Nneur x Nneur)
        U_f, s_f, _ = np.linalg.svd(C, full_matrices=False)
        
        # Basis in predictor space
        B = np.linalg.solve(PtP_reg, PtF @ U_f[:, :r_max])  # (Ppred x r)
        r_eff = min(r_max, B.shape[1])
        B = B[:, :r_eff]
        PB = P @ B
        evals = s_f[:r_eff]
        
    elif method == 'covariance':
        # Maximize covariance: find B such that Cov(P*B, F) is maximized
        # Solves: (P'F F'P) b = λ (P'P + lam*I) b
        PtP = P.T @ P
        Cxy = P.T @ F  # (Ppred x Nneur)
        A = Cxy @ Cxy.T  # (Ppred x Ppred)
        S = PtP + lam_rr * np.eye(Ppred, dtype=P.dtype)
        
        evals_all, vecs_all = eigh(A, S)
        order = np.argsort(evals_all)[::-1]
        evals_all = evals_all[order]
        vecs_all = vecs_all[:, order]
        r_eff = min(r_max, vecs_all.shape[1])
        
        B = vecs_all[:, :r_eff]
        PB = P @ B
        evals = evals_all[:r_eff]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return B, PB, evals


def _fit_single_neuron_rrr(n, y, PB, B, r_grid, l1_ratio, alphas, cv, random_state, verbose):
    """Helper function to fit RRR for a single neuron (for parallelization)"""
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV
    
    T = len(y)
    Ppred = B.shape[0]
    
    best_cv_score, best_r, best_alpha = -np.inf, None, None
    
    # Rank selection
    for r in r_grid:
        Xr = PB[:, :r]
        
        if l1_ratio == 0.0:
            model = RidgeCV(alphas=alphas, cv=cv)
            model.fit(Xr, y)
            cv_score = model.score(Xr, y)
        else:
            model = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas, cv=cv,
                                 fit_intercept=True, n_jobs=1,  # Don't nest parallelism
                                 random_state=random_state)
            model.fit(Xr, y)
            cv_score = model.score(Xr, y)
        
        if cv_score > best_cv_score:
            best_cv_score, best_r, best_alpha = cv_score, r, model.alpha_
    
    # Refit with best rank
    r = best_r
    Xr = PB[:, :r]
    
    if l1_ratio == 0.0:
        model = Ridge(alpha=best_alpha, fit_intercept=True)
    else:
        model = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas, cv=cv,
                             fit_intercept=True, n_jobs=1,
                             random_state=random_state)
    model.fit(Xr, y)
    
    w = model.coef_.copy() if hasattr(model.coef_, 'copy') else model.coef_
    b0 = model.intercept_
    k = (B[:, :r] @ w)
    
    # CV for explained variance
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_preds = np.zeros(T)
    for train_idx, test_idx in kf.split(Xr):
        Xr_train, Xr_test = Xr[train_idx], Xr[test_idx]
        y_train = y[train_idx]
        
        if l1_ratio == 0.0:
            cv_model = Ridge(alpha=best_alpha, fit_intercept=True)
        else:
            cv_model = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas, cv=3,
                                    fit_intercept=True, n_jobs=1,
                                    random_state=random_state)
        cv_model.fit(Xr_train, y_train)
        cv_preds[test_idx] = cv_model.predict(Xr_test)
    
    ev = r2_score(y, cv_preds)
    y_hat = Xr @ w + b0
    
    if verbose:
        print(f"Neuron {n:4d} | r={r:2d} | alpha={best_alpha:.2e} | CV-EV={ev:.3f}")
    
    # Return best_alpha so it can be reused for UEV
    return k, b0, ev, y_hat, r, best_alpha


def fit_rrr_elasticnet(P, F, lam_rr=1e-2, l1_ratio=0.0, alphas=None, cv=5,
                       r_grid=None, n_jobs=None, random_state=0, verbose=False,
                       rrr_method='ridge_svd', fit_full_rank=False):
    """
    Full RRR pipeline + per-neuron Ridge/ElasticNet on PB with rank selection.
    Inputs:
      P: (T x Ppred) design matrix
      F: (Nneur x T) or (T x Nneur) — we'll accept either
      l1_ratio: 0.0 = Ridge (recommended), 0.5 = ElasticNet, 1.0 = Lasso
      rrr_method: 'ridge_svd' (default), 'classic', or 'covariance'
    Returns:
      beta:                 (Nneur x Ppred) per-neuron kernels in predictor space
      intercepts:           (Nneur,)
      explained_variances:  (Nneur,) - CROSS-VALIDATED
      preds:                (T x Nneur)
      B, PB:                basis objects (P@B == PB)
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.linear_model import RidgeCV
    
    # Ensure time-first
    if F.shape[0] < F.shape[1]:   # (Nneur x T) -> (T x Nneur)
        F = F.T
    
    T, Ppred = P.shape
    _, Nneur = F.shape

    # basis
    B, PB, evals = compute_rrr_basis(P, F, lam_rr=lam_rr, method=rrr_method)

    r_max = PB.shape[1]
    print('r_max', r_max)
    if r_grid is None:
        # Start from rank 1
        if not fit_full_rank:
            r_grid = list(range(1, min(50, r_max) + 1))
        else:
            r_grid = list(range(r_max, r_max + 1))
    
    # Set up regularization
    if l1_ratio == 0.0:
        # Use Ridge (no L1 penalty) - better for RRR
        if alphas is None:
            alphas = np.logspace(-4, 2, 30)
    else:
        # Use ElasticNet
        if alphas is None:
            alphas = np.logspace(-4, 0, 20)

    # Parallel processing across neurons
    from joblib import Parallel, delayed
    
    print(f'Fitting {Nneur} neurons with n_jobs={n_jobs}')
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_fit_single_neuron_rrr)(
            n, F[:, n], PB, B, r_grid, l1_ratio, alphas, cv, random_state, verbose
        )
        for n in range(Nneur)
    )
    
    # Unpack results
    beta = np.array([r[0] for r in results])
    intercepts = np.array([r[1] for r in results])
    explained_variances = np.array([r[2] for r in results])
    preds = np.array([r[3] for r in results]).T  # (T x Nneur)
    best_alphas = np.array([r[5] for r in results])  # Store optimal alphas

    return beta, intercepts, explained_variances, preds, B, PB, best_alphas

# ---------- unique explained variance ----------
def _compute_uev_single_neuron(n, y, P, b0, k, group_indices, unique_predictor_names,
                               l1_ratio, alphas, cv, random_state, refit, var_y_n, 
                               fixed_alpha=None):
    """Helper function to compute unique EV for a single neuron (for parallelization)"""
    from sklearn.linear_model import Ridge, ElasticNetCV, RidgeCV
    from sklearn.model_selection import KFold
    
    T = len(y)
    UEV_neuron = np.zeros(len(unique_predictor_names), dtype=float)
    
    # If fixed_alpha provided, use it for all groups (much faster)
    # Otherwise use the middle alpha value
    if fixed_alpha is None:
        use_alpha = alphas[len(alphas)//2] if l1_ratio == 0.0 else None
    else:
        use_alpha = fixed_alpha
    
    for g, idx_g in enumerate(group_indices):
        if idx_g.size == 0 or var_y_n == 0:
            UEV_neuron[g] = 0.0
            continue
        
        mask = np.ones(P.shape[1], dtype=bool)
        mask[idx_g] = False
        idx_other = np.where(mask)[0]
        
        # Use cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        cv_residuals = np.zeros(T)
        cv_res_hat = np.zeros(T)
        
        for train_idx, test_idx in kf.split(P):
            P_train, P_test = P[train_idx], P[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if refit:
                X_other_train = P_train[:, idx_other]
                X_other_test = P_test[:, idx_other]
                
                if l1_ratio == 0.0:
                    # Use fixed alpha (no CV search) for speed
                    model_other = Ridge(alpha=use_alpha, fit_intercept=True)
                else:
                    # Reduce alphas for speed
                    model_other = ElasticNetCV(l1_ratio=l1_ratio, 
                                              alphas=alphas[::2],  # Use every other alpha
                                              cv=2,  # Reduce CV folds
                                              fit_intercept=True, n_jobs=1, 
                                              random_state=random_state)
                model_other.fit(X_other_train, y_train)
                yhat_other_test = model_other.predict(X_other_test)
            else:
                yhat_other_test = P_test[:, idx_other] @ k[idx_other] + b0
            
            res_test = y_test - yhat_other_test
            cv_residuals[test_idx] = res_test
            
            if refit:
                yhat_other_train = model_other.predict(X_other_train)
            else:
                yhat_other_train = P_train[:, idx_other] @ k[idx_other] + b0
            res_train = y_train - yhat_other_train
            
            Xg_train = P_train[:, idx_g]
            Xg_test = P_test[:, idx_g]
            if Xg_train.ndim == 1:
                Xg_train = Xg_train.reshape(-1, 1)
                Xg_test = Xg_test.reshape(-1, 1)
            
            if l1_ratio == 0.0:
                # Use fixed alpha (no CV search) for speed
                model_g = Ridge(alpha=use_alpha, fit_intercept=False)
            else:
                # Reduce alphas for speed
                model_g = ElasticNetCV(l1_ratio=l1_ratio, 
                                      alphas=alphas[::2],  # Use every other alpha
                                      cv=2,  # Reduce CV folds
                                      fit_intercept=False,
                                      n_jobs=1, random_state=random_state)
            model_g.fit(Xg_train, res_train)
            cv_res_hat[test_idx] = model_g.predict(Xg_test)
        
        # Unique EV following Musall et al. approach:
        # How much variance in the RESIDUAL (from model without g) does group g explain?
        # This tells us what group g uniquely contributes beyond other predictors
        var_residual_without_g = np.var(cv_residuals)
        var_residual_with_g = np.var(cv_residuals - cv_res_hat)
        
        # Fraction of residual variance explained by group g
        if var_residual_without_g > 0:
            UEV_neuron[g] = 1.0 - (var_residual_with_g / var_residual_without_g)
        else:
            UEV_neuron[g] = 0.0
    
    return UEV_neuron


def compute_unique_explained_variance(P, F, beta, intercepts,
                                      predictor_names, unique_predictor_names,
                                      l1_ratio=0.5, alphas=None, cv=5, n_jobs=None,
                                      random_state=0, refit=True, fixed_alpha=None,
                                      per_neuron_alphas=None):
    """
    Compute unique explained variance for each predictor group.
    
    Two methods:
    1. refit=True (default, recommended): 
       - Refit model without group g, compute residual, fit residual with group g
       - More accurate, accounts for how other predictors compensate
       
    2. refit=False (faster):
       - Use original coefficients, remove group g contribution, fit residual
       - Faster but assumes coefficients don't change much when removing group
    
    For each unique predictor group g:
      - identify columns idx_g = [i for i,name in enumerate(predictor_names) if unique in name]
      - compute residual (method depends on 'refit')
      - fit model on P[:, idx_g] to predict residual
      - UEV_g = 1 - var(res - res_hat)/var(y)
    
    Returns UEV: (Nneur x G)
    """
    from sklearn.linear_model import Ridge, ElasticNetCV
    
    # time-first
    if F.shape[0] < F.shape[1]:
        F = F.T
    T, Nneur = F.shape
    P = np.asarray(P)
    beta = np.asarray(beta)       # (Nneur x Ppred)
    intercepts = np.asarray(intercepts)

    if alphas is None:
        # Fewer alphas for speed
        alphas = np.logspace(-3, 2, 10)

    # Build index lists for each unique group
    group_indices = []
    for uname in unique_predictor_names:
        idx = [i for i, nm in enumerate(predictor_names) if uname in nm]
        group_indices.append(np.array(idx, dtype=int))

    # precompute total var(y) per neuron
    var_y = np.var(F, axis=0)

    # Parallel processing across neurons
    from joblib import Parallel, delayed
    
    print(f'Computing unique EV for {Nneur} neurons with n_jobs={n_jobs}')
    UEV_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_compute_uev_single_neuron)(
            n, F[:, n], P, intercepts[n], beta[n, :], 
            group_indices, unique_predictor_names,
            l1_ratio, alphas, cv, random_state, refit, var_y[n],
            # Use per-neuron alpha if available, otherwise fall back to fixed_alpha
            per_neuron_alphas[n] if per_neuron_alphas is not None else fixed_alpha
        )
        for n in range(Nneur)
    )
    
    UEV = np.array(UEV_list)

    return UEV

def rrr_encoding_model_with_unique_ev(
    F,                   # (n_neurons x T) or (T x n_neurons)
    design_matrix,       # (T x P)
    predictor_names,     # list[str] length P
    unique_predictor_names,  # list[str]
    lam_rr=1e-2,
    enet_l1_ratio=0.0,   # Default to Ridge (0.0) instead of ElasticNet
    alphas=None,
    cv=5,
    n_jobs=None,
    random_state=0,
    verbose=False,
    rrr_method='ridge_svd',  # 'ridge_svd', 'classic', or 'covariance'
    refit_uev=True,      # Refit models for unique EV (more accurate but slower)
    fit_full_rank=False,
    uev_fixed_alpha=None,  # Fixed alpha for UEV (faster, e.g., 1.0)
):
    beta, intercepts, explained_variances, preds, B, PB, best_alphas = fit_rrr_elasticnet(
        P=design_matrix,
        F=F,
        lam_rr=lam_rr,
        l1_ratio=enet_l1_ratio,
        alphas=alphas,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        rrr_method=rrr_method,
        fit_full_rank=fit_full_rank,
    )

    # Use per-neuron alphas from main fit for UEV (unless uev_fixed_alpha is specified)
    UEV = compute_unique_explained_variance(
        P=design_matrix,
        F=F,
        beta=beta,
        intercepts=intercepts,
        predictor_names=predictor_names,
        unique_predictor_names=unique_predictor_names,
        l1_ratio=enet_l1_ratio,
        alphas=alphas,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=refit_uev,
        fixed_alpha=uev_fixed_alpha,
        per_neuron_alphas=best_alphas if uev_fixed_alpha is None else None,
    )

    # Conform to your previous tuple signature
    # These are placeholders you said you don't need now.
    F_statistics = None
    p_values = None
    bootstrap_p_values = None
    bootstrap_F_stats = None
    confidence_intervals = None
    negative_RSS_diff_counts = None

    return (
        beta,                    # (n_neurons x P)
        intercepts,              # (n_neurons,)
        explained_variances,     # (n_neurons,)
        UEV,                     # (n_neurons x len(unique_predictor_names))
        F_statistics,
        p_values,
        bootstrap_p_values,
        bootstrap_F_stats,
        confidence_intervals,
        negative_RSS_diff_counts,
    )