import pandas as pd
import numpy as np
from scipy import interpolate, stats
from scipy.stats import zscore
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, MultiTaskLasso, MultiTaskElasticNet
from joblib import Parallel, delayed
from multiprocessing import Pool
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

    df_transformed.loc[np.isnan(df_transformed['omission']),'omission'] = 0
    
    # Add 'leftchoice' column: 1 when 'choice' is -1, NaN otherwise
    df_transformed['leftchoice'] = np.where(df_transformed['choice'] == -1, 1, 0)

    # Add 'rightchoice' column: 1 when 'choice' is 1, NaN otherwise
    df_transformed['rightchoice'] = np.where(df_transformed['choice'] == 1, 1, 0)

    # Add 'reward_times' column: 'feedback_times' where 'feedbackType' is 1, NaN otherwise
    df_transformed['reward_times'] = np.where((df_transformed['feedbackType'] == 1) & (df_transformed['omission'] == 0), df_transformed['feedback_times'], np.nan)

    # Add 'punish_times' column: 'feedback_times' where 'feedbackType' is -1, NaN otherwise
    df_transformed['punish_times'] = np.where((df_transformed['feedbackType'] == -1) & (df_transformed['omission'] == 0), df_transformed['feedback_times'], np.nan)
    df_transformed['omission_times'] = np.where((df_transformed['feedbackType']==1) & (df_transformed['omission'] == 1), df_transformed['feedback_times'], np.nan)
    # Add 'visual_trial' column: 1 where 'modality' is 0, 0 elsewhere
    df_transformed['visual_trial'] = np.where(df_transformed['modality'] == 0, 1, 0)

    # Add 'aud_trial' column: 1 where 'modality' is 1, 0 elsewhere
    df_transformed['aud_trial'] = np.where(df_transformed['modality'] == 1, 1, 0)

    df_transformed['vis_reward_times'] = np.where((df_transformed['modality'] == 0) & (df_transformed['feedbackType'] == 1) & (df_transformed['omission'] == 0), df_transformed['feedback_times'], np.nan)
    df_transformed['aud_reward_times'] = np.where((df_transformed['modality'] == 1) & (df_transformed['feedbackType'] == 1) & (df_transformed['omission'] == 0), df_transformed['feedback_times'], np.nan)

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
    n_jobs=-1
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
    # def calculate_vif(X):
    #     vif_stats = []
    #     for i in range(X.shape[1]):
    #         # Get R² from regressing feature i on all other features
    #         other_cols = list(range(X.shape[1]))
    #         other_cols.pop(i)
            
    #         reg = LinearRegression()
    #         reg.fit(X[:, other_cols], X[:, i])
    #         r2 = reg.score(X[:, other_cols], X[:, i])
            
    #         # Calculate VIF
    #         vif = 1 / (1 - r2) if r2 != 1 else float('inf')
    #         vif_stats.append(vif)
    #     return np.array(vif_stats)
    
    # # Calculate VIF for each predictor
    # vif_values = calculate_vif(design_matrix)
    # high_vif_idx = np.where(vif_values > 5)[0]  # VIF > 5 indicates potential collinearity
    # if len(high_vif_idx) > 0:
    #     warnings.warn(f"High collinearity detected for predictors at indices {high_vif_idx} "
    #                  f"with VIF values {vif_values[high_vif_idx]}")
    
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
    
    bootstrap_results = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_iteration)(
            i, F, design_matrix, kf, unique_predictors, full_predictors, RegModel, block_size
        ) for i in range(n_bootstraps)
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