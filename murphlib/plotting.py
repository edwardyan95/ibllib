
import numpy as np
import matplotlib.pyplot as plt
def plot_mean_psth(mean_psth, frame_rate, title, ax, vmin=None, vmax=None):
    
    num_frames_extracted = mean_psth.shape[1]
    # Time axis correction to center at 0
    time_axis = np.linspace(-num_frames_extracted/2, num_frames_extracted/2, num_frames_extracted) / frame_rate + 1/frame_rate/2
    start_time = time_axis[0]
    end_time = time_axis[-1]
    num_cells = mean_psth.shape[0]

    # Use the provided ax for plotting, include vmin and vmax for color normalization
    cax = ax.imshow(mean_psth, aspect='auto', origin='lower', cmap='viridis', 
                    extent=[start_time, end_time, 0, num_cells], vmin=vmin, vmax=vmax)

    fig = plt.gcf()
    fig.colorbar(cax, ax=ax, label='Z-score Normalized Activity')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cell Index')
    ax.set_title(f'{title}')

def plot_with_error_shading(data, time_points=None, ax=None, title=None, color='blue', ymin=None, ymax=None):
    """
    Plot the average response across trials with shaded standard error.

    Parameters:
    - data (numpy.ndarray): 2D array where the first dimension is trials and the second dimension is time points.
    - ax (matplotlib.axes.Axes, optional): Matplotlib axis object to plot on. If None, a new figure and axis will be created.
    - title (str, optional): Title for the plot. If None, no title will be set.
    - ymin (float, optional): Minimum value for the y-axis. If None, it will be determined automatically.
    - ymax (float, optional): Maximum value for the y-axis. If None, it will be determined automatically.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Compute the average response across trials
    mean_response = np.mean(data, axis=0)

    # Compute the standard error of the mean across trials
    std_error = np.std(data, axis=0) / np.sqrt(data.shape[0])

    # Time points
    if time_points is None:
        time_points = np.arange(data.shape[1])

    # Plot the mean response
    ax.plot(time_points, mean_response, color=color, label='Mean Response')

    # Shade the standard error
    ax.fill_between(time_points, mean_response - std_error, mean_response + std_error, color=color, alpha=0.3, label='Standard Error')

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Response')
    if title is not None:
        ax.set_title(title)

    # Set y-axis limits
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add grid lines
    #ax.grid(True)



def plot_roi_masks(ax=None, dim=(1024, 1024), roi_stat=None, roi_index=None, roi_data=None, vmin=None, vmax=None):
    """
    Plot ROI masks on an image.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to plot on. If None, a new figure will be created.
    dim (tuple): The dimensions of the image (default is 1024x1024).
    roi_stat (list): List containing ROI statistics (each element contains 'ypix' and 'xpix').
    roi_index (list): List of indices to select ROIs to plot.
    roi_data (list or np.ndarray): Data to plot for each ROI (must match the size of roi_index).
    vmin, vmax (float): Min and max values for scaling the color map.

    Returns:
    im (matplotlib.image.AxesImage): The image object, used for colorbar creation.
    """
    im = np.zeros(dim)
    if roi_index is None or roi_stat is None:
        print("index for rois or roi stat cannot be none!")
        return
    
    # Fill in the pixels for each ROI
    for idx in roi_index:
        ypix = roi_stat[idx]['ypix']
        xpix = roi_stat[idx]['xpix']
        if roi_data is not None:
            im[ypix, xpix] = roi_data[idx]
        else:
            im[ypix, xpix] = 1
    
    # Plot the image
    if ax is None:
        fig, ax = plt.subplots()
        im_obj = ax.imshow(im, vmin=vmin, vmax=vmax)
        plt.show()
    else:
        im_obj = ax.imshow(im, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    # Return the image object for colorbar creation
    return im_obj

def plot_aligned_traces(traces, plot_indices, labels, event_arrays=None, event_labels=None, figsize=(8, 6), trace_spacing=10):
    """
    Plots aligned traces from top to bottom with non-overlapping colors and even spacing,
    with the first trace at the top and labels on the right. Optionally, shades regions where event arrays have 1
    across all traces (i.e., one shading for the entire plot covering all traces).

    Parameters:
    traces (list of np.ndarray): List of numpy arrays containing the traces to plot. All arrays must have the same length.
    plot_indices (list or np.ndarray): Index of points to plot (applied to all traces).
    labels (list of str): List of labels for each trace.
    event_arrays (list of np.ndarray): Optional. List of arrays with 0 or 1 values, where 1 indicates event occurrence.
    event_labels (list of str): Optional. List of labels for the event arrays.
    figsize (tuple): Figure size, default is (8, 6).
    trace_spacing (float): The vertical spacing between each trace. Default is 10.

    Raises:
    ValueError: If input traces are not of the same length or if plot_indices do not match.
    """
    
    # Check if all traces are the same length
    trace_lengths = [len(trace) for trace in traces]
    if len(set(trace_lengths)) != 1:
        raise ValueError("All traces must have the same length.")
    
    # Ensure plot_indices is valid and within bounds
    trace_length = trace_lengths[0]
    if any(idx < 0 or idx >= trace_length for idx in plot_indices):
        raise ValueError("plot_indices must be within the range of the trace length.")
    
    # Ensure the number of labels matches the number of traces
    if len(labels) != len(traces):
        raise ValueError("Number of labels must match the number of traces.")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Generate non-overlapping colors for each trace
    colors = plt.cm.viridis(np.linspace(0, 1, len(traces)))

    # Generate event colors for shading, if provided
    if event_arrays is not None:
        event_colors = plt.cm.Set1(np.linspace(0, 1, len(event_arrays)))
    
    # Plot each trace from top to bottom
    for i, trace in enumerate(reversed(traces)):  # Reverse to plot first trace on top
        # Extract the subset of points to plot
        plot_data = trace[plot_indices]
        
        # Offset each trace by a fixed amount to ensure even spacing
        offset = i * trace_spacing
        
        # Plot the trace with its offset
        ax.plot(plot_indices, plot_data + offset, color=colors[len(traces) - 1 - i])

        # Add label on the right of the trace
        ax.text(plot_indices[-1] + 4, plot_data[-1] + offset, labels[len(traces) - 1 - i],
                va='center', fontsize=10, color=colors[len(traces) - 1 - i])

    # Shade the regions where event_arrays == 1, across all traces
    if event_arrays is not None:
        for j, event_array in enumerate(event_arrays):
            event_data = event_array[plot_indices]
            # Find regions where the event_array has value 1
            event_indices = np.where(event_data == 1)[0]
            
            if len(event_indices) > 0:
                # Group contiguous indices
                contiguous_blocks = np.split(event_indices, np.where(np.diff(event_indices) != 1)[0] + 1)
                
                # Shade each contiguous block
                for block in contiguous_blocks:
                    if len(block) > 0:
                        ax.fill_between(plot_indices[block], 
                                        (len(traces) - 1) * trace_spacing + 1,   # Upper boundary of shading
                                        -trace_spacing,  # Lower boundary of shading
                                        color=event_colors[j], alpha=0.3)
    
    # Add legend for event labels, if provided
    if event_arrays is not None and event_labels is not None:
        for j, event_label in enumerate(event_labels):
            ax.fill_between([], [], [], color=event_colors[j], alpha=0.3, label=event_label)
        ax.legend(loc='best')

    # Remove box, ticks, and tick labels
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.tight_layout()
    plt.show()