import json
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