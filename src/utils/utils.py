import numpy as np
import config as config

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printSectionHeader(message):
    """
    Print a formatted section header.

    Args:
        message (str): The message to be displayed in the header.
    """
    print("\n" + "=" * config.TERMINAL_WIDTH)
    print(f'{message.center(config.TERMINAL_WIDTH)}')
    print("=" * config.TERMINAL_WIDTH)


def normalize_z_score(array):
    """
    Normalize a NumPy array with a z-score.
    
    Parameters:
        array (numpy.ndarray): Input array of shape (N, 9, 127).
        
    Returns:
        numpy.ndarray: Z-score normalized array with the same shape as the input.
    """
    print('Normalizing EEG using Z-Score')
    mean = np.mean(array, axis=-1, keepdims=True)
    std = np.std(array, axis=-1, keepdims=True)
    
    normalized_array = (array - mean) / (std + 1e-8)  
    
    return normalized_array