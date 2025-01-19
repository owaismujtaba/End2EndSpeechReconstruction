import os
import shutil
from pathlib import Path

# Directories
CURRENT_DIR = os.getcwd()
BIDS_DIR = Path(CURRENT_DIR, 'Data')
TRAINED_DIR= Path(CURRENT_DIR, 'Trained')

TERMINAL_WIDTH = shutil.get_terminal_size().columns

# Data Specification
EEG_SR = 1024
AUDIO_SR = 48000

# Audio and EEG features specifications
WIN_LENGTH = 0.05
FRAME_SHIFT = 0.01
MODEL_ORDER = 4
STEP_SIZE = 5
TARGET_AUDIO_SR = 16000
NO_FORMANTS = 128

# Training Parameters
EPOCHS = 100
BATCH_SIZE=128