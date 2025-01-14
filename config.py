import os
from pathlib import Path

CURRENT_DIR = os.getcwd()
BIDS_DIR = Path(CURRENT_DIR, 'Data')

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
