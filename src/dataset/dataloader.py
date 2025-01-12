import os
import numpy as np
import pandas as pd
import scipy
from pynwb import NWBHDF5IO
import warnings

import config as config
import pdb

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="")


class DataLoader:
    def __init__(self, subject_id='01'):
        print('Initializing DataLoader Class for subject:::', subject_id)
        self.subject_id = subject_id 
        self.read_data()
        
    def read_data(self):
        print('Reading data for subject::: ', self.subject_id)
        bids_path = config.BIDS_DIR
        io = NWBHDF5IO(
            os.path.join(
                bids_path, self.subject_id, 'ieeg', 
                f'{self.subject_id}_task-wordProduction_ieeg.nwb'),
            'r'
        )
        nwbfile = io.read()

        self.eeg = nwbfile.acquisition['iEEG'].data[:]
        self.audio = nwbfile.acquisition['Audio'].data[:]
        
        self.words = np.array(nwbfile.acquisition['Stimulus'].data[:], dtype=str)
        io.close()

        channels = pd.read_csv(
            os.path.join(bids_path, self.subject_id, 'ieeg', 
                         f'{self.subject_id}_task-wordProduction_channels.tsv'),
            delimiter='\t'
        )
        self.channels = np.array(channels['name'])
        print(f'Audio::{self.audio.shape}, iEEG::{self.eeg.shape}')
        print('Data read sucessfully')

class EegAudioFeatureExtractor:
    def __init__(self, eeg, audio):
        print('Initializing EegAudioFeatureExtractor')
        self.eeg_sr = config.EEG_SR
        self.eeg = eeg
        self.audio = audio

        self.load_audio_features()

    def load_audio_features(self):
        features = EegFeatures(self.eeg)

    

class EegFeatures:
    def __init__(self, eeg):
        print('Initializing EegFeatures Class')
        self.eeg = eeg
        self.eeg_sr = config.EEG_SR
        self.win_length = config.WIN_LENGTH
        self.frameshift = config.FRAME_SHIFT
        self.model_order = config.MODEL_ORDER
        self.step_size = config.STEP_SIZE

        self.preprocess_eeg()
        self.extract_features()

    def preprocess_eeg(self):
        print('Cleaning the EEG')
        eeg = scipy.signal.detrend(self.eeg, axis=0)
        sos = scipy.signal.iirfilter(4, [70 / (self.eeg_sr / 2), 170 / (self.eeg_sr / 2)], btype='bandpass', output='sos')
        eeg = scipy.signal.sosfiltfilt(sos, eeg, axis=0)

        for freq in [100, 150]:
            sos = scipy.signal.iirfilter(
                4, [(freq - 2) / (self.eeg_sr / 2), (freq + 2) / (self.eeg_sr / 2)],
                btype='bandstop', output='sos'
            )
            eeg = scipy.signal.sosfiltfilt(sos, eeg, axis=0)
        eeg = self.hilbert_transform(eeg)
        self.eeg = eeg
        print('EEG Cleaned Sucessfully')


    def extract_features(self):
        num_windows = int(np.floor(
            (self.eeg.shape[0] - self.win_length * self.eeg_sr) / (self.frameshift * self.eeg_sr))
        )
        
        feat = np.zeros((num_windows, self.eeg.shape[1]))

        for win in range(num_windows):
            start = int(np.floor((win * self.frameshift) * self.eeg_sr))
            stop = int(np.floor(start + self.win_length * self.eeg_sr))
            feat[win, :] = np.mean(self.eeg[start:stop, :], axis=0)

    @staticmethod
    def hilbert_transform(x):
        return scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)), axis=0)[:len(x)]

