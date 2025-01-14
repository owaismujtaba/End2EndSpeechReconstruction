import os
import numpy as np
import pandas as pd
import scipy
from pynwb import NWBHDF5IO
import warnings

import config as config
from src.utils.mel_filter_bank import MelFilterBank
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

        self.load_eeg_features()

    def load_eeg_features(self):
        print('Loading AUDIO and EEG Features')
        eeg_feature_extractor = EegFeatures(self.eeg)
        self.eeg_features = eeg_feature_extractor.features
        audio_feature_extractor = AudioFeatures(self.audio)
        self.audio_features = audio_feature_extractor.feataures

        if self.audio_features.shape[0] != self.eeg_features.shape[0]:
            t_len = min(self.audio_features.shape[0], self.eeg_features.shape[0])
            self.audio_features = self.audio_features[:t_len, :]
            self.eeg_features = self.eeg_features[:t_len, :]


class AudioFeatures:
    def __init__(self, audio):
        print('Initializing AudioFeatures Class')
        self.audio = audio
        self.audio_sr = config.AUDIO_SR
        self.win_length = config.WIN_LENGTH
        self.frameshift = config.FRAME_SHIFT
        self.model_order = config.MODEL_ORDER
        self.step_size = config.STEP_SIZE

        self.preprocess_audio()
        self.segment_and_extract_mel_spectrograms()

    def preprocess_audio(self):
        print('Preprocessing Audio')
        audio = scipy.signal.decimate(self.audio, int(self.audio_sr/config.TARGET_AUDIO_SR))
        audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
        self.audio_sr = config.TARGET_AUDIO_SR
        self.audio = audio
        print('Preprocessing completed')

    def segment_and_extract_mel_spectrograms(self):
        print('Segementing and extrating spectrograms')
        audio = self.audio
        num_windows = int(
            np.floor((audio.shape[0] - self.win_length * self.audio_sr)
                      / (self.frameshift * self.audio_sr)
            )
        )
        win = scipy.signal.get_window('hann', int(self.win_length * self.audio_sr), fftbins=False)
        spectrogram = np.zeros((num_windows, int(np.floor(self.win_length * self.audio_sr / 2 + 1))), dtype='complex')

        for w in range(num_windows):
            start_audio = int(np.floor((w * self.frameshift) * self.audio_sr))
            stop_audio = int(np.floor(start_audio + self.win_length * self.audio_sr))
            a = audio[start_audio:stop_audio]
            spec = np.fft.rfft(win * a)
            spectrogram[w, :] = spec

        mfb = MelFilterBank(spectrogram.shape[1], config.NO_FORMANTS, self.audio_sr)
        spectrogram = np.abs(spectrogram)
        spectrograms = mfb.toLogMels(spectrogram).astype('float')
        self.feataures = spectrograms
        print("Segmentation and spectrogram extraction completed")

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
        self.segment_and_extract_features()
        self.add_temporal_context()

    def preprocess_eeg(self):
        print('Preprocessing  EEG')
        eeg = scipy.signal.detrend(self.eeg, axis=0)
        sos = scipy.signal.iirfilter(4, [70 / (self.eeg_sr / 2), 170 / (self.eeg_sr / 2)], btype='bandpass', output='sos')
        eeg = scipy.signal.sosfiltfilt(sos, eeg, axis=0)

        for freq in [100, 150]:
            sos = scipy.signal.iirfilter(
                4, [(freq - 2) / (self.eeg_sr / 2), (freq + 2) / (self.eeg_sr / 2)],
                btype='bandstop', output='sos'
            )
            eeg = scipy.signal.sosfiltfilt(sos, eeg, axis=0)
        self.eeg = eeg
        print('Preprocessing Sucessfully')


    def segment_and_extract_features(self):
        print('Segemnting the EEG and extracting features')
        eeg = self.hilbert_transform(self.eeg)
        num_windows = int(np.floor(
            (eeg.shape[0] - self.win_length * self.eeg_sr) / (self.frameshift * self.eeg_sr))
        )
        feat = np.zeros((num_windows, eeg.shape[1]))
        
        # segement the eeg and store the mean of the segment in every channel
        # as a feature 
        for win in range(num_windows):
            start = int(np.floor((win * self.frameshift) * self.eeg_sr))
            stop = int(np.floor(start + self.win_length * self.eeg_sr))
            feat[win, :] = np.mean(eeg[start:stop, :], axis=0)
        self.segmented_features = feat
        print('EEG segemntation and feature extraction completed')

    def add_temporal_context(self):
        print(f'Adding temporal context with model order {config.MODEL_ORDER}, step size {config.STEP_SIZE}')
        features = self.segmented_features
        num_windows = features.shape[0] - (2 * self.model_order * self.step_size)
        feat_stacked = np.zeros((num_windows, (2 * self.model_order + 1) * features.shape[1]))

        for i in range(self.model_order * self.step_size, features.shape[0] - self.model_order * self.step_size):
            ef = features[i - self.model_order * self.step_size:i + self.model_order * self.step_size + 1:self.step_size, :]
            feat_stacked[i - self.model_order * self.step_size, :] = ef.flatten()

        self.features = feat_stacked
        print('Adding temporal context completed')

    @staticmethod
    def hilbert_transform(x):
        return scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)), axis=0)[:len(x)]

