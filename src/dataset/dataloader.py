from sklearn.model_selection import train_test_split



from src.utils.utils import printSectionHeader, Colors, normalize_z_score
from src.dataset.data_reader import DataReader, EegAudioFeatureExtractor

import pdb

class DataLoader:
    def __init__(self, subject_id='sub-01'):
        printSectionHeader(f'{Colors.OKBLUE}🗂️ Initializing DataLoader Class for subject:::{Colors.ENDC} {subject_id}')
        self.subject_id = subject_id
        self.read_data()

    def read_data(self):
        data_reader = DataReader(subject_id=self.subject_id)
        eeg, audio = data_reader.eeg, data_reader.audio
        feature_extractor = EegAudioFeatureExtractor(
            eeg = eeg,
            audio = audio
        )
        self.eeg_features = feature_extractor.eeg_features
        self.audio_features = feature_extractor.audio_features
  

def dataset_loader_pipeline(subject_id='sub-01'):
    dataset_loader = DataLoader(subject_id=subject_id)
    eeg_features = normalize_z_score(dataset_loader.eeg_features)

    return eeg_features, dataset_loader.audio_features
