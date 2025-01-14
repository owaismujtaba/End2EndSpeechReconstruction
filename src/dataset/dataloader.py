from src.utils.utils import printSectionHeader, Colors
from src.dataset.data_reader import DataReader, EegAudioFeatureExtractor

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


       
    