from src.utils.utils import printSectionHeader, Colors
from src.dataset.data_reader import DataReader, EegAudioFeatureExtractor
from sklearn.model_selection import train_test_split

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

    def get_train_test_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.eeg_features, self.audio_features, 
            test_size=0.2, random_state=42
        )
       
        return X_train, y_train, X_test, y_test
    

def dataset_loader_pipeline(subject_id='sub-01'):
    dataset_loader = DataLoader(subject_id=subject_id)
    X_train, y_train, X_test, y_test = dataset_loader.get_train_test_data()

    return X_train, y_train, X_test, y_test
