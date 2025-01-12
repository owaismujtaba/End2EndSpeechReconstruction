from src.dataset.dataloader import DataLoader, EegAudioFeatureExtractor

import pdb
if __name__=='__main__':
    dataloader = DataLoader(subject_id='sub-01')
    eeg, audio = dataloader.eeg, dataloader.audio

    features = EegAudioFeatureExtractor(
        eeg=eeg,
        audio=audio
    )