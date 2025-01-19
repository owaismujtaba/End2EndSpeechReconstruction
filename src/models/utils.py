
import pandas as pd
from pathlib import Path

import config as config
from src.dataset.dataloader import dataset_loader_pipeline
from src.models.models import NeuroInceptDecoder
from src.models.trainer import ModelTrainer
from src.utils.utils import printSectionHeader

import pdb

def decoder_training_pipeline(subject_id='sub-01'):
    printSectionHeader(f'Starting decoder training on {subject_id}')
    X_train, y_train = dataset_loader_pipeline(subject_id)
    model = NeuroInceptDecoder(
            n_classes=128,
            n_channels=127,
            n_features=9
    )

    trainer = ModelTrainer(
            model_name='NeuroInceptDecoder',
            subject_id = subject_id
    )
    trainer.train_model(
            model=model,
            X=X_train,
            y=y_train
    )

    return trainer.pcc_mean


def train_decoder_on_all_subjects():
    printSectionHeader('Training Decoder on All Subjects')
    pcc_means = []
    subject_ids = []
    for index in range(1, 4):
        subject_id = f'sub-0{index}'
    
        pcc_mean = decoder_training_pipeline(subject_id)
        pcc_means.append(pcc_mean)
        subject_ids.append(subject_id)

    
    results = {'sub':subject_ids, 'PCC':pcc_means}
    results = pd.DataFrame(results)
    results.to_csv(Path(config.TRAINED_DIR, 'Decoder_PCC.csv'))