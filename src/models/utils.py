
import pandas as pd
from pathlib import Path
import tensorflow as tf
import config as config
from src.dataset.dataloader import dataset_loader_pipeline
from src.models.models import NeuroInceptDecoder
from src.models.trainer import ModelTrainer
from src.dataset.data_reader import EegAudioFeatureExtractor
from src.dataset.dataloader import DataLoader
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

def vcoder_training_pipepline(subject_id='sub-01'):
    dataset_loader = DataLoader(subject_id)
    feature_extractor = EegAudioFeatureExtractor(
        eeg=dataset_loader.__annotations__
    )

def train_decoder_on_all_subjects():
    printSectionHeader('Training Decoder on All Subjects')
    pcc_means = []
    subject_ids = []
    for index in range(1, config.N_SUBJECTS):
        subject_id = f'sub-0{index}'
    
        pcc_mean = decoder_training_pipeline(subject_id)
        pcc_means.append(pcc_mean)
        subject_ids.append(subject_id)

    
    results = {'sub':subject_ids, 'PCC':pcc_means}
    results = pd.DataFrame(results)
    results.to_csv(Path(config.TRAINED_DIR, 'Decoder_PCC.csv'))





def pcc_loss(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Input shapes must match"
    
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    
    mean_pred = tf.reduce_mean(y_pred)
    mean_true = tf.reduce_mean(y_true)
    
    covariance = tf.reduce_sum((y_pred - mean_pred) * (y_true - mean_true))
    
    std_pred = tf.sqrt(tf.reduce_sum((y_pred - mean_pred) ** 2))
    std_true = tf.sqrt(tf.reduce_sum((y_true - mean_true) ** 2))
    
    pcc = covariance / (std_pred * std_true)
    
    loss = -pcc
    return loss
