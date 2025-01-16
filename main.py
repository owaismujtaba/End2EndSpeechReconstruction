from src.dataset.dataloader import dataset_loader_pipeline
from src.models.models import NeuroInceptDecoder
from src.models.trainer import ModelTrainer

import pdb
if __name__=='__main__':

    X_train, y_train, X_test, y_test = dataset_loader_pipeline(subject_id='sub-01')
    model = NeuroInceptDecoder(
        n_classes=128,
        n_channels=127,
        n_features=9
    )

    trainer = ModelTrainer(
        model_name='NeuroInceptDecoder',
    )
    trainer.train_model(
        model=model,
        X=X_train,
        y=y_train
    )