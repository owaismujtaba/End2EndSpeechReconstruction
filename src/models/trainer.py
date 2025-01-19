import os
import numpy as np
import pandas as pd
from pathlib import Path
import pdb

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from src.utils.utils import  printSectionHeader
import config as config


class ModelTrainer:
    def __init__(self, model_name, val_size=0.15):
        printSectionHeader("📊 Initializing ModelTrainer Class ")
        self.name = model_name
        self.val_size = val_size
        self.dir = config.TRAINED_DIR
        self.model_dir = Path(self.dir, 'Decoder', model_name)
        self.model_path = Path(self.model_dir, f'{model_name}.h5')
        os.makedirs(self.model_dir, exist_ok=True)

        
        print("✅ ModelTrainer Initialization Complete ✅")

    def train_model(self, model, X, y):
        self.model = model
        print("易 Training Model 易")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        history = self.model.train(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        history_path = Path(self.model_dir, 'history.csv')
        history.to_csv(history_path)
        model.save(self.model_path)
        print("✅ Model Training Complete ✅")

    def evaluate_model(self, X, y):
        pdb.set_trace()
        predictions = self.model.predict(X)
        pcc_values = [np.pearsonr(predictions[:, i], y[:, i])[0, 1] for i in range(predictions.shape[1])]
        pcc_mean = np.nanmean(pcc_values)
        np.save(Path(self.model, 'pcc_values.npy'), pcc_values)
        print(f'Mean PCC across {X.shape[1]} formants : {pcc_mean}')
        
