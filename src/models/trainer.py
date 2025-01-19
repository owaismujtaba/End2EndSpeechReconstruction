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
        print("🔧 Starting Model Training 🔧")
        print(f"🟢 Initial Data Shapes: X={X.shape}, y={y.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        print(f"📊 Training Data Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"📊 Test Data Shapes: X_test={X_test.shape}, y_test={y_test.shape}")

        history = self.model.train(X_train, y_train)
        print("✅ Model training completed")

        history_path = Path(self.model_dir, 'history.csv')
        history.to_csv(history_path)
        print(f"💾 Training history saved at: {history_path}")

        model.save(self.model_path)
        print(f"💾 Model saved at: {self.model_path}")

        self.evaluate_model(X_test, y_test)

    def evaluate_model(self, X, y):
        print("🔍 Evaluating Model 🔍")
        print(f"🟢 Input Data Shapes: X={X.shape}, y={y.shape}")

        predictions = self.model.predict(X)
        print(f"📊 Predictions Shape: {predictions.shape}")

        pcc_values = [np.corrcoef(predictions[:, i], y[:, i])[0, 1] for i in range(predictions.shape[1])]
        pcc_mean = np.nanmean(pcc_values)
        print(f"📊 Mean PCC: {pcc_mean}")

        np.save(str(Path(self.model_dir, 'pcc_values.npy')), np.array(pcc_values))
        print(f"💾 PCC values saved at: {str(Path(self.model_dir, 'pcc_values.npy'))}")
