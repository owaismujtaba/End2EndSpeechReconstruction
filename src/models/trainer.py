import os
import pandas as pd
from pathlib import Path
import pdb

from sklearn.metrics import classification_report
from src.utils.utils import  printSectionHeader
import config as config


class ModelTrainer:
    def __init__(self, model_name, val_size=0.15):
        printSectionHeader("📊 Initializing ModelTrainer Class ")
        self.name = model_name
        self.val_size = val_size
        self.dir = config.TRAINED_DIR
        self.model_dir = Path(self.dir, 'Models', model_name)
        self.model_path = Path(self.model_dir, model_name)
        os.makedirs(self.model_dir, exist_ok=True)

        
        print("✅ ModelTrainer Initialization Complete ✅")

    def train_model(self, model, X, y):
        print("易 Training Model 易")
        
        history = model.train(X, y)
        
        history_path = Path(self.model_dir, 'history.csv')
        pdb.set_trace()
        history.to_csv(history_path)
        model.save(self.model_path)
        print("✅ Model Training Complete ✅")

    
