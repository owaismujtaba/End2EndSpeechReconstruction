from pathlib import Path
import pandas as pd

from sklearn.metrics import classification_report
from src.utils.utils import printSectionFooter, printSectionHeader
import config as config


class ModelTrainer:
    def __init__(self, 
            model_name, destination_dir,
            validationSize=0.15, randomState=42):
        printSectionHeader(" Initializing ModelTrainer ")
        self.name = model_name
        self.destination=destination_dir
        self.classificationReport = None
        self.validationSize = validationSize
        self.randomState = randomState
        
        printSectionFooter("✅ ModelTrainer Initialization Complete ✅")

    def trainModel(self, model, X, y):
        printSectionHeader("易 Training Model 易")
        
        history = model.train(X, y)
        
        modelpath = Path(self.destination, f'{self.name}.h5')
        historypath = Path(self.destination, 'history.csv')
        history.history.to_csv(historypath)
        model.save(modelpath)
        print("✅ Model Training Complete ✅")

    def getClassificationReport(self):
        return self.classificationReport