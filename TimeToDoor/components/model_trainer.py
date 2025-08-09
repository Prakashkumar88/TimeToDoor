from TimeToDoor.constants import *
from TimeToDoor.logger import logging
from TimeToDoor.exceptions import CustomException
from TimeToDoor.config.configuration import *
from TimeToDoor.utils import evaluate_model, save_obj
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import os, sys

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

class ModelTrainerConfig:
    train_model_file_path = MODEL_FILE_PATH
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig
        
    def initiate_model_training(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1],test_array[:,:-1], test_array[:,-1])
            
            models = {
                "XGBRegressor": XGBRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "SVR": SVR(),
                "GradientBoostingRegressor": GradientBoostingRegressor()
            }
            
            model_report : dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            print(f"Best model found: {best_model_name} with score: {best_model_score}, R2 Score: {best_model_score}")
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}, R2 Score: {best_model_score}")
            
            
            save_obj(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)
        except Exception as e:
            raise CustomException(e, sys)