# Auto-generated file.
from TimeToDoor.constants import *
from TimeToDoor.logger import logging
from TimeToDoor.exceptions import CustomException
import os, sys
from TimeToDoor.config.configuration import *
from TimeToDoor.components.data_transformation import DataTransformation, DataTransformationConfig
from TimeToDoor.components.model_trainer import ModelTrainer, ModelTrainerConfig   
from TimeToDoor.components.data_ingestion import DataIngestion, DataIngestionConfig

class Train:
    def __init__(self):
        self.c = 0
        print(f"*****************{self.c}*****************")

    def main(self):
        print("🔹 Starting Data Ingestion...")
        obj = DataIngestion()
        train_data_path, test_data_path = obj.iniitiate_data_ingestion()
        print("✅ Data Ingestion completed.")

        print("🔹 Starting Data Transformation...")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.inititate_data_transformation(train_data_path, test_data_path)
        print("✅ Data Transformation completed.")

        print("🔹 Starting Model Training...")
        model_trainer = ModelTrainer()
        result = model_trainer.initiate_model_training(train_array=train_arr, test_array=test_arr)
        print("✅ Model Training completed. Result:", result)
