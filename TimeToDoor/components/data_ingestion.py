from TimeToDoor.constants import *
from TimeToDoor.config.configuration import *
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from TimeToDoor.logger import logging
from TimeToDoor.exceptions import CustomException
from TimeToDoor.components.data_transformation import DataTransformation, DataTransformationConfig
from TimeToDoor.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str = TRAIN_FILE_PATH
    test_data_path:str = TEST_FILE_PATH
    raw_data_path:str = RAW_FILE_PATH


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def iniitiate_data_ingestion(self):
        try:
            df = pd.read_csv(DATASET_PATH)

            #df = pd.read_csv(os.path.join('C:\Users\shiva\Desktop\project_template\New-Machine-Learning-Modular-Coding-project\Data\finalTrain.csv'))

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path, index = False)

            train_set, test_set = train_test_split(df, test_size = 0.20, random_state= 42)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True )
            train_set.to_csv(self.data_ingestion_config.train_data_path, header = True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path),exist_ok=True )
            test_set.to_csv(self.data_ingestion_config.test_data_path, header = True)

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path

            )


        except Exception as e:
            raise CustomException( e, sys)
# Data Ingestion

if __name__ == "__main__":
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
