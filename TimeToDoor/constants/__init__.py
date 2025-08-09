# Auto-generated file.
import os, sys
from datetime import datetime

# artifact -> pipelien folder -> timestamp -> output

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

ROOT_DIR_KEY = os.getcwd()
DATA_DIR = "Data"
DATA_DIR_KEY = "finalTrain.csv"

#new_machine_learning/DATA_DIR/DATA_DIR_KEY

ARTIFACT_DIR_KEY = "Artifact"

# Data ingestion related variable
DATA_INGESTION_KEY = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR = "raw_data_dir"
DATA_INGESTION_INGESTED_DATA_DIR_KEY = "ingested_dir"
RAW_DATA_DIR_KEY = "raw.csv"
TRAIN_DATA_DIR_KEY = "train.csv"
TEST_DATA_DIR_KEY = "test.csv"

#data transformation related variable

DATA_TRANSFORMATION_ARTIFACT = "data_transformation"
DATA_PREPROCED_DIR = "proceor"
DATA_TRANSFORMATION_PROCESSING_OBJ = "processor.pkl"
DATA_TRANSFORMATION_DIR = "transformation"
TRANSFORM_TRAIN_DIR_KEY = "transformed_train.csv"
TRANSFORM_TEST_DIR_KEY = "transformed_test.csv"

# artifact / data_transformation / procceor->processor.pkl abnd transformation -> train.csv and test.csv
#model training

MODEL_TRAINING_KEY = "model_trainer"
MODEL_OBJECT = "model.pkl"