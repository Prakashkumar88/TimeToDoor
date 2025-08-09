from TimeToDoor.constants import *
from TimeToDoor.logger import logging
from TimeToDoor.exceptions import CustomException
from TimeToDoor.config.configuration import *
from TimeToDoor.utils import load_model
import pandas as pd
import os, sys


class PredictionPipeline():
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = PREPROCESSING_OBJ_FILE
            model_path = MODEL_FILE_PATH
            
            preprocessor = load_model(preprocessor_path)
            model = load_model(model_path)
            
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
            
        except Exception as e:
            logging.info("Error occurred during prediction")
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 Delivery_person_Age: int,
                 Delivery_person_Ratings: float,
                 Weather_conditions: str,
                 Road_traffic_density: str,
                 Type_of_vehicle: str,
                 Vehicle_condition: str,
                 multiple_deliveries: int,
                 distance: float,
                 Type_of_order: str,
                 Festival: str,
                 City: str):
        
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Type_of_vehicle = Type_of_vehicle
        self.Vehicle_condition = Vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.distance = distance
        self.Type_of_order = Type_of_order
        self.Festival = Festival
        self.City = City
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age': [self.Delivery_person_Age],
                'Delivery_person_Ratings': [self.Delivery_person_Ratings],
                'Weather_conditions': [self.Weather_conditions],
                'Road_traffic_density': [self.Road_traffic_density],
                'Type_of_vehicle': [self.Type_of_vehicle],
                'Vehicle_condition': [self.Vehicle_condition],
                'multiple_deliveries': [self.multiple_deliveries],
                'distance': [self.distance],
                'Type_of_order': [self.Type_of_order],
                'Festival': [self.Festival],
                'City': [self.City]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data converted to DataFrame")
            return df
        
        except Exception as e:
            logging.info("Error occurred while converting custom data to DataFrame")
            raise CustomException(e, sys)
        
