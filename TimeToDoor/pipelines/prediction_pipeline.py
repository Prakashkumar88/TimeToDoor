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
        Vehicle_condition: int,
        Type_of_order: str,
        Type_of_vehicle: str,
        multiple_deliveries: int,
        Festival: str,
        City: str,
        TimeOrder_Hour: int,
        Delivery_city: str,
        distance: float
    ):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival
        self.City = City
        self.TimeOrder_Hour = TimeOrder_Hour
        self.Delivery_city = Delivery_city
        self.distance = distance

    def get_data_as_dataframe(self):
        import pandas as pd
        custom_data_dict = {
            "Delivery_person_Age": [self.Delivery_person_Age],
            "Delivery_person_Ratings": [self.Delivery_person_Ratings],
            "Weather_conditions": [self.Weather_conditions],
            "Road_traffic_density": [self.Road_traffic_density],
            "Vehicle_condition": [self.Vehicle_condition],
            "Type_of_order": [self.Type_of_order],
            "Type_of_vehicle": [self.Type_of_vehicle],
            "multiple_deliveries": [self.multiple_deliveries],
            "Festival": [self.Festival],
            "City": [self.City],
            "TimeOrder_Hour": [self.TimeOrder_Hour],
            "Delivery_city": [self.Delivery_city],
            "distance": [self.distance]
        }
        return pd.DataFrame(custom_data_dict)
