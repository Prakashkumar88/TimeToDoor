# TimeToDoor/components/data_transformation.py
from TimeToDoor.constants import *
from TimeToDoor.logger import logging
from TimeToDoor.exceptions import CustomException
from TimeToDoor.config.configuration import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from TimeToDoor.utils import save_obj
import os, sys


class Feature_Engineering(BaseEstimator, TransformerMixin):
    """
    Custom transformer that computes distance and drops unwanted columns.
    Implements fit and transform so it behaves correctly in sklearn Pipelines.
    """
    def __init__(self):
        logging.info("Feature Engineering initialized")

    def distance_numpy(self, df, lat1, lon1, lat2, lon2):
        """Add 'distance' column (Haversine) to df (in km)."""
        p = np.pi / 180.0
        a = (
            0.5
            - np.cos((df[lat2] - df[lat1]) * p) / 2
            + np.cos(df[lat1] * p)
            * np.cos(df[lat2] * p)
            * (1 - np.cos((df[lon2] - df[lon1]) * p))
            / 2
        )
        df["distance"] = 12742 * np.arcsin(np.sqrt(a))  # 2 * R; R = 6371 km

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Do in-place transformations on a copy of df and return it.
        - drop ID
        - compute distance
        - drop raw lat/lon and other irrelevant columns
        """
        try:
            # operate on copy to avoid side effects
            df = df.copy()

            # safe checks for required columns
            required_cols = [
                "Restaurant_latitude",
                "Restaurant_longitude",
                "Delivery_location_latitude",
                "Delivery_location_longitude",
            ]
            for c in required_cols:
                if c not in df.columns:
                    raise KeyError(f"Required column missing for distance calc: {c}")

            if "ID" in df.columns:
                df.drop(["ID"], axis=1, inplace=True)

            # compute distance
            self.distance_numpy(
                df,
                "Restaurant_latitude",
                "Restaurant_longitude",
                "Delivery_location_latitude",
                "Delivery_location_longitude",
            )

            drop_cols = [
                "Delivery_person_ID",
                "Restaurant_latitude",
                "Restaurant_longitude",
                "Delivery_location_latitude",
                "Delivery_location_longitude",
                "Order_Date",
                "Time_Orderd",
                "Time_Order_picked",
            ]
            # only drop columns that exist
            drop_cols = [c for c in drop_cols if c in df.columns]
            if drop_cols:
                df.drop(drop_cols, axis=1, inplace=True)

            logging.info("Dropped columns and computed distance in FE transformer")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    # sklearn-compatible methods
    def fit(self, X, y=None):
        # no fitting required, but must return self
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        try:
            return self.transform_data(X)
        except Exception as e:
            raise CustomException(e, sys) from e


@dataclass
class DataTransformationConfig:
    processed_obj_file_path: str = PREPROCESSING_OBJ_FILE
    transform_train_path: str = TRANSFORMED_TRAIN_FILE_PATH
    transform_test_path: str = TRANSFORM_TEST_FILE_PATH
    feature_engg_obj_path: str = FEATURE_ENGG_OBJ_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Build ColumnTransformer that:
         - numeric pipeline: impute(0) -> standard scale
         - categorical pipeline: most_frequent impute -> one-hot -> scale(with_mean=False)
         - ordinal pipeline: impute -> ordinal encode (with provided categories) -> scale(with_mean=False)
        """
        try:
            Road_traffic_density = ["Low", "Medium", "High", "Jam"]
            Weather_conditions = ["Sunny", "Cloudy", "Fog", "Sandstorms", "Windy", "Stormy"]

            categorical_columns = ["Type_of_order", "Type_of_vehicle", "Festival", "City"]
            ordinal_encoder = ["Road_traffic_density", "Weather_conditions"]
            numerical_columns = [
                "Delivery_person_Age",
                "Delivery_person_Ratings",
                "Vehicle_condition",
                "multiple_deliveries",
                "distance",
            ]

            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            # Ordinal pipeline
            ordinal_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder(categories=[Road_traffic_density, Weather_conditions])),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns),
                    ("ordinal_pipeline", ordinal_pipeline, ordinal_encoder),
                ],
                remainder="drop",  # drop any columns not specified
            )

            logging.info("Data transformation (ColumnTransformer) object created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps=[("fe", Feature_Engineering())])
            return feature_engineering
        except Exception as e:
            raise CustomException(e, sys)

    def inititate_data_transformation(self, train_path: str, test_path: str):
        try:
            # read
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining FE (feature engineering) pipeline object")
            fe_obj = self.get_feature_engineering_object()

            # apply FE transformer (it is a pipeline with a single transformer)
            logging.info("Applying feature engineering to train and test dataframes")
            train_df = fe_obj.fit_transform(train_df)  # fit not required but allowed
            test_df = fe_obj.transform(test_df)

            # save intermediate copies (optional)
            train_df.to_csv("train_data_after_fe.csv", index=False)
            test_df.to_csv("test_data_after_fe.csv", index=False)

            # get preprocessing object (ColumnTransformer)
            processing_obj = self.get_data_transformation_object()

            target_column_name = "Time_taken (min)"
            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise KeyError(f"Target column '{target_column_name}' not found in train/test dataframes.")

            # split into X/y
            X_train = train_df.drop(columns=target_column_name, axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=target_column_name, axis=1)
            y_test = test_df[target_column_name]

            # fit_transform processing object on X_train and transform X_test
            logging.info("Fitting preprocessing (ColumnTransformer) on X_train and transforming X_train/X_test")
            X_train_transformed = processing_obj.fit_transform(X_train)
            X_test_transformed = processing_obj.transform(X_test)

            # combine with targets
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # save as DataFrames (optional)
            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_path), exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transform_train_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_path), exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transform_test_path, index=False, header=True)

            # save fitted preprocessing object and FE object
            # processing_obj is fitted, so save that
            save_obj(file_path=self.data_transformation_config.processed_obj_file_path, obj=processing_obj)
            # FE object (pipeline) - saving the transformer pipeline (it doesn't need to be fitted)
            save_obj(file_path=self.data_transformation_config.feature_engg_obj_path, obj=fe_obj)

            logging.info("Saved preprocessing and feature engineering objects to disk")

            return train_arr, test_arr, self.data_transformation_config.processed_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
