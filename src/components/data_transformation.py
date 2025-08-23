import sys
import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    RobustScaler,
)

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationconfig:
    preprocessing_obj_file_path = os.path.join("artifacts", "preprocessing.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    # preprocessing object
    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation initiated")

            categorical_features_lb = ["fuel_type", "brand", "model", "color"]
            categorical_features_one = ["transmission_type"]
            numeric_features = [
                "mileage_in_km",
                "power_kw",
                "power_ps",
                "fuel_consumption_l_100km",
                "fuel_consumption_g_km",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                ]
            )

            cat_lb_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinalencoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                    ("scaler", RobustScaler()),
                ]
            )
            cat_one_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("onehot_scaled", cat_one_pipeline, categorical_features_one),
                    ("ordinal_scaled", cat_lb_pipeline, categorical_features_lb),
                    ("numeric_scaled", num_pipeline, numeric_features),
                ],
                remainder="drop",
            )

            return preprocessor

            logging.info("Pipeline Completed")

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e, sys)

    def initaite_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head  : \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_obj()

            target_column_name = "price_in_euro"  # y
            drop_columns = [target_column_name]  # x

            # TRAIN

            input_feature_train_df = train_df.drop(
                columns=drop_columns, axis=1
            )  # xtrain
            target_feature_train_df = np.log1p(train_df[target_column_name])  # ytrain

            # TEST

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)  # xtest
            target_feature_test_df = np.log1p(test_df[target_column_name])  # ytest

            # X DATA

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )  # Xtrain

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )  # Xtest

            logging.info(
                "Applying preprocessing object on training and testing datasets."
            )

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessing_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Preprocessor pickle file saved")
            logging.info("Data Transformation Completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_file_path,
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e, sys)
