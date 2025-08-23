import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import numpy as np


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:

            preprocessor_path = os.path.join("artifacts", "preprocessing.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # take the input and apply  scaling
            data_scaled = preprocessor.transform(features)

            # pred on scaled data
            pred_log = model.predict(data_scaled)
 
            pred=np.expm1(pred_log)
            
            # final prediction it should be send app.py
            return pred

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e, sys)


# to collect user inputs and store it structured way
class CustomData:

    def __init__(
        self,
        brand: str,
        model: str,
        color: str,
        year: int,
        power_kw: float,
        power_ps: float,
        transmission_type: str,
        fuel_type: str,
        fuel_consumption_l_100km: float,
        fuel_consumption_g_km: float,
        mileage_in_km: float,
    ):
        # stores values in obj itself
        self.brand = brand
        self.model = model
        self.color = color
        self.year = year
        self.power_kw = power_kw
        self.power_ps = power_ps
        self.transmission_type = transmission_type
        self.fuel_type = fuel_type
        self.fuel_consumption_l_100km = fuel_consumption_l_100km
        self.fuel_consumption_g_km = fuel_consumption_g_km
        self.mileage_in_km = mileage_in_km

    # stored  inputs convert into df then model used it predict
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "brand": [self.brand],
                "model": [self.model],
                "color": [self.color],
                "year": [self.year],
                "power_kw": [self.power_kw],
                "power_ps": [self.power_ps],
                "transmission_type": [self.transmission_type],
                "fuel_type": [self.fuel_type],
                "fuel_consumption_l_100km": [self.fuel_consumption_l_100km],
                "fuel_consumption_g_km": [self.fuel_consumption_g_km],
                "mileage_in_km": [self.mileage_in_km],
            }
            df = pd.DataFrame(custom_data_input_dict)  # dict to df
            logging.info("Dataframe Gathered")
            return df  # THIS DF IS GIVEN TO PREDICT PIPE AS FEATURE PARAMETER

        except Exception as e:
            logging.info("Exception Occured in prediction pipeline")
            raise CustomException(e, sys)
