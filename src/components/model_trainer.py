import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    VotingRegressor,
    StackingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluvate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info(
                "Splitting Dependent and Independent variables from train and test data"
            )
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            estimators = [
                ("rf", RandomForestRegressor()),
                ("xg", XGBRegressor()),
                ("cat", CatBoostRegressor(verbose=False)),
                ("lgbm", LGBMRegressor()),
            ]
            models = {
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "StackingRegressor": StackingRegressor(
                    estimators=estimators, final_estimator=XGBRegressor()
                ),
                "VotingRegressor": VotingRegressor(estimators=estimators),
                "Random Forest Regressor": RandomForestRegressor(),
                "LGBMRegressor": LGBMRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Gradient boost": GradientBoostingRegressor(),
            }

            model_report = evaluvate_model(x_train, y_train, x_test, y_test, models)

            print(model_report)
            print("\n=================================")
            logging.info(f"Model Report:{model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model:{best_model_name},R2 Score:{best_model_score}")
            print("\n======================================")
            logging.info(f"Best Model:{best_model_name},R2 Score:{best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

        except Exception as e:
            logging.info("Exception occured at Model taining")
            raise CustomException(e, sys)
