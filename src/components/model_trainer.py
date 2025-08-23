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
                ("Gradientboost", GradientBoostingRegressor()),
                ("xg", XGBRegressor()),
                ("cat", CatBoostRegressor(verbose=False)),
                ("lgbm", LGBMRegressor()),
            ]
            
            
            logging.info("Model building started")
            
            models = {
                "Random Forest Regressor": RandomForestRegressor(
                    max_depth=15,
                    min_samples_leaf=1,
                    min_samples_split=5,
                    n_estimators=500,
                    random_state=42,
                    max_features='sqrt',
                    n_jobs=-1
                ),
                 "VotingRegressor": VotingRegressor(estimators=estimators),
                "GradientBoostingRegressor": GradientBoostingRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.1
                ),
                "XGBRegressor": XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=7,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1
                ),
                "CatBoosting Regressor": CatBoostRegressor(
                    depth=6,
                    iterations=500,
                    l2_leaf_reg=1,
                    learning_rate=0.1,
                    verbose=False,
                
                ),
                "LGBMRegressor": LGBMRegressor(
                    learning_rate=0.1,
                    max_depth=-1,
                    n_estimators=200,
                    num_leaves=31,
                    force_col_wise=True,
                    lambda_1=0.5,
                    lambda_2=1.0,
                    n_jobs=-1,
                    min_child_weight=5,
                    objective="regression",
                    metric="rmse",
                    verbose = -1,
                    models = {
                "Random Forest Regressor": RandomForestRegressor(
                    max_depth=15,
                    min_samples_leaf=1,
                    min_samples_split=5,
                    n_estimators=500,
                    random_state=42,
                    max_features='sqrt',
                    n_jobs=-1
                ),
                 "VotingRegressor": VotingRegressor(estimators=estimators),
                "GradientBoostingRegressor": GradientBoostingRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.1
                ),
                "XGBRegressor": XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=7,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1
                ),
                "CatBoosting Regressor": CatBoostRegressor(
                    depth=6,
                    iterations=500,
                    l2_leaf_reg=1,
                    learning_rate=0.1,
                    verbose=False,
                
                ),
                "LGBMRegressor": LGBMRegressor(
                    learning_rate=0.1,
                    max_depth=-1,
                    #n_estimators=200,
                    num_leaves=31,
                    force_row_wise=True,
                    lambda_1=0.5,
                    lambda_2=1.0,
                    n_jobs=-1,
                    min_child_weight=5,
                    objective="regression",
                    metric="rmse",
                    verbose = -1,
                    n_estimators=10000,
                    early_stopping_rounds=100


                ),
                "StackingRegressor": StackingRegressor(
                   estimators=estimators, final_estimator=XGBRegressor()
                 ),
            }

            model_report = evaluvate_model(x_train, y_train, x_test, y_test, models)

            print(model_report)
            print("\n=================================")
            logging.info(f"Model Report:{model_report}")
            """
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            """

            # best_model_name = max(model_report, key=lambda k: model_report[k]["r2"])
            # best_model_score = model_report[best_model_name]["r2"]

            best_model_name = max(
                model_report, key=lambda k: model_report[k]["test_score"]
            )
            best_model_score = model_report[best_model_name]["test_score"]

            # print(f"Best Model: {best_model_name} with Test R²: {best_model_score}")

            best_model = models[best_model_name]

            print(f"Best Model:{best_model_name},R2 Score:{best_model_score}")
            print("\n======================================")
            logging.info(f"Best Model:{best_model_name},R2 Score:{best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
               


            model_report = evaluvate_model(x_train, y_train, x_test, y_test, models)

            print(model_report)
            print("\n=================================")
            logging.info(f"Model Report:{model_report}")
            """
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            """

            # best_model_name = max(model_report, key=lambda k: model_report[k]["r2"])
            # best_model_score = model_report[best_model_name]["r2"]

            best_model_name = max(
                model_report, key=lambda k: model_report[k]["test_score"]
            )
            best_model_score = model_report[best_model_name]["test_score"]

            # print(f"Best Model: {best_model_name} with Test R²: {best_model_score}")

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
