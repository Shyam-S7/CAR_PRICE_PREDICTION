import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score,KFold

from src.exception import CustomException
from src.logger import logging


def evaluvate_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for name, model in models.items():
          
            cv=KFold(n_splits=5,shuffle=True,random_state=42)
            cv_scores = cross_val_score(
                model, x_train, y_train, cv=cv, scoring="r2", n_jobs=-1
            )
            train_score = np.mean(cv_scores)
            
            model.fit(x_train, y_train)

            ## train_pred = model.predict(x_train)
            ##  train_score = r2_score(np.expm1(y_train), np.expm1(train_pred))

            test_pred = model.predict(x_test)
            test_score = r2_score(np.expm1(y_test), np.expm1(test_pred))

            report[name] = {"train_score": train_score, "test_score": test_score}

            # report[name] = test_score

        return report

    except Exception as e:
        logging.info("Exception occured during model training")
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception Occured in save_object function utils")

        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Exception Occured in load_object function utils")

        raise CustomException(e, sys)
