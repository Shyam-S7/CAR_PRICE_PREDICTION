import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


from src.exception import CustomException
from src.logger import logging


def evaluvate_model(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]

            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)


            test_score=r2_score(y_test,y_pred)

            report[list(model).keys()[i]]=test_score

            return report
        
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
        

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(file_path,exist_ok=True)

        with open (file_path,"wb"):
            pickle.dump(obj,file_path)
    
    except Exception as e:
        logging.info('Exception Occured in save_object function utils')

        raise CustomException(e,sys)
    



def load_object(file_path):
    try:
        with open (file_path,"rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info('Exception Occured in load_object function utils')

        raise CustomException(e,sys)















