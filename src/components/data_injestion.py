import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods Starts")
        try:
            df = pd.read_csv(os.path.join("notebooks/data", "MODEL_DATA.csv"))
            logging.info("Dataset read as pandas Dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            df = df.drop_duplicates()

            logging.info("Train test split")

            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Ingestion of Data is completed")

            train = pd.read_csv("artifacts/train.csv")
            test = pd.read_csv("artifacts/test.csv")

            overlap = pd.merge(train, test, how="inner")
            logging.info(f"Duplicates across train & test: {len(overlap)}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise CustomException(e, sys)


"""
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data =obj.initiate_data_ingestion()

    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initaite_data_transformation(
        train_data, test_data
    )

    model_train = ModelTrainer()
    model_train.initiate_model_training(train_arr, test_arr)
    # python src/components/data_injestion.py
#python src/pipeline/train_pipeline.py 

#python src/pipeline/prediction_pipeline.py 

"""
