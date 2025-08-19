import os
import sys
from src.exception import CustomException
from src.logger import logging


from src.components.data_injestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initaite_data_transformation(
        train_data_path, test_data_path
    )

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)
