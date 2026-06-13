## Reading data from a source
import os
import sys ##To use custom exception
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass ##Directly define the class variables without using __init__ method
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","data.csv")


'''THIS IS SIMILAR TO:dataclass shortens the code
class DataIngestionConfig:

    def __init__(
        self,
        train_data_path="artifacts/train.csv",
        test_data_path="artifacts/test.csv",
        raw_data_path="artifacts/data.csv"
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.raw_data_path = raw_data_path
        '''

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Enter the data ingestion method or component')
        try:
            df=pd.read_csv(r'notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_array,test_array,_=data_transformation.initiate_data_transformation(train_data,test_data)

    Modeltrainer=ModelTrainer()
    print(Modeltrainer.initiate_model_trainer(train_array,test_array))  ##r2 score