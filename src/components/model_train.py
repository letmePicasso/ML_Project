import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object,evaluate_model
import os
import sys

import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and testing input")
            X_train, Y_train, X_test, Y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1],
            )
            models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
            }
            model_report:dict=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test
                                             ,models=models)

            
            return model_report
        except Exception as e:
            raise CustomException(e,sys)