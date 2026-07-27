import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        # Get the folder path from the full file path
        dir_path = os.path.dirname(file_path)

        # Create the folder if it doesn't already exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,Y_train,X_test,Y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,Y_train)
            Y_train_predict=model.predict(X_train)
            Y_test_predict=model.predict(X_test)
            train_model_score=r2_score(Y_train,Y_train_predict)
            test_model_score=r2_score(Y_test,Y_test_predict)
            report[list(models.keys())[i]]=test_model_score
        return report
    except Exception as e:
        raise CustomException(sys,e)
            