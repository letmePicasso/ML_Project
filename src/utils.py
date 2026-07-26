import os
import sys
import pickle
from src.exception import CustomException

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