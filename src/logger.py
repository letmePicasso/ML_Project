import logging
import os
from datetime import datetime

# Create the log file name using current date and time
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Create the path for the logs folder
logs_path = os.path.join(os.getcwd(), "logs")

# Create the logs folder if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Complete path of the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Test the logger
if __name__ == "__main__":
    print("Logger is running...")
    print(f"Logs Folder : {logs_path}")
    print(f"Log File    : {LOG_FILE_PATH}")

    logging.info("Logging has started.")

    print("Log file created successfully!")
    logging.info("Logging has started")