import os
import logging

class Logger:
    def __init__(self, log_file):
        """
        Initializes the logger to log information to a file and the console.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create file handler to log to file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler to log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Define the log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, message):
        """
        Log a message at INFO level.
        """
        self.logger.info(message)

    def close(self):
        """
        Close the logger and its handlers.
        """
        logging.shutdown()
