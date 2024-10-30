import logging
from logging.handlers import RotatingFileHandler
import os
from settings import Config

# Define whether logging should be enabled
is_logging = True  # Set to False to disable logging

conf = Config()
# Log file settings
LOG_FILE = f"{conf.LOG_DIR}/app.log"
LOG_LEVEL = logging.INFO

def setup_logging():
    """Configure logging for the entire application."""
    if not is_logging:
        # Disable all logging if is_logging is False
        logging.disable(logging.CRITICAL)
        return
    
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)
    
    # Create a rotating file handler to store logs in a file
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2, encoding='utf-8')
    file_handler.setLevel(LOG_LEVEL)
    
    # Create a stream handler to log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    
    # Define the format for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logging.info("Logging setup complete")

# Call the function to set up logging when the module is imported
setup_logging()