import logging

def setup_logger(name='my_app', level=logging.INFO):
    """
    Sets up a generic logger with a console handler and a basic formatter.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding duplicate handlers if the logger is retrieved multiple times
    if not logger.handlers:
        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)

    return logger


# Create a logger instance
logger = setup_logger()
