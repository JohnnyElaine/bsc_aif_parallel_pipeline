import logging

def setup_logging(name: str, log_to_file=False, log_file_path=None):
    # Define a log format
    log_format = "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Set up the root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        datefmt=date_format,
    )

    # Get a logger instance
    logger = logging.getLogger(name)

    # If file logging is enabled, add a file handler
    '''
    if log_to_file:
        if log_file_path is None:
            log_file_path = GlobalVariables.PROJECT_ROOT / 'log' / 'worker.log'

        log_file = Path(log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
        file_handler.setLevel(logging.INFO)  # Log INFO and above to the file
        logger.addHandler(file_handler)
    '''

    return logger