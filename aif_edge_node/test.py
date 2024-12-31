# HOW TO USE LOG:
# import logging.handlers
# log = logging.getLogger("my_app")
# setup_logging(enable_file_logging=False)
# logging.basicConfig(level="INFO")

# logger.debug("debug message", extra={"x": "hello"})
# logger.info("info message")
# logger.warning("warning message")
# logger.error("error message")
# logger.critical("critical message")
# logger.exception("exception message")

'''
def setup_logging(enable_file_logging=False):
    config_file = pathlib.Path("logging/config.json")
    with open(config_file) as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)
    queue_handler = logging.getHandlerByName("queue_handler")

    if not enable_file_logging:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)

    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
'''