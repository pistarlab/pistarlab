import logging
import os
import platform
import warnings
from logging import StreamHandler


class RedisHandler(StreamHandler):

    def __init__(self, context, redis_client):
        StreamHandler.__init__(self)
        self.redis_client = redis_client
        self.context = context

    def emit(self, record):
        msg = self.format(record)
        try:
            self.redis_client.publish(f"PISTARLAB_LOGS_{self.context}", msg)
        except:
            pass


DEFAULT_LOG_FORMAT = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] [%(module)s] %(message)s")


def setup_default_logging(level):
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').disabled = True
    logging.getLogger('filelock').setLevel(logging.ERROR)
    rootLogger = logging.getLogger()
    rootLogger.setLevel(level)
    if "IPKernelApp" in logging.Logger.manager.loggerDict:
        return
    else:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(DEFAULT_LOG_FORMAT)
        rootLogger.addHandler(consoleHandler)


def setup_logging(path, level: logging.INFO, redis_client, verbose=True):

    warnings.filterwarnings('ignore', category=FutureWarning)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').disabled = True
    logging.getLogger('filelock').setLevel(logging.ERROR)

    for mod in ['sh']:
        modLogger = logging.getLogger(mod)
        modLogger.setLevel(logging.ERROR)

    log_path = os.path.join(path, "logs")
    os.makedirs(log_path, exist_ok=True)

    log_filename = os.path.join(log_path, "{}_{}.log".format(platform.node(), os.getpid()))

    rootLogger = logging.getLogger()
    rootLogger.setLevel(level)

    if (rootLogger.hasHandlers()):
        rootLogger.handlers.clear()

    fileHandler = logging.FileHandler(log_filename)
    fileHandler.setFormatter(DEFAULT_LOG_FORMAT)
    rootLogger.addHandler(fileHandler)

    if verbose:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(DEFAULT_LOG_FORMAT)
        rootLogger.addHandler(consoleHandler)


def new_entity_logger(path, entity_type, uid, level: logging.INFO, redis_client, log_to_stdout=True, sub_id="default"):

    warnings.filterwarnings('ignore', category=FutureWarning)

    log_path = os.path.join(path, entity_type, uid)
    os.makedirs(log_path, exist_ok=True)

    log_filename = os.path.join(log_path, f"log_{sub_id}.txt")  # .format(platform.node(),os.getpid()))
    logformatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    custom_logger = logging.getLogger(entity_type + "/" + uid)
    custom_logger.setLevel(level)
    if len(custom_logger.handlers) == 0:
        fileHandler = logging.FileHandler(log_filename)
        fileHandler.setFormatter(logformatter)
        custom_logger.addHandler(fileHandler)

        redisHandler = RedisHandler("{}_{}".format(entity_type, uid), redis_client)
        redisHandler.setFormatter(logformatter)
        custom_logger.addHandler(redisHandler)

    return custom_logger


def new_scoped_logger(path, scope_name, level: logging.INFO, redis_client):

    warnings.filterwarnings('ignore', category=FutureWarning)
    log_filename = os.path.join(path, f"{scope_name}.txt")
    logformatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    custom_logger = logging.getLogger(scope_name)
    custom_logger.setLevel(level)
    if len(custom_logger.handlers) == 0:
        fileHandler = logging.FileHandler(log_filename)
        fileHandler.setFormatter(logformatter)
        custom_logger.addHandler(fileHandler)

        redisHandler = RedisHandler(scope_name, redis_client)
        redisHandler.setFormatter(logformatter)
        custom_logger.addHandler(redisHandler)

    return custom_logger
