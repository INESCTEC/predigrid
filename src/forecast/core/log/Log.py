import sys
import socket
import logging
from typing import Union
from logging.handlers import RotatingFileHandler

console_msg_fmt = "%(asctime)s [%(levelname)-8.8s] %(message)s" # Message format # noqa

consoleLog = logging.getLogger('just_console')  # Logger only for console
consoleLog.setLevel(logging.DEBUG)

logFormatter = logging.Formatter(console_msg_fmt)  # Format output

# Stdout handler
outputHandler = logging.StreamHandler(sys.stdout)
outputHandler.setFormatter(logFormatter)

consoleLog.addHandler(outputHandler)


def is_reachable(host: str, port: Union[str, int]):
    """
    Tests if a specific port and host can be reached.

    :param host: The host to test
    :type host: str
    :param port: Port number to test
    :type port: str, int
    :return:
    :rtype:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, int(port)))
        s.shutdown(2)
        return True
    except BaseException:
        return False


def just_console_log():
    return consoleLog


class MyLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' in kwargs:
            kwargs["extra"].update(self.extra)
        else:
            kwargs["extra"] = self.extra
        return msg, kwargs


def create_log(service, inst_id=None, path=None):
    logger_suffix = f"_{inst_id}" if inst_id is not None else ''
    new_logger = logging.getLogger(f'{service}{logger_suffix}')

    if not new_logger.hasHandlers():
        # Set it to whatever level you want - default will be info
        new_logger.setLevel(logging.DEBUG)
        logFormatter = logging.Formatter(console_msg_fmt)

        # Console output handler
        outputHandler = logging.StreamHandler(sys.stdout)
        outputHandler.setFormatter(logFormatter)
        new_logger.addHandler(outputHandler)

        if path is not None:
            # INFO file handler
            infoFileHandler = RotatingFileHandler(filename=path + '.log',
                                                  maxBytes=100000000,
                                                  backupCount=5)
            infoFileHandler.setFormatter(logFormatter)
            infoFileHandler.setLevel(logging.INFO)
            new_logger.addHandler(infoFileHandler)

    extra_dict = {'service': service}

    new_logger = MyLoggerAdapter(logger=new_logger, extra=extra_dict)
    return new_logger
