import sys
import logging
from logging.handlers import RotatingFileHandler

console_msg_fmt = "%(asctime)s [%(levelname)-8.8s] %(message)s" # Message format # noqa

consoleLog = logging.getLogger('just_console')  # Logger only for console
consoleLog.setLevel(logging.DEBUG)

logFormatter = logging.Formatter(console_msg_fmt) # Format output

# stdout handler
outputHandler = logging.StreamHandler(sys.stdout)
outputHandler.setFormatter(logFormatter)

consoleLog.addHandler(outputHandler)


def just_console_log():
    return consoleLog


class MyLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' in kwargs:
            kwargs["extra"].update(self.extra)
        else:
            kwargs["extra"] = self.extra
        return msg, kwargs


def create_log(service, path=None):
    new_logger = logging.getLogger(f'nwp_collector')
    # Set it to whatever level you want - default will be info
    new_logger.setLevel(logging.DEBUG)
    logFormatter = logging.Formatter(console_msg_fmt)

    # Console output handler
    outputHandler = logging.StreamHandler(sys.stdout)
    outputHandler.setFormatter(logFormatter)
    new_logger.addHandler(outputHandler)

    if path is not None:
        # DEBUG file handler
        debugFileHandler = RotatingFileHandler(filename=path+'.log',
                                              maxBytes=100000000)
        debugFileHandler.setFormatter(logFormatter)
        debugFileHandler.setLevel(logging.DEBUG)
        new_logger.addHandler(debugFileHandler)

    extra_dict = {'service': service}

    new_logger = MyLoggerAdapter(logger=new_logger, extra=extra_dict)
    return new_logger
