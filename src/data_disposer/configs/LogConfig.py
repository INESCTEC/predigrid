import os

from .logs.Log import create_log


class LogConfig:
    def __init__(self, root_dir):
        self.loggers = {}
        self.root_dir = root_dir

    def get_loggers(self):
        self.loggers["netload"] = create_log(
            service='data-disposer-netload')
        self.loggers["processed"] = create_log(
            service='data-disposer-processed')
        return self.loggers
