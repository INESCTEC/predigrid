import os
from .logs.Log import create_log


class LogConfig:
    def __init__(self, root_dir):
        self.loggers = {}
        self.root_dir = root_dir
        self.logs_dir = os.path.join(root_dir, "files", "logs")
        # os.makedirs(self.logs_dir, exist_ok=True)

    def get_loggers(self):
        self.loggers["system"] = create_log(
            service='rest-api-system')
        self.loggers["post_measurements"] = create_log(
            service='rest-api-measurements')
        self.loggers["post_installations"] = create_log(
            service='rest-api-installations')
        self.loggers["users"] = create_log(
            service='rest-api-users')
        # GET loggers:
        self.loggers["get_forecasts"] = create_log(
            service='rest-api-get-forecast')
        self.loggers["get_measurements"] = create_log(
            service='rest-api-get-measurements')
        self.loggers["get_kpi"] = create_log(
            service='rest-api-kpi')
        self.loggers["get_installations"] = create_log(
            service='rest-api-get-installations-info')
        return self.loggers
