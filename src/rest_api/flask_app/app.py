# -- Uncomment code bellow for debug purposes --
# from dotenv import load_dotenv
# load_dotenv(r"..\ENV\.env.dev")

import os

from flask import Flask
from flask_restful import Api
from flasgger import Swagger

from configs import SystemConfig, LogConfig
from src.resources import (
    GetTestService,
    GetUserList,
    GetForecast,
    GetMeasurements,
    GetKPI,
    GetInstallationsInfo
)
from src.resources import (
    PostLogin,
    PostMeasurements,
    PatchStatusInstallation,
    PostRegisterInstallation,
    PostRegisterUser,
    PutUpdatePermission,
    PatchResetPassword,
)

# -- Project ROOT directory:
__ROOT_DIR__ = os.path.dirname(os.path.abspath(__file__))
print("ROOT DIR:", __ROOT_DIR__)

# -- Init Loggers:
loggers = LogConfig(root_dir=__ROOT_DIR__).get_loggers()

loggers["system"].info(">" * 70)
loggers["system"].info(f"Warning! Working on {os.environ['MODE']} mode!")
loggers["system"].info("<" * 70)

# -- Init configs:
config_class = SystemConfig(root_dir=__ROOT_DIR__)

# -- Init Flask APP:
app = Flask("REST-Forecast")
app.config['BUNDLE_ERRORS'] = True
app.config['SECRET_KEY'] = config_class.SECRET_KEY

# -- Init API:
api = Api(app)

# -- Init Swagger:
swagger = Swagger(app)

# -- Add GET methods
api.add_resource(GetForecast, '/api/forecast/get-forecast',
                 resource_class_kwargs={
                     "logger": loggers["get_forecasts"],
                     "config": config_class
                 })
api.add_resource(GetMeasurements, '/api/forecast/measurements',
                 resource_class_kwargs={
                     "logger": loggers["get_measurements"],
                     "config": config_class
                 })
api.add_resource(GetUserList, '/api/user/list',
                 resource_class_kwargs={
                     "logger": loggers["users"],
                     "config": config_class
                 })
api.add_resource(GetKPI, '/api/forecast/kpi',
                 resource_class_kwargs={
                     "logger": loggers["get_kpi"],
                     "config": config_class
                 })
api.add_resource(GetInstallationsInfo, '/api/installations/info',
                 resource_class_kwargs={
                     "logger": loggers["get_installations"],
                     "config": config_class
                 })
api.add_resource(GetTestService, '/api/forecast/test-service')


# -- Add POST methods
api.add_resource(PostMeasurements, '/api/forecast/measurements',
                 resource_class_kwargs={
                     "logger": loggers["post_measurements"],
                     "config": config_class
                 })
api.add_resource(PostRegisterUser, '/api/user/register',
                 resource_class_kwargs={
                     "logger": loggers["users"],
                     "config": config_class
                 })
api.add_resource(PostRegisterInstallation, '/api/installations/register',
                 resource_class_kwargs={
                     "logger": loggers["post_installations"],
                     "config": config_class
                 })
api.add_resource(PostLogin, '/api/user/login',
                 resource_class_kwargs={
                     "logger": loggers["users"],
                     "config": config_class
                 })


# -- Add PUT methods
api.add_resource(PutUpdatePermission, '/api/user/permissions',
                 resource_class_kwargs={
                     "logger": loggers["users"],
                     "config": config_class
                 })


# -- Add PATCH methods
api.add_resource(PatchStatusInstallation, '/api/installations/status',
                 resource_class_kwargs={
                     "logger": loggers["post_installations"],
                     "config": config_class
                 })
api.add_resource(PatchResetPassword, '/api/user/password-reset',
                 resource_class_kwargs={
                     "logger": loggers["users"],
                     "config": config_class
                 })


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5001)
