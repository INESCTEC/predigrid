
import jwt
import datetime as dt

from flask import current_app, request
from passlib.apps import custom_app_context as pwd_context


class UserClass:

    @staticmethod
    def hash_password(password):
        return pwd_context.encrypt(password)

    @staticmethod
    def verify_password(password, hash_password):
        return pwd_context.verify(password, hash_password)

    @staticmethod
    def generate_auth_token(username, permissions, password, expiration=600):
        token_duration = expiration
        token = jwt.encode(
            {
                'user': username,
                'permissions': permissions,
                'password': password,
                'exp': dt.datetime.utcnow() + dt.timedelta(seconds=expiration)
            },
            current_app.config["SECRET_KEY"]
        )
        return token, token_duration

    @staticmethod
    def verify_permission(permission_levels, token_location='header', field='Authorization'):

        if token_location == "header":
            token = request.headers.get(field)
        else:
            token = request.args.get(field)
        decoded_token = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=['HS256'])
        permission_lvl = decoded_token.get("permissions", '')

        return permission_lvl in permission_levels

    @staticmethod
    def get_user_from_token(field='Authorization'):
        token = request.headers.get(field)
        decoded_token = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=['HS256'])
        user_id = decoded_token.get("user", '')
        return user_id


