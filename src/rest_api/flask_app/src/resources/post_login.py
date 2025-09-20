import jwt
import datetime as dt
import jsonschema as jschema

from functools import wraps
from flask_restful import Resource
from flask import request, current_app

from ..common.databases import CassandraDB
from ..common.UserClass import UserClass
from ..common.schemas import LOGIN


def token_required_levels(permission_levels=None):
    """
    Returns a decorator that verifies desired permission levels
    """
    def token_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')

            if not token:
                return {"code": 0, 'message': 'Token is missing!'}, 401

            try:
                jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])
                if permission_levels is not None and (not UserClass.verify_permission(permission_levels=permission_levels)):
                    return {"code": 0,
                            "message": "Unauthorized: No privileges "
                                       "to access this method."}, 403
            except jwt.InvalidSignatureError:
                return {"code": 0, 'message': 'Token is invalid!'}, 401
            except jwt.ExpiredSignatureError:
                return {"code": 0, 'message': 'Token expired! New login required.'}, 401
            except Exception as ex:
                print("ERROR: Exception while validating token:", repr(ex))
                return {"code": 0, 'message': 'Token is invalid!'}, 401

            return f(*args, **kwargs)

        return decorated
    return token_required


token_required_basic = token_required_levels(['basic', 'intermediate', 'advanced', 'admin'])
token_required_intermediate = token_required_levels(['intermediate', 'advanced', 'admin'])
token_required_advanced = token_required_levels(['advanced', 'admin'])
token_required_admin = token_required_levels(['admin'])


class PostLogin(Resource, UserClass):
    """
    Class used to Login into the REST API:
        * Validates if the user exists in database;
        * Validates user password by comparing with database hash password representation;
        * Provides authentication token with predefined timeout.

    """
    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config')
        self.client_ip = request.remote_addr
        self.request_route = request.url
        self.client_usr = ''
        self.log_msg = "{ip}|{user}|{route}|'{msg}'"

    def __repr__(self):
        return "PostLogin"

    def post(self):
        """
                User login and token generation
                ---
                tags:
                  - Authentication
                parameters:
                  - name: body
                    in: body
                    required: true
                    schema:
                      id: Login
                      required:
                        - username
                        - password
                      properties:
                        username:
                          type: string
                          description: The username of the account
                          example: "alice"
                        password:
                          type: string
                          format: password
                          description: The user's password
                          example: "SecretPass123"
                responses:
                  200:
                    description: Successful login, JWT token generated
                    schema:
                      type: object
                      properties:
                        code:
                          type: integer
                          example: 1
                        token:
                          type: string
                          description: Bearer token to be used in subsequent requests
                          example: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                        duration:
                          type: integer
                          description: Token duration in seconds
                          example: 3600
                  400:
                    description: Invalid request body (schema validation failed)
                    schema:
                      type: object
                      properties:
                        code:
                          type: integer
                          example: 0
                        message:
                          type: string
                          example: "'password' is a required property"
                  401:
                    description: Unauthorized (invalid username or password)
                    schema:
                      type: object
                      properties:
                        code:
                          type: integer
                          example: 0
                        message:
                          type: string
                          example: "Unauthorized: Invalid username/password."
                  500:
                    description: Internal server error
                    schema:
                      type: object
                      properties:
                        code:
                          type: integer
                          example: 0
                        message:
                          type: string
                          example: "Internal Server Error."
                """
        user_table = self.config.TABLES["users"]

        # -- Validate JSON Schema (return any problem if detected):
        try:
            jschema.validate(request.get_json(), LOGIN)
        except jschema.exceptions.ValidationError as ex:
            self.logger.exception(
                msg=self.log_msg.format(ip=self.client_ip,
                                        user=self.client_usr,
                                        route=self.request_route,
                                        msg=repr(ex)))
            return {"code": 0, "message": repr(ex).replace("\"", "")}, 400

        username = request.json.get('username', None)
        password = request.json.get('password', None)
        self.client_usr = username
        log_msg_ = self.log_msg.format(ip=self.client_ip,
                                       user=self.client_usr,
                                       route=self.request_route,
                                       msg="{}")
        try:
            # Connect to Cassandra DB:
            db_con = CassandraDB.get_cassandra_instance(config=self.config)

            # -- Get User Data:
            q_res = db_con.read_query(
                f"SELECT * "
                f"FROM {user_table} "
                f"WHERE username='{username}';"
            )

            if q_res.empty:
                msg = 'Unauthorized: Invalid username/password.'
                self.logger.error(msg=log_msg_.format(msg))
                return {'code': 0, 'message': msg}, 401
            else:
                password_hash = q_res["password_hash"][0]
                permissions = q_res["permissions"][0]

            # -- Validate Username/Password combination:
            valid_hash = self.verify_password(password, password_hash)

            if valid_hash == False:
                # Warn user for invalid password:
                msg = "Unauthorized: Invalid username/password."
                self.logger.error(msg=log_msg_.format(msg))
                return {'code': 0, 'message': msg}, 401
            else:
                # If password is valid, generate auth token and update
                # database with login datetime info:
                token, token_duration = self.generate_auth_token(
                    username=username,
                    permissions=permissions,
                    password=password_hash,
                    expiration=self.config.TOKEN_DURATION
                )
                try:
                    last_login = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    db_con.execute_query(
                        f"UPDATE {user_table} " \
                        f"SET last_login='{last_login}' " \
                        f"WHERE username='{username}';"
                    )
                except BaseException:
                    msg = f"Error updating last_login for username {username}."
                    self.logger.exception(msg=log_msg_.format(msg))

                msg = "Successfully logged in."
                self.logger.info(msg=log_msg_.format(msg))
                return {'code': 1,
                        'token': token,
                        'duration': token_duration}, 200

        except BaseException as ex:
            self.logger.exception(msg=log_msg_.format(repr(ex)))
            return {"code": 0, "message": "Internal Server Error."}, 500
