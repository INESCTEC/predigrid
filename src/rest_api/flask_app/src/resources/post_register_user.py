import datetime as dt

from flask import request
from flask_restful import Resource

from ..common.UserClass import UserClass
from ..common.databases import CassandraDB
from ..resources.post_login import token_required_admin


class PostRegisterUser(Resource, UserClass):
    method_decorators = [token_required_admin]

    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config')
        self.client_ip = request.remote_addr
        self.request_route = request.path
        self.client_usr = 'admin'
        self.log_msg = "{ip}|{user}|{route}|'{msg}'"

    def post(self):
        """
        Register a new user
        ---
        tags:
          - Users
        parameters:
          - name: Authorization
            in: header
            type: string
            required: true
            description: >
              Bearer JWT token with `admin` privileges.
              Example: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
          - name: body
            in: body
            required: true
            schema:
              id: RegisterUser
              required:
                - username
                - password
              properties:
                username:
                  type: string
                  description: Unique username for the new account
                  example: "newuser"
                password:
                  type: string
                  format: password
                  description: Plain-text password to be hashed before storage
                  example: "MySecurePassword123"
                description:
                  type: string
                  description: Optional description or user information
                  example: "Energy operator account"
                permissions:
                  type: string
                  enum: [none, basic, intermediate, advanced, admin]
                  description: Permission level assigned to the user
                  example: "intermediate"
        responses:
          201:
            description: User successfully registered
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 1
                message:
                  type: string
                  example: "ok"
                token:
                  type: string
                  description: JWT token issued for the new user
                  example: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
          400:
            description: Missing required fields or invalid request body
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "Missing password."
          401:
            description: Unauthorized (invalid or missing admin token)
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "Invalid token."
          500:
            description: Internal server error while registering user
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "Failed to register new user."
        """
        try:
            # -- Get Username from token:
            self.client_usr = self.get_user_from_token()
            log_msg_ = self.log_msg.format(ip=self.client_ip,
                                           user=self.client_usr,
                                           route=self.request_route,
                                           msg="{}")
        except Exception as ex:
            self.logger.exception(
                msg=self.log_msg.format(ip=self.client_ip,
                                        user=self.client_usr,
                                        route=self.request_route,
                                        msg=repr(ex)))
            return {'code': 0, 'message': 'Invalid token.'}, 401

        if request.json is None:
            return {"code": 0, "message": "Must provide username/password combination."}, 400

        users_table = self.config.TABLES["users"]
        username = request.json.get('username', None)
        password = request.json.get('password', None)
        userinfo = request.json.get('description', '')
        permissions = request.json.get('permissions', 'none')

        if username is None:
            return {"code": 0, "message": "Missing username."}, 400
        if password is None:
            return {"code": 0, "message": "Missing password."}, 400

        password_hash = self.hash_password(password)
        last_login = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        register_date = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        q = f"INSERT INTO {users_table} (" \
            f"username, password_hash, " \
            f"permissions, userinfo, " \
            f"registered_at, last_login) " \
            f"VALUES (" \
            f"'{username}', '{password_hash}', '{permissions}', " \
            f"'{userinfo}', '{register_date}', '{last_login}')"

        # Connect to Cassandra DB:
        try:
            db_con = CassandraDB.get_cassandra_instance(config=self.config)
            db_con.execute_query(q)
        except BaseException as ex:
            msg = "Failed to register user {}. ERROR: {}".format(username, repr(ex))
            self.logger.error(msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=msg))
            return {"code": 0, "message": "Failed to register new user."}, 500

        # If password is valid, generate auth token and update database with login datetime info:
        token, _ = self.generate_auth_token(
            username=username,
            permissions=permissions,
            password=password_hash,
            expiration=self.config.TOKEN_DURATION
        )

        msg = "User {} registered with success.".format(username)
        self.logger.info(msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=msg))
        return {"code": 1, "message": "ok", "token": token}, 201
