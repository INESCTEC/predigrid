import jsonschema as jschema

from flask import request
from flask_restful import Resource

from ..common.UserClass import UserClass
from ..common.databases import CassandraDB
from ..common.schemas import RESET_PASSWORD
from ..resources.post_login import token_required_admin


class PatchResetPassword(Resource, UserClass):
    method_decorators = [token_required_admin]

    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config')
        self.client_ip = request.remote_addr
        self.request_route = request.path
        self.client_usr = ''
        self.log_msg = "{ip}|{user}|{route}|'{msg}'"

    def patch(self):
        """
        Reset a user's password
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
              id: ResetPassword
              required:
                - username
                - old_password
                - new_password
              properties:
                username:
                  type: string
                  description: Username of the account to reset
                  example: "alice"
                old_password:
                  type: string
                  format: password
                  description: The current password of the user
                  example: "OldSecret123"
                new_password:
                  type: string
                  format: password
                  description: The new password to assign to the account
                  example: "NewSecret456"
        responses:
          201:
            description: Password reset successful
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 1
                message:
                  type: string
                  example: "Successful password reset. Please login again with the new credentials."
          400:
            description: Invalid request body or authorization mismatch
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "Invalid authorization token."
          401:
            description: Unauthorized (invalid token or wrong username/password)
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
            description: Internal server error while updating password
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
        try:
            # -- Get Username from token:
            self.client_usr = self.get_user_from_token()
        except Exception as ex:
            self.logger.error(msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=repr(ex)))
            return {'code': 0, 'message': 'Invalid authorization token.'}, 401

        # -- Validate JSON Schema (return any problem if detected):
        try:
            jschema.validate(request.get_json(), RESET_PASSWORD)
        except jschema.exceptions.ValidationError as ex:
            self.logger.error(msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=repr(ex)))
            return {"code": 0, "message": repr(ex).replace("\"", "")}, 400

        username = request.json.get('username', None)

        if username != self.client_usr:
            return {"code": 0, "message": "Invalid authorization token."}, 400

        old_password = request.json.get('old_password', None)
        new_password = request.json.get('new_password', '')

        try:
            # Connect to Cassandra DB:
            db_con = CassandraDB.get_cassandra_instance(config=self.config)

            # -- Get User Data:
            user_table = self.config.TABLES["users"]
            q_res = db_con.read_query(f"SELECT * FROM {user_table} "
                                      f"WHERE username='{username}'")

            if q_res.empty:
                return {'code': 0, 'message': 'Unauthorized: Invalid username/password.'}, 401
            else:
                expected_old_password_hash = q_res["password_hash"][0]

            # -- Validate Username/Password combination:
            valid_hash = self.verify_password(old_password, expected_old_password_hash)

            if valid_hash == False:
                # Warn user for invalid password:
                msg = "Unauthorized: Invalid username/password."
                self.logger.error(msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=msg))
                return {'code': 0, 'message': msg}, 401
            else:
                new_password_hash = self.hash_password(new_password)
                q = f"UPDATE {user_table} " \
                    f"SET password_hash='{new_password_hash}' " \
                    f"WHERE username='{username}' IF EXISTS;"
                db_con.execute_query(query=q)
                msg = "Successful password reset. " \
                      "Please login again with the new credentials."
                self.logger.info(
                    msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=msg))
                return {'code': 1, 'message': msg}, 201
        except BaseException as ex:
            self.logger.error(
                msg=self.log_msg.format(self.client_ip, self.client_usr,
                                        self.request_route, repr(ex)))
            return {"code": 0, "message": "Internal Server Error."}, 500
