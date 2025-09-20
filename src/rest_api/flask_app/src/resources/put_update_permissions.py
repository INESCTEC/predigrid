
from flask import request
from flask_restful import Resource

from ..common.UserClass import UserClass
from ..common.databases import CassandraDB
from ..resources.post_login import token_required_admin


class PutUpdatePermission(Resource, UserClass):
    method_decorators = [token_required_admin]

    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config')
        self.client_ip = request.remote_addr
        self.request_route = request.path
        self.client_usr = 'admin'
        self.log_msg = "{ip}|{user}|{route}|'{msg}'"

    def put(self):
        """
        Update a user's permissions
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
              id: UpdateUserPermission
              required:
                - username
                - permissions
              properties:
                username:
                  type: string
                  description: The username whose permissions will be updated
                  example: "jane"
                permissions:
                  type: string
                  enum: [none, basic, intermediate, advanced, admin]
                  description: New permission level for the user
                  example: "advanced"
                cod_restriction:
                  type: string
                  description: Optional code restriction to apply
                  example: "GRID-001"
        responses:
          201:
            description: User permissions successfully updated
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 1
                message:
                  type: string
                  example: "ok"
          400:
            description: Invalid request body or invalid permission level
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "That's not a valid permission level. Valid levels: ['none','basic','intermediate','advanced','admin']"
          401:
            description: Unauthorized (invalid token or username not found)
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "Invalid username."
          500:
            description: Internal server error while updating permissions
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "Failed to update permissions for user 'jane'."
        """
        user_table = self.config.TABLES["users"]
        valid_permission_levels = ["none", "basic", "intermediate",
                                   "advanced", "admin"]
        if request.json is None:
            return {"code": 0, "message": "Must provide username/password combination."}, 400

        username = request.json.get('username', '')
        permission = request.json.get('permissions', 'basic').lower()
        cod_restriction = request.json.get('cod_restriction', None)

        if username is None:
            return {"code": 0, "message": "Missing username."}, 400

        if permission not in valid_permission_levels:
            return {"code": 0, "message": "That's not a valid permission level. "
                                          "Valid levels: {}".format(valid_permission_levels)}, 400

        # -- Get User Data:
        db_con = CassandraDB.get_cassandra_instance(config=self.config)
        q_res = db_con.read_query("SELECT * FROM {} where username='{}'".format(user_table, username))

        if q_res.empty:
            return {'code': 0, 'message': 'Invalid username.'}, 401

        try:
            q = "UPDATE {} SET permissions='{}' WHERE username='{}' IF EXISTS;"
            q = q.format(user_table, permission, username)
            db_con.execute_query(query=q)
        except BaseException as ex:
            msg = "Failed to update user {} permissions. ERROR: {}".format(username, repr(ex))
            self.logger.error(msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=msg))
            return {"code": 0, "message": "Failed to update permissions for user '{}'.".format(username)}, 500

        if cod_restriction is not None:
            try:
                q = "UPDATE {} SET cod_restriction='{}' WHERE username='{}' IF EXISTS;"
                q = q.format(user_table, cod_restriction, username)
                db_con.execute_query(query=q)
            except BaseException as ex:
                msg = "Failed to update user {} cod restriction. ERROR: {}".format(username, repr(ex))
                self.logger.error(msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=msg))
                return {"code": 0, "message": "Failed to update cod restriction for user '{}'.".format(username)}, 500

        msg = "User {} permissions successfully changed.".format(username)
        self.logger.info(msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=msg))
        return {"code": 1, "message": "ok"}, 201
