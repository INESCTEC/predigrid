import datetime as dt
import jsonschema as jschema

from flask import request
from flask_restful import Resource

from ..common.UserClass import UserClass
from ..common.databases import CassandraDB
from ..common.schemas import CHANGE_INST_STATUS
from ..resources.post_login import token_required_admin


class PatchStatusInstallation(Resource, UserClass):
    method_decorators = [token_required_admin]

    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config')
        self.client_ip = request.remote_addr
        self.request_route = request.path
        self.client_usr = 'admin'
        self.log_msg = "{ip}|{user}|{route}|'{msg}'"

    def patch(self):
        """
        Update the active status of an installation
        ---
        tags:
          - Installations
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
              id: PatchInstallationStatus
              required:
                - installation_code
                - is_active
              properties:
                installation_code:
                  type: string
                  description: Unique identifier of the installation to update
                  example: "inst_001"
                is_active:
                  type: integer
                  enum: [0, 1]
                  description: >
                    New active status:
                    - 1 → active
                    - 0 → inactive
                  example: 1
        responses:
          201:
            description: Installation status updated successfully
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
            description: Invalid request body (schema validation failed)
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "'installation_code' is a required property"
          401:
            description: Unauthorized (missing or invalid token)
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
            description: Internal server error (failed DB update)
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "Failed to register installation."
        """
        # -- Validate JSON Schema (return any problem if detected):
        try:
            jschema.validate(request.get_json(), CHANGE_INST_STATUS)
        except jschema.exceptions.ValidationError as ex:
            self.logger.exception(
                msg=self.log_msg.format(self.client_ip, self.client_usr,
                                        self.request_route, repr(ex)))
            return {"code": 0, "message": repr(ex).replace("\"", "")}, 400

        inst_table = self.config.TABLES["metadata"]
        inst_id = request.json.get('installation_code', None)
        is_active = request.json.get('is_active', None)

        last_updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        q = f"UPDATE {inst_table} " \
            f"SET is_active={is_active}, last_updated='{last_updated}'" \
            f"WHERE id='{inst_id}' IF EXISTS;"

        # Connect to Cassandra DB:
        try:
            db_con = CassandraDB.get_cassandra_instance(config=self.config)
            db_con.execute_query(q)
        except BaseException as ex:
            msg = f"Failed to register installation {inst_id}. " \
                  f"ERROR: {repr(ex)}."
            self.logger.error(
                msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=msg)) # noqa
            return {"code": 0,
                    "message": "Failed to register installation."}, 500

        msg = f"Installation {inst_id} registered with success."
        self.logger.info(
            msg=self.log_msg.format(ip=self.client_ip, user=self.client_usr, route=self.request_route, msg=msg)) # noqa
        return {"code": 1, "message": "ok"}, 201