import numpy as np
import pandas as pd

from flask import request
from flask_restful import Resource, reqparse, inputs

from ..common.UserClass import UserClass
from ..common.databases import CassandraDB
from ..resources.post_login import token_required_intermediate




class GetInstallationsInfo(Resource, UserClass):
    """
    Class used to retrieve installations metadata from the database:
        * Validates user privileges to access this method;
        * Queries Cassandra Database and returns JSON with installations metadata.
    """
    method_decorators = [token_required_intermediate]

    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config')
        self.client_ip = request.remote_addr
        self.request_route = request.url
        self.client_usr = ''
        self.log_msg = "{ip}|{user}|{route}|'{msg}'"

    def __repr__(self):
        return "GetInstallationsInfo"

    def url_args(self):

        # -- more info: https://flask-restful.readthedocs.io/en/0.3.5/reqparse.html # noqa
        parser = reqparse.RequestParser()
        parser.add_argument('installation_code',
                            type=str,
                            location='args',
                            help="{'installation_code' param missing.}")
        args = parser.parse_args()

        return args

    def get(self):
        """
        Retrieve installations metadata from the database
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
          - in: query
            name: installation_code
            required: false
            schema:
              type: string
            description: >
              Comma-separated list of installation codes.  
              If omitted, metadata for all installations is returned.
        responses:
          200:
            description: Installations metadata successfully retrieved
            content:
              application/json:
                example:
                  code: 1
                  data:
                    - installations:
                        - installation_code: "12345"
                          installation_type: "LV"
                          location: "Downtown"
                          owner: "ACME Corp"
                          last_updated: "2025-09-19T08:00:00Z"
                        - installation_code: "67890"
                          installation_type: "MV"
                          location: "Uptown"
                          owner: "EnergyCo"
                          last_updated: "2025-09-18T14:30:00Z"
          400:
            description: Invalid or missing parameters
            content:
              application/json:
                example:
                  code: 0
                  message: "Invalid request arguments."
          401:
            description: Unauthorized (invalid token)
            content:
              application/json:
                example:
                  code: 0
                  message: "Invalid token."
          500:
            description: Internal server error
            content:
              application/json:
                example:
                  code: 0
                  message: "Internal Server Error."
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

        try:
            # Parse Request Arguments:
            args = self.url_args()
        except BaseException as ex:
            return {"code": 0, "message": str(ex)}, 400

        try:
            # Response structure:
            json_response = dict(data=[], code=1)

            # Connect to Cassandra DB:
            db_con = CassandraDB.get_cassandra_instance(config=self.config)

            # Get installations codes to retrieve (assumes all if None)
            if args["installation_code"] is None:
                inst_code_list = []
            else:
                inst_code_ = args["installation_code"].split(',')
                inst_code_list = tuple([x for x in inst_code_])

            data_dict = self.query_installations(
                db_con=db_con,
                inst_list=inst_code_list,
            )
            json_response["data"].append(data_dict)
            # --- JSON Response:
            self.logger.info(msg=log_msg_.format("ok"))
            return json_response, 200

        except BaseException as ex:
            self.logger.exception(msg=log_msg_.format(repr(ex)))
            return {"code": 0, "message": "Internal Server Error."}, 500

    def query_installations(self,
                            db_con,
                            inst_list):

        # --- Tables:
        inst_table = self.config.TABLES["metadata"]

        # --- Fetch from DB information of all installations available:
        query = f"select * from {inst_table};"
        inst_metadata = db_con.read_query(query)

        # --- Filter clients to retrieve from database, if installation_code
        # arg is specified:
        if len(inst_list) > 0:
            sch_ = np.isin(inst_metadata["id"], inst_list)
            to_search = inst_metadata.loc[sch_, :].copy()
        else:
            to_search = inst_metadata.copy()

        data_dict = dict(
            installations=[]
        )

        to_search["last_updated"] = to_search["last_updated"].dt.strftime("%Y-%m-%dT%H:%M:%SZ") # noqa
        to_search = to_search.rename(columns={"id":"installation_code", "id_type":"installation_type"}) # noqa
        data_dict["installations"].append(eval(to_search.to_json(orient='records'))) # noqa

        return data_dict
