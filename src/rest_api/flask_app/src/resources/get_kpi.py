import numpy as np
import pandas as pd

from flask import request
from flask_restful import Resource, reqparse, inputs

from ..common.UserClass import UserClass
from ..common.databases import CassandraDB
from ..resources.post_login import token_required_intermediate


class GetKPI(Resource, UserClass):
    """
    Class used to retrieve KPIs from the database:
        * Validates user privileges to access this method;
        * Queries Cassandra Database and returns JSON with KPIs.
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
        return "GetKPI"

    def url_args(self):
        # -- info: https://flask-restful.readthedocs.io/en/0.3.5/reqparse.html
        parser = reqparse.RequestParser()
        parser.add_argument('request_day',
                            type=inputs.datetime_from_iso8601,
                            required=True,
                            location='args',
                            help="{'request_day' param missing.}")
        parser.add_argument('installation_code',
                            type=str,
                            location='args',
                            help="{'installation_code' param missing.}")
        parser.add_argument('register_type',
                            type=str,
                            location='args',
                            default="",
                            help="{'register_type' param missing..}")
        parser.add_argument('term',
                            type=str,
                            location='args',
                            default='long',
                            help="{'term' param missing.}")
        parser.add_argument('metrics',
                            type=str,
                            location='args',
                            default=None,
                            help="{'metric' param missing.}")
        args = parser.parse_args()

        return args

    def get(self):
        """
        Retrieve Key Performance Indicators (KPIs) for installations
        ---
        tags:
          - KPIs
        parameters:
          - name: Authorization
            in: header
            type: string
            required: true
            description: >
                Bearer JWT token with `admin` privileges.
                Example: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
          - in: query
            name: request_day
            required: true
            schema:
              type: string
              format: date-time
            description: >
              Date (ISO8601 format) for which KPIs should be retrieved.
          - in: query
            name: installation_code
            required: false
            schema:
              type: string
            description: >
              Comma-separated list of installation codes.
              If omitted, KPIs for all installations are returned.
          - in: query
            name: register_type
            required: false
            schema:
              type: string
              enum: [P, Q, ""]
            description: >
              Register type:
              - "P" for active power (kW)
              - "Q" for reactive power (kvar)
              - Empty string for both (default).
          - in: query
            name: term
            required: false
            schema:
              type: string
              enum: [short, long]
              default: long
            description: >
              Time horizon for KPIs:
              - "short" → based on last 2 weeks of forecasts
              - "long" → based on last year of forecasts (default).
          - in: query
            name: metrics
            required: false
            schema:
              type: string
              example: "mae,rmse"
            description: >
              Comma-separated list of metrics to return.
              Supported: `crps`, `mae`, `nmae`, `rmse`, `nrmse`.
              Defaults to all.
        responses:
          200:
            description: KPIs successfully retrieved
            content:
              application/json:
                example:
                  code: 1
                  data:
                    - kpis:
                        - installation_type: "LV"
                          installation_code: "12345"
                          values:
                            - horizon: 1
                              mae: 12.5
                              rmse: 15.2
                              nmae: 0.07
                              nrmse: 0.09
                              variable_name: "P"
                            - horizon: 1
                              mae: 8.1
                              rmse: 10.3
                              nmae: 0.05
                              nrmse: 0.06
                              variable_name: "Q"
          400:
            description: Invalid or missing parameters
            content:
              application/json:
                example:
                  code: 0
                  message: "Field 'term' must be either 'short' or 'long'"
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

            # -- Verify register type arg:
            if args["register_type"] not in ['P', 'Q', ""]:
                msg = "Field 'register_type' must be either 'P' for active " \
                      "power or 'Q' for reactive power (if param is not " \
                      "provided, defaults to P and Q)."
                self.logger.error(msg=log_msg_.format(msg))
                return {"code": 0, "message": msg}, 400

            # -- Verify term arg:
            if args["term"] not in ["short", "long"]:
                msg = "Field 'term' must be either 'short' for metrics " \
                      "based on last 2 weeks of forecasts or 'long' for " \
                      "metrics based on last year of forecasts " \
                      "(default is 'long')."
                self.logger.error(msg=log_msg_.format(msg))
                return {"code": 0, "message": msg}, 400

            # Get list of metrics to retrieve (assumes all if None)
            if args["metrics"] is None:
                metrics_list = []
            else:
                metrics_ = args["metrics"].split(',')
                metrics_list = tuple([x for x in metrics_])

            # -- Select to which grid will data be inserted:
            data_dict = self.query_kpis(
                db_con=db_con,
                inst_list=inst_code_list,
                metrics_list=metrics_list,
                term=args["term"],
                request_time=args["request_day"],
                register_type=args["register_type"]
            )
            json_response["data"].append(data_dict)
            # --- JSON Response:
            self.logger.info(msg=log_msg_.format("ok"))
            return json_response, 200

        except BaseException as ex:
            self.logger.exception(msg=log_msg_.format(repr(ex)))
            return {"code": 0, "message": "Internal Server Error."}, 500

    def query_kpis(self,
                   db_con,
                   inst_list,
                   metrics_list,
                   term,
                   request_time,
                   register_type):

        # --- Installations table:
        inst_table = self.config.TABLES["metadata"]

        # --- Fetch from DB information of all installations available:
        query = f"select * from {inst_table};"
        inst_metadata = db_con.read_query(query)

        # --- Filter clients to retrieve from database, if installation_code
        # arg is specified:
        meta_cols_ = ["id", "id_type"]
        if len(inst_list) > 0:
            sch_ = np.isin(inst_metadata["id"], inst_list)
            to_search = inst_metadata.loc[sch_, meta_cols_]
        else:
            to_search = inst_metadata[meta_cols_]

        data_dict = dict(
            kpis=[]
        )

        # -- Filters type of data to retrieve from database:
        if register_type == "":
            register_type_list = ['P', 'Q']
        else:
            register_type_list = [register_type]

        # -- Filters type of data to retrieve from database:
        if term == "":
            term = "long"
        metrics_table = self.config.TABLES[f"metrics_{term}term"]

        # Check list of chosen metrics
        all_metrics = ["crps", "mae", "nmae", "rmse", "nrmse"]
        if len(metrics_list) == 0:
            metrics_list = all_metrics
        else:
            invalid_metrics = [x for x in metrics_list if x not in all_metrics]
            metrics_list = [x for x in metrics_list if x in all_metrics]

        # --- Get information from db
        for r in to_search.itertuples():
            inst_dict = dict(
                installation_type=r.id_type,
                installation_code=r.id,
                values=[],
            )

            data_containers = {
                'P': {'units': 'kW', 'data': pd.DataFrame()},
                'Q': {'units': 'kvar', 'data': pd.DataFrame()}
            }

            for register_type in register_type_list:
                query = f"SELECT " \
                        f"{','.join(metrics_list)}, horizon " \
                        f"FROM {metrics_table} " \
                        f"WHERE id='{r.id}' " \
                        f"and register_type='{register_type}' " \
                        f"and request = '{request_time}';"

                data = db_con.read_query(query)
                data["variable_name"] = register_type
                data_containers[register_type]['data'] = data

            # -- Convert DataFrame to Python Dictionary to prepare JSON payload:
            values = []

            for reg_type in register_type_list:
                dframe = data_containers[reg_type]['data'].copy()
                if not dframe.empty:
                    subset_json = dframe.to_json(orient="records")
                    values += eval(subset_json)

            inst_dict["values"] = values
            data_dict["kpis"].append(inst_dict)

        return data_dict
