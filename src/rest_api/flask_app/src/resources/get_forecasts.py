import numpy as np
import pandas as pd

from flask import request
from flask_restful import Resource, reqparse, inputs

from ..common.UserClass import UserClass
from ..common.databases import CassandraDB
from ..resources.post_login import token_required_basic




class GetForecast(Resource, UserClass):

    method_decorators = [token_required_basic]

    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config')
        self.client_ip = request.remote_addr
        self.request_route = request.url
        self.client_usr = ''
        self.log_msg = "{ip}|{user}|{route}|'{msg}'"

    def __repr__(self):
        return "GetForecast"

    def url_args(self):
        # -- more info: https://flask-restful.readthedocs.io/en/0.3.5/reqparse.html
        parser = reqparse.RequestParser()
        parser.add_argument('start_date',
                            type=inputs.datetime_from_iso8601,
                            required=True,
                            location='args',
                            help="{'start_date' param missing.}")
        parser.add_argument('end_date',
                            type=inputs.datetime_from_iso8601,
                            required=True,
                            location='args',
                            help="{'end_date' param missing.}")
        parser.add_argument('quantiles',
                            type=str,
                            location='args',
                            default="50",
                            help="{'quantiles' param missing.}")
        parser.add_argument('installation_code',
                            type=str,
                            location='args',
                            help="{'installation_code' param missing.}")
        parser.add_argument('register_type',
                            type=str,
                            location='args',
                            default="",
                            help="{'register_type' param missing..}")
        args = parser.parse_args()

        return args

    def get(self):
        """
                Retrieve forecast data from the database
                ---
                tags:
                  - Forecast
                parameters:
                  - name: Authorization
                    in: header
                    type: string
                    required: true
                    description: >
                      Bearer JWT token with `admin` privileges.
                      Example: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
                  - in: query
                    name: start_date
                    required: true
                    schema:
                      type: string
                      format: date-time
                    description: Start date (ISO8601 format) of the forecast window.
                  - in: query
                    name: end_date
                    required: true
                    schema:
                      type: string
                      format: date-time
                    description: End date (ISO8601 format) of the forecast window.
                  - in: query
                    name: quantiles
                    required: false
                    schema:
                      type: string
                      example: "10,50,90"
                    description: >
                      Comma-separated list of quantiles to retrieve.
                      Defaults to "50".
                  - in: query
                    name: installation_code
                    required: false
                    schema:
                      type: string
                    description: >
                      Comma-separated list of installation codes.
                      If omitted, all installations are returned.
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
                      - Empty string for both.
                      Defaults to empty.
                responses:
                  200:
                    description: Forecast data successfully retrieved
                    content:
                      application/json:
                        example:
                          code: 1
                          data:
                            - forecasts:
                                - installation_type: "LV"
                                  installation_code: "12345"
                                  values:
                                    - timestamp: "2025-09-19T10:00:00Z"
                                      computed_at: "2025-09-19T09:55:00Z"
                                      quantile: "10"
                                      value: 120.5
                                      units: "kW"
                                      variable_name: "P"
                  400:
                    description: Invalid or missing parameters
                    content:
                      application/json:
                        example:
                          code: 0
                          message: "Field 'register_type' must be either 'P' or 'Q'"
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
        # Get Username from token:
        try:
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
                inst_code_list = tuple([x for x in args["installation_code"].split(',')])

            # -- Verify register type arg:
            if args["register_type"] not in ['P', 'Q', ""]:
                msg = "Field 'register_type' must be either 'P' for active " \
                      "power or 'Q' for reactive power (if param is not " \
                      "provided, defaults to P and Q)."
                self.logger.error(msg=log_msg_.format(msg))
                return {"code": 0, "message": msg}, 400

            # -- Select to which grid will data be inserted:
            data_dict = self.query_data(
                db_con=db_con,
                inst_list=inst_code_list,
                start_date=args["start_date"],
                end_date=args["end_date"],
                quantiles=args["quantiles"],
                register_type_filter=args["register_type"]
            )
            json_response["data"].append(data_dict)
            self.logger.info(msg=log_msg_.format("ok"))
            return json_response, 200

        except BaseException as ex:
            self.logger.exception(msg=log_msg_.format(repr(ex)))
            return {"code": 0, "message": "Internal Server Error."}, 500

    def query_data(self,
                   db_con,
                   inst_list,
                   start_date,
                   end_date,
                   quantiles,
                   register_type_filter):

        # --- Tables:
        inst_table = self.config.TABLES["metadata"]
        forecast_table = self.config.TABLES[f"forecast"]

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

        # --- Check which quantiles to retrieve from database:
        quantiles_nomenclature = ["{}".format(x.zfill(2))
                                  for x in quantiles.split(',')]
        quantiles = ["q{}".format(x.zfill(2))
                     for x in quantiles.split(',')]

        # structure to save d ata
        data_dict = dict(
            forecasts=[]
        )

        # -- Filters type of data to retrieve from database:
        if register_type_filter == "":
            register_type_list = ['P', 'Q']
        else:
            register_type_list = [register_type_filter]

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
                        f"datetime as timestamp, " \
                        f"request as computed_at, " \
                        f"{','.join(quantiles)} " \
                        f"FROM {forecast_table} " \
                        f"WHERE id='{r.id}' " \
                        f"and register_type='{register_type}' " \
                        f"and datetime >= '{start_date}' " \
                        f"and datetime <= '{end_date}' " \
                        f"order by datetime asc, request asc;"

                data = db_con.read_query(query)
                data.drop_duplicates("timestamp", keep="last", inplace=True)
                data.dropna(inplace=True)  # avoid JSON with Null data
                data["units"] = data_containers[register_type]['units']
                data["variable_name"] = register_type

                for i, q in enumerate(quantiles):
                    data.rename(columns={q: quantiles_nomenclature[i]}, inplace=True)

                data = data.set_index(["timestamp", "computed_at", "units", "variable_name"]).stack().reset_index()
                data.rename(columns={"level_4": "quantile", 0: "value"}, inplace=True)
                data_containers[register_type]['data'] = data

            values = []

            for reg_type in register_type_list:
                dframe = data_containers[reg_type]['data'].copy()
                if not dframe.empty:
                    dframe["computed_at"] = dframe["computed_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    dframe["timestamp"] = dframe["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    subset_json = dframe.to_json(orient="records")
                    values += eval(subset_json)

            inst_dict["values"] = values
            data_dict["forecasts"].append(inst_dict)

        return data_dict
