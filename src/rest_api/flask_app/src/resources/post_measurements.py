
import numpy as np
import pandas as pd
import datetime as dt
import jsonschema as jschema

from time import time
from flask import request
from flask_restful import Resource

from ..common.UserClass import UserClass
from ..common.databases import CassandraDB
from ..common.schemas import SEND_MEASUREMENTS
from ..resources.post_login import token_required_advanced


def find_closest_nwp_point(latitude, longitude, nwp_grid, logger):
    coordinates = (float(latitude), float(longitude))

    if nwp_grid.empty:
        logger.warning(msg="WARNING! Empty NWP grid!! Assuming client coordinates as default nearest location.")
        print("Empty NWP grid")
        return "%.8f" % coordinates[0], "%.8f" % coordinates[1]
    else:
        ll_finder = np.sqrt((nwp_grid['latitude'].values - coordinates[0]) ** 2 + (nwp_grid['longitude'].values - coordinates[1]) ** 2)
        point_coords = nwp_grid.iloc[ll_finder.argmin(), ]
        closest_lat, closest_lon = "%.8f" % point_coords["latitude"], "%.8f" % point_coords["longitude"]
        return closest_lat, closest_lon


class PostMeasurements(Resource, UserClass):

    method_decorators = [token_required_advanced]

    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config')
        self.client_ip = request.remote_addr
        self.request_route = request.url
        self.client_usr = ''
        self.log_msg = "{ip}|{user}|{route}|'{msg}'"

    def post(self):
        """
                Submit new measurements for one or more installations
                ---
                tags:
                  - Measurements
                parameters:
                  - name: Authorization
                    in: header
                    type: string
                    required: true
                    description: >
                      Bearer JWT token with at least `advanced` privileges.
                      Example: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
                  - name: body
                    in: body
                    required: true
                    schema:
                      id: Measurements
                      required:
                        - measurements
                      properties:
                        measurements:
                          type: array
                          description: List of installation measurement blocks
                          items:
                            type: object
                            required:
                              - installation_code
                              - values
                            properties:
                              installation_code:
                                type: string
                                description: Installation identifier
                                example: "12345"
                              values:
                                type: array
                                description: Timestamped measurements
                                items:
                                  type: object
                                  required:
                                    - timestamp
                                    - variable_name
                                    - value
                                  properties:
                                    timestamp:
                                      type: string
                                      format: date-time
                                      description: UTC timestamp (ISO8601)
                                      example: "2025-09-19T10:00:00Z"
                                    variable_name:
                                      type: string
                                      enum: [P, Q]
                                      description: >
                                        P = active power (kW)
                                        Q = reactive power (kvar)
                                      example: "P"
                                    value:
                                      type: number
                                      format: float
                                      description: Measurement value
                                      example: 152.7
                responses:
                  201:
                    description: Measurements accepted
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
                    description: Invalid request body
                    schema:
                      type: object
                      properties:
                        code:
                          type: integer
                          example: 0
                        message:
                          type: string
                          example: "'measurements' is a required property"
                  401:
                    description: Unauthorized (invalid or missing token)
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

        # -- Validate JSON Schema (return any problem if detected):
        try:
            jschema.validate(request.get_json(), SEND_MEASUREMENTS)
        except jschema.exceptions.ValidationError as ex:
            self.logger.exception(msg=log_msg_.format(repr(ex)))
            return {"code": 0, "message": repr(ex).replace("\"", "")}, 400

        try:
            # -- Load Cassandra Connection:
            db_con = CassandraDB.get_cassandra_instance(config=self.config)

            # -- Invalid installations list:
            invalid = []
            try:
                invalid = self.insert_measurements(
                    db_con=db_con,
                    request_data=request.json
                )
            except BaseException as ex:
                self.logger.exception(msg=log_msg_.format(repr(ex)))
                return {"code": 0, "message": "Internal Server Error."}, 500
        except BaseException as ex:
            self.logger.exception(
                msg=log_msg_.format(repr(ex)))
            return {"code": 0, "message": "Internal Server Error."}, 500

        if len(invalid) == 0:
            self.logger.info(msg=log_msg_.format("ok"))
            return {"code": 1, "message": "ok"}, 201
        else:
            msg = 'Found invalid clients. {}'.format(','.join(invalid))
            self.logger.warning(msg=log_msg_.format(msg))
            return {"code": 0, "message": "Received invalid installation_code.", "invalid_ids": invalid}, 201

    def insert_measurements(self, db_con, request_data):
        # -- Tables where data will be inserted:
        inst_table = self.config.TABLES["metadata"]
        raw_table = self.config.TABLES["raw"] \
            if self.config.PRODUCTION else 'raw'

        # -- List of installation_id that not previously registered in db:
        invalid_clients = []

        # -- Process 'measurements' field and insert installation data to db:
        for section in request_data["measurements"]:
            # Installation Identifier:
            inst_id = section["installation_code"]
            # Check if installation ID exists in DB:
            query = f"SELECT id " \
                    f"FROM {inst_table} " \
                    f"WHERE id='{inst_id}';"
            installation_data = db_con.read_query(query)

            if installation_data.empty:
                invalid_clients.append(inst_id)
                continue

            # Parse installation data:
            data = pd.DataFrame(section["values"])

            # -- Update client metadata to DB:
            last_updated = dt.datetime.utcnow()
            last_updated_str = last_updated.strftime("%Y-%m-%d %H:%M:%S")
            cql_query = f"UPDATE {inst_table} " \
                        f"SET last_updated='{last_updated_str}' " \
                        f"WHERE id='{inst_id}' " \
                        f"IF EXISTS;"
            db_con.session.execute(cql_query)
            self.logger.debug(f"Inserting data in table {raw_table} ..")
            t_0 = time()
            # -- Insert client data to DB:
            data["id"] = inst_id
            data["timestamp"] = pd.to_datetime(data["timestamp"],
                                               format="%Y-%m-%dT%H:%M:%SZ")
            data["last_updated"] = last_updated
            data["value"] = data["value"].astype(float)
            data.dropna(inplace=True)  # avoid uploading with Null data
            data.rename(columns={"timestamp": "datetime", "variable_name": "register_type"}, inplace=True)
            self.logger.info(f"data head (2):\n{data.head(2)}")
            db_con.insert_query(data,
                                table=raw_table,
                                use_async=False,
                                logger=self.logger)
            self.logger.info(f"Inserting data in table {raw_table} .. ok! "
                              f"({(time()- t_0):4f}s)")

        return invalid_clients

