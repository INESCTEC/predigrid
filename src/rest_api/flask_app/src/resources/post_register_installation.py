import numpy as np
import pandas as pd
import datetime as dt
import jsonschema as jschema

from flask import request
from flask_restful import Resource

from ..common.UserClass import UserClass
from ..common.databases import CassandraDB
from ..common.schemas import REGISTER_INSTALLATION
from ..resources.post_login import token_required_admin


class PostRegisterInstallation(Resource, UserClass):
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
        Register a new installation
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
              id: RegisterInstallation
              required:
                - installation_code
                - country
                - generation
                - installation_type
                - latitude
                - longitude
                - net_power_types
                - source_nwp
                - is_active
                - forecast_algorithm
              properties:
                installation_code:
                  type: string
                  description: Unique installation identifier
                  example: "inst_001"
                country:
                  type: string
                  enum: [portugal, spain, france]
                  description: Country where the installation is located
                  example: "portugal"
                generation:
                  type: number
                  format: float
                  description: Installed generation capacity (MW or kW depending on type)
                  example: 500.0
                installation_type:
                  type: string
                  enum: [load, wind, solar]
                  description: Type of installation
                  example: "solar"
                latitude:
                  type: number
                  format: float
                  description: Latitude of installation
                  example: 41.1579
                longitude:
                  type: number
                  format: float
                  description: Longitude of installation
                  example: -8.6291
                net_power_types:
                  type: array
                  items:
                    type: string
                    enum: [P, Q]
                  description: Power types measured at the installation
                  example: ["P", "Q"]
                source_nwp:
                  type: string
                  description: NWP source provider
                  example: "meteogalicia"
                is_active:
                  type: integer
                  enum: [0, 1]
                  description: >
                    Installation status:
                    - 1 → active
                    - 0 → inactive
                  example: 1
                forecast_algorithm:
                  type: string
                  enum: [lqr, default]
                  description: Forecast algorithm reference
                  example: "default"
        responses:
          201:
            description: Installation successfully registered
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
            description: Invalid request parameters
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "'country' param must be one of the following: ['portugal', 'spain', 'france']"
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
            description: Internal server error (DB or NWP grid failure)
            schema:
              type: object
              properties:
                code:
                  type: integer
                  example: 0
                message:
                  type: string
                  example: "Unable to register. Failed to insert installation metadata."
        """
        # -- Validate JSON Schema (return any problem if detected):
        try:
            jschema.validate(request.get_json(), REGISTER_INSTALLATION)
        except jschema.exceptions.ValidationError as ex:
            self.logger.exception(
                msg=self.log_msg.format(ip=self.client_ip,
                                        user=self.client_usr,
                                        route=self.request_route,
                                        msg=repr(ex)))
            return {"code": 0, "message": repr(ex).replace("\"", "")}, 400

        # Init log msg:
        log_msg = self.log_msg.format(ip=self.client_ip,
                                      user=self.client_usr,
                                      route=self.request_route,
                                      msg="{}")
        # Payload args:
        inst_id = request.json.get('installation_code', None)
        country = request.json.get('country', None)
        generation = request.json.get('generation', None)
        id_type = request.json.get('installation_type', None)
        latitude = request.json.get('latitude', None)
        longitude = request.json.get('longitude', None)
        net_power_types = request.json.get('net_power_types', ["P", "Q"])
        source_nwp = request.json.get('source_nwp', "meteogalicia")
        is_active = request.json.get('is_active', None)
        f_algorithm = request.json.get('forecast_algorithm', "default")

        country = country.lower()
        if country not in ["portugal", "spain", "france"]:
            return {"code": 0,
                    "message": "'country' param must be one of the following: "
                               "['portugal', 'spain', 'france']"}, 400

        id_type = id_type.lower()
        if id_type not in ['load', 'wind', 'solar']:
            return {"code": 0,
                    "message": "'installation_type' param must be one of the following: "
                               "['load', 'wind', 'solar']"}, 400

        net_power_types = ''.join(net_power_types)
        if net_power_types not in ["P", "Q", "PQ"]:
            return {"code": 0,
                    "message": "'net_power_types' param must be one of "
                               "the following: 'P', 'Q' or 'PQ' "
                               "(case sensitive)"}, 400

        if is_active not in [0, 1]:
            return {"code": 0,
                    "message": "'is_active' param must be one of "
                               "the following: 1 - active, 0 - inactive"}, 400

        if f_algorithm not in ["lqr", "default"]:
            return {"code": 0,
                    "message": "Forecast algorithm must be one of the "
                               "following: 'lqr' or 'default'"}, 400

        # Connect to Cassandra DB:
        try:
            db_con = CassandraDB.get_cassandra_instance(config=self.config)
        except BaseException as ex:
            msg = f"Failed to register installation {inst_id}. " \
                  f"ERROR: {repr(ex)}."
            self.logger.error(msg=log_msg.format(msg))
            return {"code": 0,
                    "message": "Failed to register installation."}, 500

        try:
            # Find nearby NWP coordinates for actual location:
            latitude_nwp, longitude_nwp = self.find_nearby_nwp_coordinates(
                db_con=db_con,
                log_msg=log_msg,
                latitude=latitude,
                longitude=longitude
            )
        except BaseException as ex:
            msg = f"Failed to find nearby nwp coordinates for installation {inst_id}. " \
                  f"ERROR: {repr(ex)}."
            self.logger.error(msg=log_msg.format(msg))
            return {"code": 0,
                    "message": "Unable to register. Failed to find nearby coordinates."}, 500

        try:
            # Insert installation metadata in DB:
            self.insert_installation_metadata(
                db_con=db_con,
                log_msg=log_msg,
                inst_id=inst_id,
                country=country,
                generation=generation,
                id_type=id_type,
                latitude=latitude,
                latitude_nwp=latitude_nwp,
                longitude=longitude,
                longitude_nwp=longitude_nwp,
                net_power_types=net_power_types,
                source_nwp=source_nwp,
                is_active=is_active
            )
        except BaseException as ex:
            msg = f"Failed to register installation {inst_id}. " \
                  f"ERROR: {repr(ex)}."
            self.logger.error(msg=log_msg.format(msg))
            return {"code": 0,
                    "message": "Unable to register. Failed to insert installation metadata."}, 500

        try:
            self.insert_model_ref(
                db_con=db_con,
                log_msg=log_msg,
                inst_id=inst_id,
                id_type=id_type,
                f_algorithm=f_algorithm
            )
        except BaseException as ex:
            msg = f"Failed to register model_ref for installation {inst_id}. " \
                  f"ERROR: {repr(ex)}."
            self.logger.error(msg=log_msg.format(msg))
            return {"code": 0,
                    "message": "Unable to register. Failed to insert model_ref."}, 500

        return {"code": 1, "message": "ok"}, 201

    @staticmethod
    def __get_nearest_coords_idx(grid, latitude, longitude):
        distances = (np.sqrt(
            (grid['latitude'] - latitude) ** 2 + (
                    grid['longitude'] - longitude) ** 2))

        return distances.idxmin()

    def find_nearby_nwp_coordinates(self, db_con, log_msg,
                                    latitude, longitude):
        # NWP grid table:
        nwp_grid_table = self.config.TABLES["nwp_grid_table"]
        # Get information about available NWP grid
        grid = db_con.read_query(f"select * from {nwp_grid_table};")
        # Compute nearest NWP grid coordenates to installation's location
        ncidx = self.__get_nearest_coords_idx(grid, latitude, longitude)
        latitude_nwp, longitude_nwp = grid.loc[ncidx, ['latitude', 'longitude']]
        # If NWP grid point not active, update db table
        if not grid.loc[ncidx, 'is_active']:
            grid.loc[ncidx, 'is_active'] = True
            db_con.insert_query(grid.loc[ncidx].to_frame().T, nwp_grid_table)
            msg = f"New coordinate pair ({latitude_nwp},{longitude_nwp}) activated in NWP grid."
            self.logger.info(msg=log_msg.format(msg))
        return latitude_nwp, longitude_nwp

    def insert_installation_metadata(self, db_con, log_msg, inst_id, country,
                                     generation, id_type, latitude,
                                     latitude_nwp, longitude, longitude_nwp,
                                     net_power_types, source_nwp, is_active):
        inst_table = self.config.TABLES["metadata"]
        last_updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        q = f"INSERT INTO {inst_table} (" \
            f"id, country, " \
            f"generation, id_type, " \
            f"last_updated, " \
            f"latitude, latitude_nwp," \
            f"longitude, longitude_nwp," \
            f"net_power_types, source_nwp, is_active) " \
            f"VALUES (" \
            f"'{inst_id}', '{country}', " \
            f"{generation}, '{id_type}', " \
            f"'{last_updated}', " \
            f"{latitude}, {latitude_nwp}, " \
            f"{longitude}, {longitude_nwp}, " \
            f"'{net_power_types}', '{source_nwp}', " \
            f"{is_active});"

        # Perform insertion:
        db_con.execute_query(q)
        msg = f"Installation {inst_id} registered with success in " \
              f"table '{inst_table}'"
        self.logger.info(msg=log_msg.format(msg))

    def insert_model_ref(self, db_con, log_msg, inst_id, id_type, f_algorithm):
        from json import load
        from os.path import join
        from os.path import dirname as dd
        ref_path = join(dd(dd(dd(__file__))),
                        "static", "default_model_ref", f"{id_type}.json")

        with open(ref_path, "r") as f:
            default_model_ref = load(f)

        if f_algorithm == "lqr":
            # Replace 'gbt' refs. by proposed 'lqr' then reconvert to dict:
            default_model_ref = eval(str(default_model_ref).replace("gbt", f_algorithm))

        models_table = self.config.TABLES["models_info"]
        last_updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        q = f"INSERT INTO {models_table} (" \
            f"id, " \
            f"last_train, " \
            f"last_updated, " \
            f"model_in_use, " \
            f"model_ref, " \
            f"to_train" \
            f") " \
            f"VALUES (" \
            f"'{inst_id}', " \
            f"'1970-01-01 00:00:00', " \
            f"'{last_updated}', " \
            f"{default_model_ref}, " \
            f"{default_model_ref}, " \
            f"1);"

        # Perform insertion:
        db_con.execute_query(q)
        msg = f"Installation {inst_id} model_ref registered with success in " \
              f"table '{models_table}'"
        self.logger.info(msg=log_msg.format(msg))
