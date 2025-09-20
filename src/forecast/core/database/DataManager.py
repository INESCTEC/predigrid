import gc
import cassandra
import numpy as np
import pandas as pd

from typing import NoReturn

from cassandra.connection import ConnectionException
from core.database.DatabaseConfig import DatabaseConfig
from core.forecast.helpers.holidays_finder import HolyFinder

from core.database.helpers.nwp_queries import query_nwp_data
from core.database.helpers.load_queries import query_measurements
from core.database.helpers.metadata_queries import (
    query_models_metadata,
    query_installation_metadata
)


class DataManager:
    """
        This Class contains the necessary functions to interact with the
        project Cassandra Database

        The following class variables are available:

        * DATABASE_EXCEPTIONS: (:class:`list`) List of DB exceptions
        * main_logs_table: (:obj:`str`) Table for task logs (high-level logs)
        * models_logs_table: (:obj:`str`) Table for models logs
        (low-level logs)

        The main methods of this class are:

        * :meth:`~load_configs`:
        * :meth:`~get_dataset`:
        * :meth:`~get_statistics`:


    """

    DATABASE_EXCEPTIONS = (cassandra.ReadFailure, cassandra.ReadTimeout,
                           cassandra.OperationTimedOut, cassandra.Timeout,
                           cassandra.FunctionFailure, cassandra.Timeout,
                           cassandra.WriteFailure, cassandra.WriteTimeout,
                           ConnectionException)

    config = DatabaseConfig()

    # -- Installations Table:
    installations_table = config.TABLES["installations"]
    models_info_table = config.TABLES["models_info"]

    # -- Logs Tables:
    main_logs_table = config.TABLES["load_main_logs"]
    models_logs_table = config.TABLES["load_models_logs"]

    # -- Stored Objects Table:
    objects_table = config.TABLES["objects"]

    # -- Use Ficticious data:
    unit_tests_mode = False

    def __repr__(self):
        return "DataManager"

    def activate_unit_tests_mode(self) -> NoReturn:
        """
        Change flag indicating this class is being used in unit tests.

        """
        self.unit_tests_mode = True

    def __init__(self,
                 db_con,
                 inst_id: str,
                 ) -> NoReturn:
        """
        Initialize DataManager instance base attributes

        Args:

            db_con: (:obj:`class`) Database Engine
            inst_id: (:obj:`str`) Installation Identifier
            forecast_horizon: (:obj:`int`) Forecast horizon (in hours)
            mode: (:obj:`str`) "train" or "forecast"
        """
        # -- Initialize class attributes:
        # ---- Database connection:
        self.engine = db_con

        # ---- Forecast Configs:
        self.launch_time = None
        self.launch_time_hour_utc = None
        self.f_horizon = None
        self.forecast_range = None
        self.mode = None
        self.use_full_historical = None
        self.last_date_horizon_utc = None

        # ---- Installation Info:
        self.inst_id = str(inst_id)
        self.inst_metadata = {}
        self.inst_type = None
        self.inst_capacity = 0
        self.country = None
        self.forecast_tz = None
        self.f_targets = []

        # ---- Data Trackers:
        self.data_tracker = {
            "P": False,
            "Q": False,
            "NWP": False,
        }

        # ---- NWP data structures:
        self.nwp_vars = None

        # ---- Forecasting models aux. information:
        self.models_configs = {}
        self.models_metadata = {}

    def set_launch_time(self, launch_time: str) -> NoReturn:
        """
        Initialize launch time object with truncated minute/second/microseconds

        :param launch_time: Datetime in UTC timezone
        """
        # Parse_launch_time
        self.launch_time = pd.to_datetime(
            launch_time,
            format="%Y-%m-%d %H:%M:%S",
            utc=True,
        )
        self.launch_time_hour_utc = self.launch_time.replace(
            minute=0,
            second=0,
            microsecond=0
        )

    def set_forecast_horizon(self, forecast_horizon: int = 168) -> NoReturn:
        """
        Set dates in forecast horizon (counting since launch time)

        :param forecast_horizon: The forecast horizon in hours.
        """
        if forecast_horizon > 168:
            raise ValueError("Error! Forecast horizon limit is 168.")
        if self.launch_time_hour_utc is None:
            raise AttributeError("Error! Must set launch time first.")
        self.f_horizon = forecast_horizon
        # -- get dates in forecast horizon:
        self.forecast_range = pd.date_range(
            start=self.launch_time_hour_utc,
            end=self.launch_time_hour_utc + pd.DateOffset(
                hours=self.f_horizon),  # noqa
            freq='H',
            tz="UTC"
        )[:-1]
        # -- Convert to forecast timezone:
        self.last_date_horizon_utc = self.forecast_range[-1]

    def set_mode(self, mode: str) -> NoReturn:
        """
        Define mode for DataManager.

        :param mode: "train" or "forecast"
        """
        self.mode = mode
        if mode not in ["train", "forecast"]:
            raise ValueError("mode must be train or forecast")
        # use_full_historical bool flag:
        # True - loads 2 years of historical data. False - loads last 30 days
        self.use_full_historical = self.mode == "train"

    def __query_installation_metadata(self) -> NoReturn:
        """
        Query installation metadata & unpack it to class attributes
        """
        self.inst_metadata = query_installation_metadata(
            conn=self.engine,
            table=self.installations_table,
            inst_id=self.inst_id,
            use_mock_data=self.unit_tests_mode
        )
        # Unpack installation metadata:
        self.country = self.inst_metadata["country"]
        self.inst_type = self.inst_metadata["id_type"]
        self.inst_capacity = self.inst_metadata["installed_capacity"]
        self.source_nwp = self.inst_metadata["source_nwp"]
        self.latitude_nwp = self.inst_metadata["latitude_nwp"]
        self.longitude_nwp = self.inst_metadata["longitude_nwp"]
        self.has_generation = bool(self.inst_metadata["generation"])
        self.net_power_types = self.inst_metadata["net_power_types"]

    def __query_models_metadata(self) -> NoReturn:
        # Unpack models metadata:
        self.models_metadata = query_models_metadata(
            conn=self.engine,
            table=self.models_info_table,
            inst_id=self.inst_id,
            use_mock_data=self.unit_tests_mode
        )

    def __set_forecast_timezone(self) -> NoReturn:
        """
        Select forecast timezone from country metadata info
        """
        _selector = {
            "portugal": "Europe/Lisbon",
            "spain": "Europe/Madrid",
            "france": "Europe/Paris",
        }
        self.forecast_tz = _selector[self.country]

    def __set_register_types(self) -> NoReturn:
        """
        Select register types and define forecast targets
        """
        # Register Type Options:
        _rtype_lst = {
            "P": ["P"],
            "Q": ["Q"],
            "PQ": ["P", "Q"],
        }
        self.register_types = sorted(_rtype_lst[self.net_power_types])
        self.f_targets = [f"real_{x}" for x in self.register_types]

    def __set_nwp_variables(self) -> NoReturn:
        """
        Define NWP variables to query depending on installation type
        """
        # Variables to query:
        _nwp_vars = {
            "load": ["temp"],
            "load_gen": ["temp", "swflx", "cfl", "cfm"],
            "solar": ["swflx", "cfl", "cfm", "cfh", "cft"],
            "wind": [
                'u', 'ulev1', 'ulev2', 'ulev3',
                'v', 'vlev1', 'vlev2', 'vlev3'
            ]
        }
        if self.inst_type == "load":
            _subtype = "load_gen" if self.has_generation else "load"
            self.nwp_vars = _nwp_vars[_subtype]
        else:
            self.nwp_vars = _nwp_vars[self.inst_type]

    def __set_db_tables(self) -> NoReturn:
        # -- Tables:
        # If inst type is 'load' fetches data from solar table (temp, swflx)
        _nwp_tbl = f"nwp_{self.inst_type}" if self.inst_type != "load" else "nwp_solar"  # noqa
        self.tables = dict(
            measurements=self.config.TABLES["measurements"],
            forecasts=self.config.TABLES["forecasts"],
            nwp=self.config.TABLES[_nwp_tbl]
        )

    def __load_holidays_info(self) -> NoReturn:
        """
        Load holiday info for current country & forecast horizon
        """
        # -- Holidays Class:
        self.holyclass = HolyFinder(
            launch_time=self.launch_time_hour_utc,
            country=self.country
        )
        # -- Check for holidays in forecast range:
        self.has_holidays = self.holyclass.holidays_in_time_period(
            start=self.forecast_range[0],
            end=self.forecast_range[-1]
        )
        if self.has_holidays:
            self.holidays_in_period = self.holyclass.get_holidays_in_period(
                start=self.forecast_range[0],
                end=self.forecast_range[-1]
            )
            self.bridgedays_in_period = self.holyclass.get_bridges_in_period(
                start=self.forecast_range[0],
                end=self.forecast_range[-1]
            )
        else:
            self.holidays_in_period = []
            self.bridgedays_in_period = []

    def query_measurements(self,
                           historical_start_utc: pd.Timestamp,
                           register_type: str,
                           ) -> pd.DataFrame:
        """

        Query measurements data from DB.

        :param historical_start_utc: timestamp of starting date
        in historical data
        :param register_type: Active (P) or reactive (Q) power
        :return: Dataset of historical data
        """
        return query_measurements(
            conn=self.engine,
            inst_id=self.inst_id,
            table=self.tables["measurements"],
            register_type=register_type,
            date_start_utc=historical_start_utc,
            date_end_utc=self.launch_time_hour_utc,
            use_mock_data=self.unit_tests_mode
        )

    def query_nwp_variables(self,
                            variables: list,
                            historical_start_utc: pd.Timestamp,
                            ) -> pd.DataFrame:
        """
        Query NWP variables data from DB

        :param variables: list of target variables
        :param historical_start_utc: timestamp of starting date
        in historical data
        :return: Dataset of NWP variables
        """
        # End date should be last date in horizon + 3 hours
        # (for feature engineering -> lag creation)
        _end_date = self.last_date_horizon_utc + pd.DateOffset(hours=3)
        # Retrieve NWP data from the database:
        return query_nwp_data(
            conn=self.engine,
            source_nwp=self.source_nwp,
            table=self.tables["nwp"],
            latitude=self.latitude_nwp,
            longitude=self.longitude_nwp,
            variables=variables,
            date_start_utc=historical_start_utc,
            date_end_utc=_end_date,
            use_mock_data=self.unit_tests_mode
        )

    def __valid_hours_in_hist(self,
                              dataset: pd.DataFrame,
                              target: str,
                              ) -> int:
        """
        Count number of valid hours in current dataset
        """
        return len(dataset[[target] + self.nwp_vars].dropna(how="any").index)

    def __check_historical_start(self) -> NoReturn:
        """
        Returns historical start reference to download either:
          1) 2 years of historical data
          2) 30 days of historical data

        When to download all historical data:
          - If train mode is activated (self.use_full_historical is True)
          - There is at least one holiday in forecast horizon
        """
        if self.use_full_historical or self.has_holidays:
            return self.launch_time_hour_utc - pd.DateOffset(years=2)
        else:
            return self.launch_time_hour_utc - pd.DateOffset(days=31)

    def __get_avg_profiles(self,
                           dataset: pd.DataFrame,
                           target: str,
                           ) -> (pd.DataFrame, pd.DataFrame):
        """
        Create average profiles for 1) weekday and 2) weekend
        """
        _start = self.launch_time_hour_utc - pd.DateOffset(days=31)
        _df = dataset.loc[_start:self.launch_time_hour_utc, target].copy()
        _df_weekday = _df.loc[_df.index.weekday < 5, ]
        _df_weekend = _df.loc[_df.index.weekday >= 5, ]
        _df_weekday = _df_weekday.groupby(_df_weekday.index.hour).mean()
        _df_weekend = _df_weekend.groupby(_df_weekend.index.hour).mean()
        return _df_weekday, _df_weekend

    def __add_avg_profiles_to_configs(self,
                                      dataset: pd.DataFrame,
                                      f_horizons: dict,
                                      target: str,
                                      ) -> dict:
        """
        - Create average profiles for 1) weekday and 2) weekend
        - Associate profiles to each forecast horizon
        """
        _profile_map = lambda x: _df_weekday[x.hour] if x.weekday() < 5 else _df_weekend[x.hour]  # noqa
        _df_weekday, _df_weekend = self.__get_avg_profiles(
            dataset=dataset,
            target=target
        )
        avg_profiles = {}
        for k, v in f_horizons.items():
            # Map func to profile series:
            _map_obj = map(_profile_map, v)
            # Remove NaNs and store:
            avg_profiles[k] = [x for x in _map_obj if not np.isnan(x)]
        return avg_profiles

    def __remove_holidays(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Create holidays & bridges columns references
            - True = holiday
            - False = Normal day

        If no holidays are available in the forecast horizon, discards holidays
        timestamps (not needed to train new model)
        Else, keeps holidays information (to calc holidays profile later on)

        :param dataset: Dataset to modify
        :return: Modified dataset
        """
        # -- Holidays Preprocessing (Flags Bridge and Holy days):
        dataset.loc[:, "is_holy"] = dataset.index.map(lambda x: self.holyclass.is_holiday(date=x))  # noqa
        dataset.loc[:, "is_bridge"] = dataset.index.map(lambda x: self.holyclass.is_bridge(date=x))  # noqa
        if (not self.has_holidays) or (self.mode == "train"):
            # If there are no holidays in forecast horizon
            # removes holidays and bridge days from historical data
            dataset.loc[dataset["is_holy"], self.f_targets] = np.nan
        return dataset

    def __get_forecast_models_id(self,
                                 search_dates: pd.DatetimeIndex,
                                 avail_dates_measur: pd.DatetimeIndex,
                                 avail_dates_nwp: pd.DatetimeIndex,
                                 avg_profile: list,
                                 register_type: str,
                                 ) -> (str, list):
        """
        Possible return scenarios:
        Active power:
            * D-X/D-7 -> Use conventional model
            * D-7 -> Use conventional model (without most recent lag)
            * D-X (x->[1,6]) and avg_profile length != 24: returns "backup"
            * backup -> No recent (base_models_ref_lags) lags nor week lags
            (week_suffix_lags)

        Reactive power:
            * D-7 -> Use conventional model
            * backup -> No recent (base_models_ref_lags) lags nor week lags
            (week_suffix_lags)

        :param search_dates: Set of reference dates
        :param avail_dates_measur: Subset dates for available
        historical measures
        :param avail_dates_nwp: Subset dates for available historical NWP
        :param avg_profile: Average measures per day of week
        :param register_type: Active (P) or reactive (Q)

        """
        # -- Forecast models and expected lags:
        base_models_ref_lags = {
            1: "D-1",
            2: "D-2",
            3: "D-3",
            -1: "",
        }
        week_suffix_lags = {
            7: "D-7",
            -1: ""
        }
        #
        #########################################
        # Check if average profile is available:
        #########################################
        _has_avg_profile = len(avg_profile) == 24
        ##############################
        # Check if NWP are available:
        ##############################
        _has_nwp, _nwp_counter = self.__check_if_dates_exist(
            search_dates=search_dates,
            avail_dates=avail_dates_nwp,
            min_matches=len(search_dates),
            return_counter=True
        )
        #############################################
        # Search Base Lag (D-X) , x between 1 and 3
        #############################################
        _base_lag = -1
        if register_type == "P":
            # -- Check which base lag exists :
            _max_search_days = max(base_models_ref_lags.keys())
            lag_search_space = [x for x in range(1, _max_search_days + 1)]
            for day_diff in lag_search_space:
                _search = search_dates.tz_convert('UTC') - pd.DateOffset(days=day_diff) # noqa
                _search = _search.tz_convert(self.forecast_tz)
                if self.__check_if_dates_exist(_search, avail_dates_measur, min_matches=len(_search) - 2): # Only allow 2 dates missing so interpolation when forecasting resolves the issue # noqa
                    _base_lag = day_diff
                    break
        _base_lag_name = base_models_ref_lags.get(_base_lag, "")
        ########################
        # Search Week Lag (D-7)
        ########################
        _week_lag = -1
        if _has_avg_profile:
            # If has average profile, D-7 is considered as "available"
            _week_lag = 7
        else:
            # If no avg. profile, tries to search real lag
            # -- Check which week lag exists (defines backup prefix):
            _week_lag = -1
            lag_search_space = [7]  # , 14
            for day_diff in lag_search_space:
                _search = search_dates.tz_convert('UTC') - pd.DateOffset(days=day_diff) # noqa
                _search = _search.tz_convert(self.forecast_tz)
                if self.__check_if_dates_exist(_search, avail_dates_measur):
                    _week_lag = day_diff
                    break
        _week_lag_name = week_suffix_lags.get(_week_lag, "")
        #########################################################
        # Decide to activate backup model or conventional model:
        #########################################################
        if _has_nwp:
            # Has sufficient NWP to use normal models:
            if _week_lag == -1:
                base_model = "backup_weather"
            else:
                # Else, business as usual
                base_model = [_base_lag_name, _week_lag_name]
                base_model = '/'.join([x for x in base_model if x != ""])
        else:
            # Does not have sufficient NWP to use normal models:
            if (_week_lag == -1) and (_nwp_counter > 12):
                base_model = "backup_mix"
            elif (_week_lag == -1) and (_nwp_counter <= 12):
                base_model = "backup_no_weather"
            else:
                # week_lag exists:
                base_model = "D-7/no_weather"

        ####################################################################
        # Return base model and lag references (base and week) if existent:
        ####################################################################
        lag_references = [x for x in [_base_lag, _week_lag] if x != -1]
        return base_model, lag_references

    @staticmethod
    def __check_if_dates_exist(search_dates: pd.DatetimeIndex,
                               avail_dates: pd.DatetimeIndex,
                               min_matches: int = 20,
                               return_counter: bool = False):
        """
        Intersection between timestamp arrays
        """
        _found_dt = avail_dates.intersection(search_dates)
        if return_counter:
            return len(_found_dt) >= min_matches, len(_found_dt)
        else:
            return len(_found_dt) >= min_matches

    def __split_horizon_per_model(self) -> dict:
        """
        Split horizon and associate each batch to a model
        Example:
            - first 0h to 24h - model "D"
            - second 24h to 48h - model "D+1"
            - ...
        """
        # split forec horizon in 24h batches:
        _splits = np.array_split(self.forecast_range, np.arange(24, 168, 24))
        # associate each 24h batch to model:
        _models_names = ["D", "D+1", "D+2", "D+3", "D+4", "D+5", "D+6"]
        models_horizon = {}
        for i, model_id in enumerate(_models_names):
            if len(_splits[i]) > 0:
                models_horizon[model_id] = _splits[i]
        return models_horizon

    def __get_models_res(self, dataset: pd.DataFrame) -> dict:
        _nwp_not_null = dataset[self.nwp_vars].dropna().index
        models = {}
        _nr_ts_nwp = len(_nwp_not_null.intersection(self.forecast_range))
        if _nr_ts_nwp == len(self.forecast_range):
            # NWP available for each element in forecast horizon
            models["D+X"] = "weather"
        elif _nr_ts_nwp > 12:
            # NWP available for at least half of ts in horizon
            models["D+X"] = "mix"
        else:
            # Other (no or short NWP availability)
            models["D+X"] = "no_weather"
        return models

    def __get_models_lag_ref(self,
                             dataset: pd.DataFrame,
                             models_horizon: dict,
                             avg_profiles: dict,
                             target: str,
                             register_type: str,
                             ) -> (dict, list):
        """
        Get available lag references for each model
        """
        if (not self.data_tracker[register_type]) and (self.mode == "train"):
            return {}, {}

        _idx_measurements_not_null = dataset[target].dropna().index
        _idx_nwp_not_null = dataset[self.nwp_vars].dropna().index
        # check available lags inputs for each forecasting model:
        models = {}
        _ref_lags_list = []

        # search available model/lag pairs:
        for model_id, horizon in models_horizon.items():
            models[model_id], ref_lags = self.__get_forecast_models_id(
                search_dates=horizon,
                avail_dates_measur=_idx_measurements_not_null,
                avail_dates_nwp=_idx_nwp_not_null,
                avg_profile=avg_profiles[model_id],
                register_type=register_type
            )
            # append reference input lags
            _ref_lags_list += ref_lags

        # List of unique lags to be computed over measurements data
        valid_lag_list = list(set([x for x in _ref_lags_list if x != -1]))
        return models, valid_lag_list

    def __validate_initial_preconditions(self):
        # -- Check initial pre-conditions:
        if self.mode is None:
            raise NameError("Error! DataManager 'mode' is not defined.")
        if self.launch_time_hour_utc is None:
            raise NameError("Error! DataManager 'launch_time' is not defined.")
        if self.forecast_range is None:
            raise NameError(
                "Error! DataManager 'forecast_range' is not defined."
            )

    def load_configs(self) -> NoReturn:
        """
        Load installation metadata and initialize configurations
        """
        # -- Validate initial pre-conditions:
        self.__validate_initial_preconditions()
        # -- Load installation metadata:
        self.__query_installation_metadata()
        # -- Query models metadata:
        self.__query_models_metadata()
        # -- Set forecast timezone:
        self.__set_forecast_timezone()
        # -- Set forecast register types & targets:
        self.__set_register_types()
        # -- Set NWP variables:
        self.__set_nwp_variables()
        # -- Set necessary DB tables:
        self.__set_db_tables()
        # -- Load holidays info:
        self.__load_holidays_info()

    def __get_model_configs_res(self,
                                dataset: pd.DataFrame,
                                target: str,
                                rtype: str,
                                ) -> NoReturn:
        # Get number of valid hours in history:
        _hours_in_hist = self.__valid_hours_in_hist(
            dataset=dataset,
            target=target
        )
        # -- Get Lag Reference:
        _models = self.__get_models_res(dataset=dataset)
        # -- Set models configs:
        self.models_configs[rtype] = {
            "models": _models,
            "horizon": {"D+X": self.forecast_range},
            "hours_in_hist": _hours_in_hist
        }

    def __get_model_configs_load(self,
                                 dataset: pd.DataFrame,
                                 target: str,
                                 rtype: str,
                                 f_horizons: dict,
                                 ) -> NoReturn:
        # Get number of valid hours in history:
        _hours_in_hist = self.__valid_hours_in_hist(
            dataset=dataset,
            target=target
        )
        # Get avg profiles (week or weekend depending on horizon weekday):
        _avg_profiles = self.__add_avg_profiles_to_configs(
            dataset=dataset,
            f_horizons=f_horizons,
            target=target
        )
        # -- Get Lag Reference:
        _models, _valid_lag_list = self.__get_models_lag_ref(
            dataset=dataset,
            models_horizon=f_horizons,
            avg_profiles=_avg_profiles,
            target=target,
            register_type=rtype
        )
        # Set models configs:
        self.models_configs[rtype] = {
            "models": _models,
            "lag_ref": _valid_lag_list,
            "horizon": f_horizons,
            "avg_profiles": _avg_profiles,
            "hours_in_hist": _hours_in_hist
        }

    def get_statistics(self, dataset: pd.DataFrame) -> dict:
        """
        Builds a set of statistic descriptors of the data: max and min values,
        5th and 95th quantiles.

        :param dataset: Dataset to compute statistics
        :return: Dictionary with statistics of the historical data
        """
        _st = self.launch_time_hour_utc - pd.DateOffset(weeks=1)
        aux_df = dataset.loc[_st:self.launch_time_hour_utc, self.f_targets]
        if aux_df.dropna().empty:
            aux_df = dataset.loc[:, self.f_targets]
        stats = {"real_P": {}, "real_Q": {}}
        for t in self.f_targets:
            stats[t]["max"] = aux_df[t].max()
            stats[t]["min"] = aux_df[t].min()
            stats[t]["q95"] = aux_df[t].quantile(0.95)
            stats[t]["q05"] = aux_df[t].quantile(0.05)
            stats[t]["limit_max"] = stats[t]["q95"] * 1.5
            stats[t]["limit_min"] = stats[t]["q05"] * 1.5
        return stats

    def get_dataset(self) -> (pd.DataFrame, dict):
        """
        1. Retrieves the DataSet required for a new forecast. This includes:
            - Power measurements data (P and/or Q);
            - NWP variables data
        2. Holidays processing
        3. Average profile calculation (weekday and weekend)
        4. Model configurations definition

        :return: Tuple with dataset and model configurations
        """
        # -- Query metadata for current installation:
        # Note: Any exception here will break the program for this installation
        self.load_configs()

        # Get historical start reference:
        _historical_start_utc = self.__check_historical_start()

        # -- Initialize Empty Dataset Structure:
        dataset = pd.DataFrame(
            index=pd.date_range(
                start=_historical_start_utc,
                end=self.last_date_horizon_utc,
                freq='H',
                tz='UTC')
        )

        for rtype in self.register_types:
            try:
                # -- Query load measurements for current installation:
                _measurements_df = self.query_measurements(
                    historical_start_utc=_historical_start_utc,
                    register_type=rtype
                )
                self.data_tracker[rtype] = not _measurements_df.dropna().empty
                dataset = dataset.join(_measurements_df, how="outer")
            except self.DATABASE_EXCEPTIONS as ex:
                raise ex
            except BaseException as ex:
                print(f"ERROR load_measurements ({self.inst_id})\n{repr(ex)}")

        try:
            # Query weather information:
            _nwp_df = self.query_nwp_variables(
                variables=self.nwp_vars,
                historical_start_utc=_historical_start_utc
            )
            # Update DataFrame with weather
            self.data_tracker["NWP"] = not _nwp_df.dropna().empty
            dataset = dataset.join(_nwp_df, how="outer")
        except self.DATABASE_EXCEPTIONS as ex:
            raise ex
        except BaseException as ex:
            print(f"ERROR query_nwp data ({self.inst_id})\n{repr(ex)}")

        # -- Create model configs dictionary (different for RES or LOAD):
        if self.inst_type in ["solar", "wind"]:
            # -- If it is RES Forecast:
            for rtype in self.register_types:
                _tar = "real_" + rtype
                self.__get_model_configs_res(
                    dataset=dataset,
                    target=_tar,
                    rtype=rtype,
                )
        else:
            # -- If it is Load Forecast:
            # Convert dataset dates to forecast timezone:
            dataset = dataset.tz_convert(self.forecast_tz)
            # Convert forecast horizon dates to forecast timezone:
            self.forecast_range = self.forecast_range.tz_convert(self.forecast_tz)  # noqa
            # Remove (or keep) measurements at holidays:
            dataset = self.__remove_holidays(dataset=dataset)
            for rtype in self.register_types:
                _tar = "real_" + rtype
                # Get forecast horizon per model:
                _horizon_per_model = self.__split_horizon_per_model()
                self.__get_model_configs_load(
                    dataset=dataset,
                    target=_tar,
                    rtype=rtype,
                    f_horizons=_horizon_per_model
                )

        del _measurements_df
        del _historical_start_utc
        del _nwp_df
        gc.collect()

        return dataset, self.models_configs
