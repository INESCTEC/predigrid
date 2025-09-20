import re
import pandas as pd
from pathlib import Path
from typing import Optional, Union, NoReturn

from forecast_api.util.databases import CassandraDB

from core.forecast.MvModelClass import MvModel
from core.forecast.load_forecast import (
    MvModelGBT,
    MvModelLQR,
    MvModelNaive,
)
from core.forecast.res_forecast import (
    PVForecast,
    WindForecast,
    PVBackupForecast
)
from core.forecast.custom_exceptions import (
    simple_fail_proof,
    LoadScalerError,
    ModelLoadException,
    ModelTrainException,
    ModelForecastException,
)
from core.database.DatabaseConfig import DatabaseConfig
from core.database.DataManager import DataManager
from core.forecast.helpers.model_helpers import (
    save_local_model, load_local_model,
    save_local_scalers, load_local_scalers,
    save_local_stats, load_local_stats,
    save_db_model, load_db_model,
    save_db_scalers, load_db_scalers,
    save_db_stats, load_db_stats,
    clean_models_dir
)
from core.forecast.custom_exceptions import (
    TrainEmptyDatasetError
)

from core.forecast.helpers.model_helpers import postprocessing_layer
from core.log.Log import create_log, just_console_log


class ForecastManager:
    """
    1. Select which model to use based on DataManager feedback
    2. Run children classes pipelines (for selected models)
    3. Check & apply holidays processing on forecasted data
    4. Store forecasts in database

    """

    # -- Stored Objects Table:
    forecasts_table = DatabaseConfig().TABLES["forecasts"]

    # -- Algorithms mapping:
    algorithm_map = {
        "gbt": MvModelGBT,
        "lqr": MvModelLQR,
        "gbt_solar": PVForecast,
        "gbt_wind": WindForecast,
        "naive": MvModelNaive,
        "pv_backup": PVBackupForecast
    }

    unit_tests_mode = False

    def activate_unit_tests_mode(self) -> NoReturn:
        """
        Change flag indicating this class is being used in unit tests.

        """
        # Useful for exception handling (see exception wrapper)
        self.unit_tests_mode = True

    def set_local_models_path(self, path: Union[Path, str]) -> NoReturn:
        """
        Defines where to store models on machine.

        :param path: Path to desired location for models
        """
        self.models_path = path

    def __init__(self):
        self.configs = {}
        self.db_manager = None
        self.model_ref = None
        self.model_in_use = None
        self.dataset = pd.DataFrame()
        self._rem_dataset = pd.DataFrame() # To keep part of dataset. Needed for forecasts with holidays # noqa
        self.P_forecasts = pd.DataFrame()
        self.algorithms = {}
        self.inst_id = None
        self.inst_type = None
        self.inst_capacity = 0
        self.db_con = None
        self.models_path = None
        self.dataset_stats = dict()
        self._logger = None
        self._model_location = "database"

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        elif self.inst_id is None:
            return just_console_log()
        else:
            self._logger = create_log(inst_id=self.inst_id,
                                      service='forecast')
        return self._logger

    @staticmethod
    def __verify_db_manager(db_manager):
        if db_manager.models_metadata == {}:
            raise ValueError("DataManager has no metadata for models.")
        elif db_manager.models_metadata['model_ref'] in [None, {}]:
            raise ValueError("DataManager has no reference models.")
        elif db_manager.models_metadata['model_in_use'] in [None, {}]:
            raise ValueError("DataManager has no listed models for use.")
        if db_manager.inst_type is None:
            raise ValueError("DataManager does not discriminate installation type") # noqa

    def set_model_location(self, location="database"):
        if location not in ["database", "local"]:
            raise ValueError("Model location options are: \"database\" or \"local\"") # noqa
        self._model_location = location

    def assign_dbmanager(self, db_manager: DataManager) -> NoReturn:
        """
        Stores DataManager class internally and creates additional attributes
        from the class

        :param db_manager: DataManager instance to store.
        """
        self.__verify_db_manager(db_manager)
        self.db_manager = db_manager
        self.model_ref = self.db_manager.models_metadata['model_ref']
        self.model_in_use = self.db_manager.models_metadata['model_in_use']
        self.db_con = self.db_manager.engine
        self.inst_id = self.db_manager.inst_id
        self.inst_type = self.db_manager.inst_type
        self.inst_capacity = self.db_manager.inst_capacity

    def assign_dataset(self, dataset: pd.DataFrame, consider_forecast_mode=True) -> NoReturn: # noqa
        """
        Stores dataset internally. Assumes a DataManager is loaded.
        """
        # If in "forecast" mode, will keep data more than an month old apart
        if consider_forecast_mode and self.db_manager.mode == "forecast":
            launch_time_utc = self.db_manager.launch_time_hour_utc
            a_month_ago = launch_time_utc - pd.DateOffset(days=31)
            self._rem_dataset = dataset.loc[dataset.index < a_month_ago]
            self.dataset = dataset.loc[dataset.index >= a_month_ago]
        else:
            self.dataset = dataset

    def assign_stats(self, stats: dict) -> NoReturn:
        """
        Stores statistics dictionary internally.
        """
        self.dataset_stats = stats

    def __init_algorithms(self,
                          algo_refs: set,
                          register_type: str,
                          load_active_forecasts: bool = False):
        # Initialize the algorithms required for the task:
        for algo in algo_refs:
            if algo == "not_available": # Use last resort models for active power # noqa
                if self.inst_type in ["load", "load_gen"] and register_type == "P": # noqa
                    algo = "naive"
                else:
                    algo = "pv_backup"
            # For each algo, a MvModel instance is initialized
            self.algorithms[algo]: MvModel = self.algorithm_map[algo](
                db_manager=self.db_manager,
                register_type=register_type
            )
            # Change algorithm configs if in unit tests mode
            if self.unit_tests_mode:
                self.algorithms[algo].unit_tests_configs()
            # Dataset & inputs (after feature engineering) are created at once
            # to all forecast horizons
            self.algorithms[algo].assign_dataset(dataset=self.dataset)
            # Load active power forecast (if they exist)
            if load_active_forecasts and not self.P_forecasts.empty:
                self.algorithms[algo].load_active_forecasts(predictions=self.P_forecasts) # noqa
            self.algorithms[algo].feature_engineering()

    def __validate_train_conditions(self):
        assert self.db_manager.mode == "train", "DataManager mode is not train." # noqa

    @simple_fail_proof(ModelTrainException)
    def train_active_power_models(self, save: bool = False) -> NoReturn:
        """
        Main routine to train models for active power. Uses metadata generated
        by DataManager to operate.

        :param save: Indicates whether to save models and other output locally.
        """
        self.__validate_train_conditions()
        _rtype = "P"
        _base_model_info = self.model_ref[_rtype]
        models_to_use = self._adjust_algos_base_models(
            _base_model_info=_base_model_info,
            register_type=_rtype)
        _algos_to_init = set(models_to_use.values())
        # Initialize implied algorithms
        self.__init_algorithms(
            algo_refs=_algos_to_init,
            register_type=_rtype
        )

        successful = []
        not_successful = []
        self.logger.info(f"[Train] - {self.inst_id} - {_rtype} - Start Training") # noqa
        for i, (model_ref, algo_id) in enumerate(models_to_use.items()):
            _model_id = algo_id + "__" + model_ref
            # Initialize model (based on algorithm selection)
            # If not enough historical available, algo listed "not_available"
            # and model is skipped.
            if algo_id == "not_available":
                self.logger.warning(f"[Train] - {self.inst_id} - {_rtype} - {model_ref} - No valid algorithm (probably due to short historical). Skipping.") # noqa
                models_to_use[model_ref] = "not_available"
                not_successful.append(model_ref)
                continue
            self.logger.debug(f"[Train] - {self.inst_id} - {_model_id} - {_rtype}") # noqa
            mv_model: MvModel = self.algorithms[algo_id]
            try:
                model_obj, P_forecasts, scalers, model_inputs = mv_model.train_single_model(  # noqa
                    model_ref=model_ref,
                    return_train_forecasts=True
                )
                self.logger.debug(f"[Train] - {self.inst_id} - {_rtype} - {_model_id} - Train of model successful. model_inputs: " # noqa
                      f"{', '.join(model_inputs)}.") # noqa
            except BaseException as ex:
                # This allows to catch exceptions like for when there is no data available # noqa
                self.logger.error(f"[Train] - {self.inst_id} - {_rtype} - {_model_id} - " # noqa
                      f"{model_ref} - Could not train model: {repr(ex)}.", exc_info=True) # noqa
                models_to_use[model_ref] = "not_available"
                not_successful.append(model_ref)
                continue
            if i == 0:
                # if first i, train & return forecasts on train dataset:
                self.P_forecasts = P_forecasts
            if save:
                try:
                    self.__save_objects(inst_id=self.inst_id,
                                        algorithm=algo_id,
                                        register_type=_rtype,
                                        name=model_ref,
                                        model_obj=model_obj,
                                        model_inputs=model_inputs,
                                        scalers=scalers)
                except BaseException as exc:
                    self.logger.error(
                        f"[Train] - {self.inst_id} - {_rtype} - {_model_id} - "  # noqa
                        f" Could not save model: {repr(exc)}.", exc_info=True) # noqa
                    not_successful.append(model_ref)
                    continue
            successful.append(model_ref)

        successful_models_msg = f"SUCCESFFUL models: {', '.join(successful)}. " if successful else "" # noqa
        not_successful_models_msg = f"UNSUCCESFFUL models: {', '.join(not_successful)}" if not_successful else "" # noqa
        self.logger.info(f"[Train] - {self.inst_id} - {_rtype} - {successful_models_msg}{not_successful_models_msg}") # noqa

        if self._model_location == "local":
            models_list = [m for m, algo in models_to_use.items() if algo != "not_available"] # noqa
            clean_models_dir(inst_id=self.inst_id,
                             register_type="P",
                             models_list=models_list,
                             path=self.models_path)
        return models_to_use

    @simple_fail_proof(ModelTrainException)
    def train_reactive_power_models(self, save=False):
        """
        Main routine to train models for reactive power. Uses metadata
        generated by DataManager to operate.

        :param save: Indicates whether to save models and other output locally.
        """
        self.__validate_train_conditions()
        _rtype = "Q"
        _base_model_info = self.model_ref[_rtype]
        models_to_use = self._adjust_algos_base_models(
            _base_model_info=_base_model_info,
            register_type=_rtype)
        _algos_to_init = set(models_to_use.values())
        # Initialize implied algorithms and load active forecasts in each
        self.__init_algorithms(
            algo_refs=_algos_to_init,
            register_type=_rtype,
            load_active_forecasts=True
        )

        successful = []
        not_successful = []
        self.logger.info(f"[Train] - {self.inst_id} - {_rtype} - Start Training") # noqa
        for i, (model_ref, algo_id) in enumerate(models_to_use.items()):
            # If model needs active forecasts as input
            # and they are not available, skip
            if "forecast" in model_ref and self.P_forecasts.empty:
                self.logger.warning(f"[Train] - {self.inst_id} - {_rtype} - {model_ref} - Model {model_ref} skipped. No active power forecasts available.") # noqa
                models_to_use[model_ref] = "not_available"
                continue
            _model_id = algo_id + "__" + model_ref
            # Initialize model (based on algorithm selection)
            # If not enough historical available, algo listed "not_available"
            # and model is skipped.
            if algo_id == "not_available":
                self.logger.warning(f"[Train] - {self.inst_id} - {_rtype} - {model_ref} - No valid algorithm (probably due to short historical). Skipping.") # noqa
                models_to_use[model_ref] = "not_available"
                not_successful.append(model_ref)
                continue
            mv_model: MvModel = self.algorithms[algo_id]
            # Train single model:
            try:
                model_obj, _, scalers, model_inputs = mv_model.train_single_model(  # noqa
                    model_ref=model_ref,
                    return_train_forecasts=True
                )
                self.logger.debug(f"[Train] - {self.inst_id} - {_rtype} - {_model_id} - Train of model successful. model_inputs: " # noqa
                      f"{', '.join(model_inputs)}.") # noqa
            except BaseException as ex:
                # This allows to catch exceptions like for when there is no data available # noqa
                self.logger.error(f"[Train] - {self.inst_id} - {_rtype} - {_model_id} - " # noqa
                      f"{model_ref} - Could not train model: {repr(ex)}.", exc_info=True) # noqa
                models_to_use[model_ref] = "not_available"
                not_successful.append(model_ref)
                continue
            if save:
                try:
                    self.__save_objects(inst_id=self.inst_id,
                                        algorithm=algo_id,
                                        register_type=_rtype,
                                        name=model_ref,
                                        model_obj=model_obj,
                                        model_inputs=model_inputs,
                                        scalers=scalers)
                except BaseException as exc:
                    self.logger.error(
                        f"[Train] - {self.inst_id} - {_rtype} - {_model_id} - "  # noqa
                        f" Could not save model: {repr(exc)}.",
                        exc_info=True)  # noqa
                    not_successful.append(model_ref)
                    continue
            successful.append(model_ref)

        successful_models_msg = f"SUCCESFFUL models: {', '.join(successful)}. " if successful else "" # noqa
        not_successful_models_msg = f"UNSUCCESFFUL models: {', '.join(not_successful)}" if not_successful else "" # noqa
        self.logger.info(f"[Train] - {self.inst_id} - {_rtype} - {successful_models_msg}{not_successful_models_msg}") # noqa

        if self._model_location == "local":
            models_list = [m for m, algo in models_to_use.items() if algo != "not_available"] # noqa
            clean_models_dir(inst_id=self.inst_id,
                             register_type="Q",
                             models_list=models_list,
                             path=self.models_path)
        return models_to_use

    def __save_objects(self,
                       inst_id,
                       algorithm,
                       register_type,
                       name,
                       model_obj=None,
                       model_inputs=None,
                       scalers=None,
                       ):

        kwargs = dict(
            inst_id=inst_id,
            algorithm=algorithm,
            register_type=register_type,
            name=name
        )
        if self._model_location == "database":
            save_model_method = save_db_model
            save_scalers_method = save_db_scalers
            kwargs.update({'con': self.db_con})
        elif self._model_location == "local":
            save_model_method = save_local_model
            save_scalers_method = save_local_scalers
            kwargs.update({'path': self.models_path})

        save_model_method(obj=model_obj, inputs=model_inputs, **kwargs)
        save_scalers_method(*scalers, **kwargs)

    def __load_objects(self,
                       inst_id,
                       algorithm,
                       register_type,
                       name,
                       objects='models'
                       ):

        kwargs = dict(
            inst_id=inst_id,
            algorithm=algorithm,
            register_type=register_type,
            name=name
        )

        if self._model_location == "database":
            load_model_method = load_db_model
            load_scalers_method = load_db_scalers
            kwargs.update({'con': self.db_con})
        elif self._model_location == "local":
            load_model_method = load_local_model
            load_scalers_method = load_local_scalers
            kwargs.update({'path': self.models_path})

        if objects == 'model':
            model, inputs = load_model_method(**kwargs)
            return model, inputs
        if objects == 'scalers':
            x_scaler, y_scaler = load_scalers_method(**kwargs)
            return x_scaler, y_scaler

    @simple_fail_proof(LoadScalerError)
    def load_forecast_scalers(self, register_type: str) -> NoReturn:
        """
        Loads scalers according to specifications from the DataManager.
        For each horizon 'D', 'D+1', ..., a model based upon available
        data is selected. This method creates attribute self.models
        that is a dictionary with mapping: "horizon" -> "model object"

        :param register_type: Active (P) or reactive (Q)
        """
        _scalers_to_load = []
        _inst_type = self.inst_type
        _models_in_use = self.model_in_use[register_type]
        _cfgs = self.db_manager.models_configs[register_type]
        _scalers_to_load = set((x for x in _cfgs['models'].values()
                                if x not in ["backup_mix", "mix"]))

        # "backup_mix" (load) uses both backup models, with and without weather
        if "backup_mix" in _cfgs['models'].values():
            _scalers_to_load.add("backup_weather")
            _scalers_to_load.add("backup_no_weather")
        # "mix" (res) uses both models, with and without weather
        if "mix" in _cfgs['models'].values():
            _scalers_to_load.add("weather")
            _scalers_to_load.add("no_weather")

        # If loading reactive RES models, add case when active forecasts are available # noqa
        if register_type == "Q" and "weather" in _scalers_to_load:
            _scalers_to_load.add("weather/forecast")
        # If loading reactive load models, add case when active forecasts are available # noqa
        if register_type == "Q" and "D-7" in _scalers_to_load:
            _scalers_to_load.add("D-7/forecast")

        # Add backup models for default model (for "load" installations)
        backup_model = "backup_no_weather" if _inst_type == "load" else "no_weather"  # noqa
        if "not_available" in _models_in_use.values():
            _scalers_to_load.add(backup_model)

        trained_scalers = {}
        _models_in_use = self.model_in_use[register_type]
        for model_name in _scalers_to_load:
            # Check if model is available; if not, load backup model
            if _models_in_use[model_name] == "not_available":
                model_name_ = backup_model
            else:
                model_name_ = model_name
            _algo = _models_in_use[model_name_]
            # Load Model:
            x_scaler, y_scaler = self.__load_objects(
                inst_id=self.inst_id,
                algorithm=_algo,
                register_type=register_type,
                name=model_name_,
                objects='scalers'
            )
            # Create model "name" to "obj" key-value pairs
            trained_scalers[model_name] = (x_scaler, y_scaler)

        del _scalers_to_load
        return trained_scalers

    def load_dataset_stats(self) -> NoReturn:
        """
        Loads statistical measures computed during training phase.
        """
        kwargs = dict(
            inst_id=self.inst_id,
        )

        if self._model_location == "database":
            load_stats_method = load_db_stats
            kwargs.update({'con': self.db_con})
        elif self._model_location == "local":
            load_stats_method = load_local_stats
            kwargs.update({'path': self.models_path})

        stats = load_stats_method(**kwargs)
        return stats

    @simple_fail_proof(ModelLoadException)
    def load_forecast_models(self, register_type):
        """
        Loads models according to specifications from the DataManager.
        For each horizon 'D', 'D+1', ..., a model based upon available
        data is selected. This method creates attribute self.models
        that is a dictionary with mapping:
            "horizon" -> "model object"

        :return: No return value. Models are stored internally.

        """
        _inst_type = self.inst_type
        _models_to_load = []
        _models_in_use = self.model_in_use[register_type]
        _cfgs = self.db_manager.models_configs[register_type]
        _models_to_load = set((x for x in _cfgs['models'].values()
                               if x not in ["backup_mix", "mix"]))

        # "backup_mix" (load) uses both backup models, with and without weather
        if "backup_mix" in _cfgs['models'].values():
            _models_to_load.add("backup_weather")
            _models_to_load.add("backup_no_weather")
        # "mix" (res) uses both models, with and without weather
        if "mix" in _cfgs['models'].values():
            _models_to_load.add("weather")
            _models_to_load.add("no_weather")

        # If loading RES models, add case when active forecasts are available
        if register_type == "Q" and "weather" in _models_to_load:
            _models_to_load.add("weather/forecast")
        # If loading load models, add case when active forecasts are available
        if register_type == "Q" and "D-7" in _models_to_load:
            _models_to_load.add("D-7/forecast")

        # Add backup models for default model (for "load" installations)
        backup_model = "backup_no_weather" if _inst_type == "load" else "no_weather"  # noqa
        if "not_available" in _models_in_use.values():
            _models_to_load.add(backup_model)

        trained_models = {}
        trained_inputs = {}
        _models_in_use = self.model_in_use[register_type]
        # Checking for models that are not availble
        _models_not_available = [m for m in _models_to_load if _models_in_use[m] == "not_available"] # noqa
        if _models_not_available:
            not_available_models_str = ', '.join([m+(' (backup model)' if m == backup_model else '') for m in _models_not_available]) # noqa
            self.logger.warning(f"[Forecast] - {self.inst_id} - {register_type} - Model(s) {not_available_models_str} not available.")  # noqa

        for model_name in _models_to_load:
            # Check if model is available; if not, load backup model
            if _models_in_use[model_name] == "not_available":
                model_name_ = backup_model
            else:
                model_name_ = model_name
            _algo = _models_in_use[model_name_]
            # Load Model:
            model_obj, inputs = self.__load_objects(
                inst_id=self.inst_id,
                algorithm=_algo,
                register_type=register_type,
                name=model_name_,
                objects='model'
            )
            # Create model "name" to "obj" key-value pairs
            trained_models[model_name] = model_obj
            trained_inputs[model_name] = inputs

        del _models_to_load
        return trained_models, trained_inputs

    @staticmethod
    def _check_models_scalers_inputs(model_ref: str,
                                     trained_models: dict,
                                     trained_scalers: dict,
                                     trained_inputs: dict):
        # If model to use is backup_mix, two models and pairs of scalers
        # are required (with or without NWP)
        if model_ref == "backup_mix":
            models = [trained_models["backup_no_weather"],
                      trained_models["backup_weather"]]
            scalers = [trained_scalers["backup_no_weather"],
                       trained_scalers["backup_weather"]]
            inputs = [trained_inputs["backup_no_weather"],
                      trained_inputs["backup_weather"]]
        elif model_ref == "mix":
            models = [trained_models["no_weather"],
                      trained_models["weather"]]

            scalers = [trained_scalers["no_weather"],
                       trained_scalers["weather"]]

            inputs = [trained_inputs["no_weather"],
                      trained_inputs["weather"]]
        else:
            models = [trained_models[model_ref]]
            scalers = [trained_scalers[model_ref]]
            inputs = [trained_inputs[model_ref]]

        return models, scalers, inputs

    @staticmethod
    def _algo_selector_load(days_in_hist: int,
                            algo: str):
        if days_in_hist > 90:
            return algo
        elif days_in_hist > 15:
            return "lqr"
        elif days_in_hist == 0:
            # this will help make some models available
            # (models that don't use all inputs like weather)
            # If still there is no data available, training will expectedly
            # fail afterwards
            return "lqr"
        else:
            return "not_available"

    @staticmethod
    def _algo_selector_res(days_in_hist: int,
                           algo: str):
        if days_in_hist > 30:
            return algo
        else:
            return "not_available"

    def _adjust_algos_base_models(self,
                                  _base_model_info: dict,
                                  register_type: str):
        if self.inst_type in ['solar', 'wind']:
            selector = self._algo_selector_res
        elif self.inst_type in ['load', 'load_gen']:
            selector = self._algo_selector_load
        else:
            raise ValueError(f"Installation type not recognized: {self.inst_type}") # noqa
        _hours_in_hst = self.db_manager.models_configs[register_type]['hours_in_hist'] # noqa
        _days_in_hst = _hours_in_hst // 24
        models_to_use = {
            horizon: selector(days_in_hist=_days_in_hst, algo=algo)
            for horizon, algo in _base_model_info.items()
        }

        return models_to_use

    def __validate_forecast_conditions(self):
        assert self.db_manager.mode == "forecast", "DataManager mode is not forecast." # noqa

    @staticmethod
    def __generate_list_model_refs(rtype_configs):
        _model_refs = set(rtype_configs["models"].values())
        if "backup_mix" in _model_refs:
            _model_refs.remove("backup_mix")
            _model_refs.add("backup_weather")
            _model_refs.add("backup_no_weather")
        if "mix" in _model_refs:
            _model_refs.remove("mix")
            _model_refs.add("weather")
            _model_refs.add("no_weather")

        return _model_refs

    @simple_fail_proof(ModelForecastException)
    def forecast(self,
                 trained_models: dict,
                 trained_scalers: dict,
                 trained_inputs: dict,
                 model_configs: dict,
                 register_type: str) -> pd.DataFrame:
        """
        Main routine to compute forecasts:

            - Uses metadata generated by DataManager to operate
            - Traverses horizon, applying selected method to each 24
            hour subset
            - In case of load installation, applies holiday processing
            when needed
            - Applies some post processing to fix abnormalities in quantiles

        :param trained_models: Dictionary with preloaded models
        :param trained_scalers: Dictionary with preloaded scalers
        :param trained_inputs: Dictionary with list of variables for each model
        :param model_configs: Model configurations dictionary
        :param register_type: Active (P) or Reactive (Q)
        :return: DataFrame with predictions
        """
        self.__validate_forecast_conditions()
        _inst_type = self.inst_type
        _models_in_use = self.model_in_use[register_type]
        _model_refs = self.__generate_list_model_refs(model_configs[register_type]) # noqa
        _algos_to_init = set([_models_in_use[x] for x in _model_refs])
        self.__init_algorithms(
            algo_refs=_algos_to_init,
            register_type=register_type,
            load_active_forecasts=register_type == "Q"
        )
        predictions = pd.DataFrame()
        predictions_units_dict = {
            "P": "kW",
            "Q": "kvar"
        }

        last_resort_model = "naive" if self.inst_type in ["load", "load_gen"] \
                            else "pv_backup"
        for horizon in model_configs[register_type]['horizon']:
            # Get Timestamps in forecast horizon:
            forecast_dates = model_configs[register_type]["horizon"][horizon]
            # Get Average Profile Reference (in case of load forecast):
            if _inst_type == "load":
                avg_profile = model_configs[register_type]["avg_profiles"][horizon] # noqa
            else:
                avg_profile = []
            # Get reference model to use:
            model_ref = model_configs[register_type]["models"][horizon]
            # Check if model was trained
            if _models_in_use.get(model_ref, None) == "not_available":
                # Save model_ref before changing to backup model
                _old_model_ref = model_ref
                # Define backup model ref
                if _inst_type == 'load':
                    model_ref = "backup_no_weather"
                else:
                    model_ref = "no_weather"
                was_model_ref_backup = _old_model_ref == model_ref
                if not was_model_ref_backup and _models_in_use.get(model_ref,
                                                                   None) != "not_available":  # noqa
                    self.logger.debug(
                        f"[Forecast] - {self.inst_id} - {register_type} - "
                        f"horizon {horizon} - "
                        f"model {_old_model_ref} - "
                        f"Model not available, "
                        f"switching to backup ({model_ref}).")
                else:
                    if register_type == "P":
                        self.logger.debug(
                            f"[Forecast] - {self.inst_id} - {register_type} - "
                            f"horizon {horizon} - "
                            f"model {_old_model_ref} - "
                            f"Switching to {last_resort_model} "
                            f"model (last resort).")
            # Check if performing Q forecast with model "weather" and P forecasts are available. Also, if model with activate forecasts can be used. # noqa
            mask = (register_type == "Q") and \
                   (not self.P_forecasts.empty) and \
                   (model_ref in ["weather", "D-7"]) \
                   and (model_ref+'/forecast' in _models_in_use) # noqa
            if mask:
                # Readjust model_ref to include P forecasts
                model_ref += "/forecast"
            # Get reference algorithm to select from init. algorithms:
            if model_ref == "backup_mix":
                _algo_to_use = _models_in_use["backup_no_weather"]
            elif model_ref == "mix":
                _algo_to_use = _models_in_use["no_weather"]
            else:
                _algo_to_use = _models_in_use[model_ref]
            if _algo_to_use == "not_available":
                if register_type == "P":
                    model_ref = last_resort_model
                    _algo_to_use = last_resort_model
                else:
                    continue
            mv_model: MvModel = self.algorithms[_algo_to_use]
            if mask:
                mv_model.load_active_forecasts(predictions=self.P_forecasts)
            # Load models and scalers
            if _algo_to_use == last_resort_model:
                models, scalers, inputs = None, None, None
            else:
                models, scalers, inputs = self._check_models_scalers_inputs(
                    model_ref=model_ref,
                    trained_models=trained_models,
                    trained_scalers=trained_scalers,
                    trained_inputs=trained_inputs)
            # Uses the models provided to generate a forecast for
            # a given horizon:
            day_predictions: pd.DataFrame = mv_model.forecast_single_horizon(
                model_list=models,
                inputs_list=inputs,
                forecast_dates=forecast_dates,
                avg_profile=avg_profile,
                scalers_list=scalers
            )
            day_predictions['algorithm'] = _algo_to_use
            day_predictions['horizon'] = horizon
            day_predictions['model_info'] = model_ref
            day_predictions['register_type'] = register_type
            day_predictions['units'] = predictions_units_dict[register_type]
            predictions = predictions.append(day_predictions)

        if not predictions.empty:
            if self.inst_type == "load":
                predictions = self.forecast_fix_holidays(
                    predictions=predictions,
                    register_type=register_type)

            if self.dataset_stats is not None:
                stats = self.dataset_stats[f"real_{register_type}"]
                predictions = postprocessing_layer(data=predictions,
                                                   max_obs=stats['max'],
                                                   min_obs=stats['min'])

            # When performing P forecast save it in aux. attribute to use
            # later as input for Q forecast
            if register_type == "P":
                self.P_forecasts = predictions

        return predictions

    def forecast_fix_holidays(self,
                              predictions: pd.DataFrame,
                              register_type: str) -> pd.DataFrame:
        """

        Performs holiday processing on precomputed predictions.

        :param predictions: DataFrame with precomputed predictions.
        :param register_type: Active (P) or Reactive (Q)

        :return: DataFrame with holiday processing applied.
        """
        dataset = pd.concat([self._rem_dataset, self.dataset])
        _target = f"real_{register_type}"
        quantiles_re = re.compile('q[0-9]{2}$')
        quantile_cols = [col for col in predictions.columns if
                         quantiles_re.match(col)]
        predictions.loc[:, "is_holy"] = predictions.index.map(
            lambda x: self.db_manager.holyclass.is_holiday(date=x))
        predictions.loc[:, "is_bridge"] = predictions.index.map(
            lambda x: self.db_manager.holyclass.is_bridge(date=x))
        if predictions["is_holy"].sum() != 0:
            holy_dates = predictions.loc[predictions["is_holy"], :].index
            holy_val = (self
                        .db_manager
                        .holyclass
                        .find_holidays_analogous(dates=holy_dates,
                                                 full_historical=dataset,
                                                 col_name='q50',
                                                 target=_target))
            if not holy_val.empty:
                # Determine absolute difference between first forecasted value
                # and hoiday adjusted forecast
                factor = (holy_val['q50'] - predictions['q50']).dropna()
                # Adjust other quantiles using same absolute difference
                q_adjusted_preds = (predictions
                                    .loc[holy_val.index, quantile_cols]
                                    .add(factor.values, axis=0))
                predictions.loc[
                    holy_val.index, quantile_cols] = q_adjusted_preds
                predictions.loc[
                    holy_val.index, "model_info"] += "+holiday"
        if predictions["is_bridge"].sum() != 0:
            holy_dates = predictions.loc[predictions["is_bridge"], :].index
            bridge_val = (self
                          .db_manager
                          .holyclass
                          .find_bridge_analogous(dates=holy_dates,
                                                 full_historical=dataset,
                                                 col_name='q50',
                                                 target=_target))
            if not bridge_val.empty:
                # Determine absolute difference between first forecasted value
                # and hoiday adjusted forecast
                factor = (bridge_val['q50'] - predictions['q50']).dropna()
                # Adjust other quantiles using same absolute difference
                q_adjusted_preds = (predictions
                                    .loc[bridge_val.index, quantile_cols]
                                    .add(factor.values, axis=0))
                predictions.loc[
                    bridge_val.index, quantile_cols] = q_adjusted_preds
                predictions.loc[
                    bridge_val.index, "model_info"] += "+holiday"
        predictions.drop(["is_holy", "is_bridge"], 1, inplace=True)

        return predictions

    def _flag_models_to_train(self):
        _table = self.db_manager.models_info_table
        cql_query = f"UPDATE {_table} " \
                    f"SET last_updated=toTimestamp(now()), " \
                    f"to_train=1 " \
                    f"where id='{self.inst_id}' IF EXISTS;"
        self.db_con.session.execute(cql_query)
        return True

    @simple_fail_proof()
    def upload_models_metadata(self, models_metadata: dict) -> bool:
        """
        Updates database with metadata regarding models.

        :param models_metadata: Dictionary of model references
        :return: Boolean value True, meaning operation was successful
        """
        _table = self.db_manager.models_info_table
        if None in models_metadata.values():
            raise ValueError("None value present in models metadata.")
        for rtype, models in models_metadata.items():
            if not isinstance(models, dict):
                models_metadata[rtype] = dict(models)
        cql_query = f"UPDATE {_table} " \
                    f"SET last_train=toTimestamp(now()), " \
                    f"last_updated=toTimestamp(now()), " \
                    f"model_in_use={models_metadata}," \
                    f"to_train=0 " \
                    f"where id='{self.inst_id}' IF EXISTS;"
        self.db_con.session.execute(cql_query)
        return True

    @simple_fail_proof("Could not process predictions for output")
    def prepare_final_output(self, predictions_dict: dict) -> pd.DataFrame:
        """
        Routine to process dictionary of predictions:

            - Concatenates DataFrames
            - Fixes timestamp columns
            - Guarantees columns are in DataFrame

        :param predictions_dict: Dictionary of DataFrames
        :return: DataFrame with processed predictions
        """
        # If predictions is dict, concat DataFrames
        if isinstance(predictions_dict, dict):
            predictions = pd.concat([preds
                                     for preds in predictions_dict.values()
                                     if preds is not None])

        if predictions.empty:
            raise ValueError("Predictions DataFrame is empty.")
        # Remove NaNs before inserting in DB:
        predictions = predictions.copy().dropna()

        # If point does not have GEN, remove values lower than zero
        # (due to historical errors or forecast errors):
        # if not self.db_manager.is_upac:
        #     mask_lower_than_zero = predictions["forecast"] < 0
        #     mask_lower_than_zero = list(mask_lower_than_zero.values)
        #     predictions.loc[mask_lower_than_zero, "forecast"] = 0
        # Prepare data for db upload:
        predictions_df = predictions.round(2).copy()
        quantiles_re = re.compile('q[0-9]{2}$')
        quantile_cols = [col
                         for col in predictions.columns
                         if quantiles_re.match(col)]
        # Convert back to UTC prior to database insert
        predictions_df = predictions_df.tz_convert("UTC").tz_convert(None)
        predictions_df['request'] = self.db_manager.launch_time_hour_utc.tz_convert(None)  # noqa
        predictions_df.reset_index(inplace=True, drop=False)
        predictions_df.rename(columns={'index': 'datetime'}, inplace=True)
        predictions_df['last_updated'] = pd.to_datetime(pd.datetime.now(), format="%Y-%m-%d %H:%M:%S.000").tz_localize(None)  # noqa
        # Include PT identifier columns:
        predictions_df.loc[:, "id"] = self.inst_id
        # Sort Columns:
        predictions_df = predictions_df[["algorithm", "id",
                                         "register_type", "datetime",
                                         "request",
                                         "horizon",
                                         "last_updated",
                                         "model_info",
                                         "units",
                                         *quantile_cols]]
        return predictions_df

    @simple_fail_proof()
    def insert_in_database(self,
                           conn: CassandraDB,
                           predictions_df: pd.DataFrame) -> bool:
        """
        Stores Forecasts data in database.

        :param conn: Database connection instance
        :param predictions_df: Copy of forecasts stored in database.
        """
        if predictions_df.empty:
            raise Exception("Error! No Forecasts data to insert")
        # Exports predictions to the DB
        conn.insert_query(predictions_df, self.forecasts_table)
        return True

    def save_dataset_stats(self, stats):
        """
        Save statistical measures computed during training phase locally.
        """
        kwargs = dict(
            stats_dict=stats,
            inst_id=self.inst_id,
        )

        if self._model_location == "database":
            save_stats_method = save_db_stats
            kwargs.update({'con': self.db_con})
        elif self._model_location == "local":
            save_stats_method = save_local_stats
            kwargs.update({'path': self.models_path})

        save_stats_method(**kwargs)
