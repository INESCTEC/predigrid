import cassandra
import numpy as np
import pandas as pd

from abc import ABC
from typing import TypeVar, List, NoReturn
from cassandra.connection import ConnectionException
from statsmodels.stats.outliers_influence import variance_inflation_factor

from forecast_api import DataClass

from core.forecast.custom_exceptions import (
    TrainEmptyDatasetError
)
from core.database.DataManager import DataManager

t_config_class = TypeVar('t_config_class')
t_model_class = TypeVar('t_model_class')
t_scaler_class = TypeVar('t_scaler_class')


class MvModel(ABC, DataClass):
    """
    Template class for defining Forecasting Model Subclasses.
    This Class contains configuration attributes and appropriate methods
    for metadata loading, feature engineering, model training
    and forecast computation that can either be used as is or
    can be overwritten to perform task in a more
    particular fashion (if the model requires so).

    The most relevant methods of this class are:

        * :meth:`~feature_engineering`: Extract extra temporal features from
        the raw NWP data.
        * :meth:`~train_single_model`: Train and save a forecasting model.
        * :meth:`~forecast_single_horizon`: Predict values for a subset of
        the horizon to predict (block forecast).


    """
    DATABASE_EXCEPTIONS = (cassandra.ReadFailure, cassandra.ReadTimeout,
                           cassandra.OperationTimedOut, cassandra.Timeout,
                           cassandra.FunctionFailure, cassandra.Timeout,
                           cassandra.WriteFailure, cassandra.WriteTimeout,
                           ConnectionException)
    SPLIT_SERIAL_SIZE = 100

    CONFIG_CLASS: t_config_class
    MODEL_CLASS: t_model_class
    MODEL_ID: str
    SCALE_DATA: bool
    INFER_DST_LAGS: bool

    def __repr__(self):
        return f"MvModel_{self.model_id.upper()}"

    def __init__(self,
                 db_manager: DataManager,
                 register_type: str,
                 verbose: bool = False):
        """
        Initialization parameters that characterize each forecast run.
        To be used as class attributes.

        :param DataManager db_manager: Data manager object with
        forecast information.
        :param str register_type: Type of power (Active - "P";
        Reactive - "Q") to perform forecast on.
        :param bool verbose: whether to print intermediate information or not.
        """

        # --- forecast_api.DataClass --
        super(MvModel, self).__init__(timezone=db_manager.forecast_tz)

        self.__load_forecast_metadata(model_id=self.MODEL_ID,
                                      register_type=register_type,
                                      db_manager=db_manager)
        self.__init_model_configs()
        self.__process_launch_time()

        # Initialize forecast attributes
        self.verbose = verbose
        self.extra_features = {"forecasts": [], "season": [], "lags": []}
        self.features_to_ignore = {"forecasts": [], "season": [], "lags": []}

    def __load_forecast_metadata(self,
                                 model_id: str,
                                 register_type: str,
                                 db_manager: DataManager,
                                 ):

        self.model_id = model_id
        self.db_manager = db_manager
        self.register_type = register_type
        self.cass_engine = self.db_manager.engine
        self.inst_id = self.db_manager.inst_id
        self.inst_type = self.db_manager.inst_type
        self.target = f"real_{self.register_type}"
        self.forecast_range = self.db_manager.forecast_range
        self.inst_capacity = self.db_manager.inst_capacity

    def __process_launch_time(self):
        self.detail_launch_time = self.db_manager.launch_time_hour_utc
        self.launch_time = self.detail_launch_time.strftime(
            "%Y-%m-%d %H:00:00")
        self.launch_time = pd.to_datetime(
            self.launch_time,
            format="%Y-%m-%d %H:%M:%S").tz_localize("UTC")
        self.launch_time_tz = self.launch_time.tz_convert(tz=self.tz)

    def __init_model_configs(self):
        self.config = self.CONFIG_CLASS(
            nwp_variables=self.db_manager.nwp_vars,
            target=self.target,
        )

    def unit_tests_configs(self) -> NoReturn:
        """
        Changes model parameters configurations for unit testing

        """
        self.config.activate_unit_tests_configs()

    def assign_dataset(self, dataset: pd.DataFrame) -> NoReturn:
        """
        Stores dataset internally.

        :param dataset: DataFrame to store
        """
        # Load dataset to DataClass container:
        self.load_dataset(dataset)

    def create_extra_features_phase1(self) -> NoReturn:
        """
        Apply ad hoc processing before main feature engineering method
        """
        pass

    def create_extra_features_phase2(self) -> NoReturn:
        """
        Apply ad hoc processing after main feature engineering method
        """
        pass

    def feature_engineering(self):
        """
        Method for Feature Engineering

             * Creates chronological/seasonal inputs
             * Creates variables to model temporal information (lags, ...)

        :return: None. Initializes a ModelGBT.inputs attribute.
        """
        # Calculate extra features ON the original Dataset:
        self.create_extra_features_phase1()
        # Create Inputs DataFrame:
        self.construct_inputs(
            season=self.config.predictors["season"],
            lags=self.config.predictors["lags"],
            forecasts=self.config.predictors["forecasts"],
            infer_dst_lags=self.INFER_DST_LAGS
        )
        # Add active forecasts to inputs
        # if they have been loaded to the dataset
        is_q = self.register_type == "Q"
        forec_in_dataset = "forecast_P" in self.dataset.columns
        forec_not_in_inputs = "forecast_P" not in self.inputs.columns
        if is_q and forec_in_dataset and forec_not_in_inputs:
            self.inputs = self.inputs.join(self.dataset['forecast_P'])
        # Create extra features on the new inputs dataset:
        self.create_extra_features_phase2()

        # Interpolate values (maximum two hours of consecutive missing values)
        self.inputs.interpolate(
            method="linear",
            limit=2,
            limit_direction="backward",
            inplace=True
        )

    def __remove_ignored_model_inputs(self, model_inputs):
        # Remove features that are to be ignored
        for input_type in ['season', 'forecasts', 'lags']:
            [model_inputs.remove(feat)
             for feat in self.features_to_ignore[input_type]
             if feat in model_inputs]
        return model_inputs

    def __generate_inputs_list_res(self, model_ref: str = None):
        """
        Method for generating a list of inputs (exogenous variables) for RES
        generation models.

        :param str model_ref: A string reference for the target model.
        :return: A sorted list of inputs for the model in question.
        """
        if model_ref == "no_weather":
            # If "no_weather", only use seasonal inputs (original and engineered) # noqa
            model_inputs = [*self.config.predictors['season'],
                            *self.extra_features["season"]]
        else:
            # Base models
            # List seasonal inputs, forecast inputs and extra, engineered lags
            model_inputs = [*self.config.predictors['season'],
                            *self.config.predictors["forecasts"],
                            *self.extra_features["season"],
                            *self.extra_features["forecasts"],
                            *self.extra_features["lags"]]
            # Check if active forecasts are listed in model reference
            if "forecast" in model_ref.split('/'):
                model_inputs.append('forecast_P')

            # Add lag columns:
            for variable, items in self.config.predictors["lags"].items():
                lag_type = items[0]
                for lag in items[1]:
                    model_inputs.append(f"{variable}_{lag}_{lag_type}")

        # Remove ignored inputs (e.g. not needed after feature engineering)
        model_inputs = self.__remove_ignored_model_inputs(model_inputs)
        return sorted(model_inputs)

    def __generate_inputs_list_load(self, model_ref: str = None):
        """
        Method for generating a list of inputs (exogenous variables) for load
        models.

        :param str model_ref: A string reference for the target model.
        :return: A sorted list of inputs for the model in question.
        """
        if model_ref == "backup_no_weather":
            # If "backup_no_weather", only use seasonal inputs (original and engineered) # noqa
            model_inputs = [*self.config.predictors['season'],
                            *self.extra_features['season']]
        elif model_ref == "backup_weather":
            # If "backup_weather", only seasonal inputs and listed NWP vars (original and engineered) # noqa
            # Add seasonal features and engineered seasonal features (original and engineered) # noqa
            model_inputs = [*self.config.predictors['season'],
                            *self.extra_features['season'],
                            *self.db_manager.nwp_vars]
        elif model_ref == "D-7/no_weather":
            model_inputs = [*self.config.predictors['season'],
                            *self.extra_features['season'],
                            f"{self.target}_-7_day"]
        else:
            # Base models
            # List seasonal inputs, forecast inputs and extra, engineered lags
            model_inputs = [*self.config.predictors['season'],
                            *self.config.predictors['forecasts'],
                            *self.extra_features['season'],
                            *self.extra_features['forecasts'],
                            *self.extra_features['lags']]

            # If no reference given, use "D-7"
            if model_ref is None:
                model_ref = "D-7"
            # List all lags in model reference
            list_model_lags = [x for x in model_ref.split('/')
                               if x != 'forecast']
            lag_cols = [f"{self.target}_{d[1:]}_day" for d in list_model_lags]
            # Check if active forecasts are listed in model reference
            if "forecast" in model_ref.split('/'):
                model_inputs.append('forecast_P')
            model_inputs.extend(lag_cols)

        # Remove ignored inputs (e.g. not needed after feature engineering)
        model_inputs = self.__remove_ignored_model_inputs(model_inputs)
        return sorted(model_inputs)

    def __scale_train_data(self, X, y):
        X = X.copy()
        y = y.copy()

        # -- Scale labels data:
        y_scaled, y_scaler = self.normalize_data(
            y.dropna(),
            **self.config.scaler_params)

        # --  Explanatory variables not to scale (seasonal dont need scaling)
        seasonal_non_scalable = [*self.config.predictors["season"],
                                 *self.extra_features["season"]]
        seasonal_non_scalable = [x for x in seasonal_non_scalable
                                 if x in X.columns]

        # -- Only seasonal explanatory variables available? no need to scale:
        if sorted(X.columns) != sorted(seasonal_non_scalable):
            X_nonscalable_cols = [x for x in X.columns
                                  if x in seasonal_non_scalable]
            X_non_scalable = X[[x for x in X.columns
                                if x in seasonal_non_scalable]]
            X.drop(X_nonscalable_cols, 1, inplace=True)

            # -- Scale inputs data:
            x_scaled, x_scaler = self.normalize_data(
                X.dropna(),
                **self.config.scaler_params)

            for c in X_nonscalable_cols:
                x_scaled.loc[:, c] = X_non_scalable.loc[:, c]

            return (x_scaled, y_scaled), (x_scaler, y_scaler)

        else:
            return (X, y_scaled), (None, y_scaler)

    def __scale_operational_data(self, X, x_scaler):
        X = X.copy()

        # -- Find explanatory variables not to scale (seasonal dont need scaling) # noqa
        seasonal_non_scalable = [*self.config.predictors["season"],
                                 *self.extra_features["season"]]
        seasonal_non_scalable = [x for x in seasonal_non_scalable
                                 if x in X.columns]

        # -- If only seasonal explanatory variables are available, no need to scale: # noqa
        if sorted(X.columns) != sorted(seasonal_non_scalable):
            X_nonscalable_cols = [x for x in X.columns
                                  if x in seasonal_non_scalable]
            X_non_scalable = X[[x for x in X.columns
                                if x in seasonal_non_scalable]]
            X.drop(X_nonscalable_cols, 1, inplace=True)

            # Scale datasets
            null_bool_dataframe = X.isnull()  # Dataframe with Boolens ( NaN-True / Value-False) # noqa
            if not null_bool_dataframe.values.any():
                # if there are no NaNs, performs a normal normalization
                X_scaled, _ = self.normalize_data(data=X, method=x_scaler)
            else:
                # If there are NaNs, fills NaNs with 0, normalizes, then removes the zeros # noqa
                aux_x = X.fillna(0)
                X_scaled, _ = self.normalize_data(data=aux_x, method=x_scaler)
                X_scaled = X_scaled[~null_bool_dataframe]

            for c in X_nonscalable_cols:
                X_scaled.loc[:, c] = X_non_scalable.loc[:, c]

            return X_scaled
        else:
            return X

    @staticmethod
    def __compute_forecasts_for_train(model, x_train, y_scaler=None):
        # Inverse scaling for final predicted values
        predictions = model.forecast(x_train)
        if y_scaler is not None:
            return pd.DataFrame(y_scaler.inverse_transform(predictions),
                                columns=predictions.columns,
                                index=predictions.index)
        else:
            return predictions

    @staticmethod
    def __check_model_inputs(inputs):
        """
        Method that validates variables by checking if they are not constant.
        """
        # Check if any variables are constant
        valid_inputs_mask = ~(inputs.min() == inputs.max())
        valid_inputs = inputs.loc[:, valid_inputs_mask]

        # Check if matrix_rank is lower than number of inputs + 1 (constant term column) # noqa
        add_const_col = lambda x: np.column_stack((np.ones(x.shape[0]), x))
        while np.linalg.matrix_rank(add_const_col(valid_inputs)) < len(valid_inputs.columns) + 1: # noqa
            # Check multicollinearity between variables
            vif_filter_func = lambda df: np.array([variance_inflation_factor(df.values, i) for i in range(df.shape[1])]) # noqa
            vif_filter = vif_filter_func(valid_inputs)
            vif_arg_max = vif_filter.argmax()
            if vif_filter[vif_arg_max] >= 5:
                valid_inputs = valid_inputs.drop(valid_inputs.columns[vif_arg_max], axis=1) # noqa
            else:
                break
            # valid_inputs = valid_inputs.loc[:, vif_filter_mask]

        return sorted(valid_inputs.columns)

    def train_single_model(self,
                           model_ref: str = None,
                           return_train_forecasts: bool = False) -> (t_config_class, pd.DataFrame, (t_scaler_class, t_scaler_class), list): # noqa
        """
        Routine for training a single model:

            * Generates inputs list
            * Applies data scaling if needed
            * Fits data to underlying model
            * Performs forecasting if requested

        :param model_ref: Reference name for model
        :param return_train_forecasts: Whether to perform forecasting or not.
        Useful for providing active power forecasts to use
        in reactive power models
        :return: - Model class
                 - DataFrame with predictions
                 - Tuple with scalers (for variables and labels)
                 - List of variables in model
        """
        # Define model input list, depending on model_ref & inst type:
        if self.inst_type in ["solar", "wind"]:
            model_inputs = self.__generate_inputs_list_res(model_ref)
        else:
            model_inputs = self.__generate_inputs_list_load(model_ref)

        # Separation of inputs and target variable
        x_train, y_train = self.split_dataset(
            target=self.target,
            dropna=True,
            inputs=self.inputs.loc[:, model_inputs]
        )

        if x_train.empty:
            raise TrainEmptyDatasetError(
                "Explanatory variables dataset is empty.")
        if y_train.empty:
            raise TrainEmptyDatasetError(
                "Target variable dataset is empty.")

        # Scale data before training:
        x_scaler = None
        y_scaler = None
        x_train_original = x_train.copy()
        y_train_original = y_train.copy()
        if self.SCALE_DATA:
            (x_train, y_train), (x_scaler, y_scaler) = self.__scale_train_data(
                X=x_train,
                y=y_train
            )

        # Define and train model:
        model = self.MODEL_CLASS(**self.config.est_params)
        try:
            model.fit_model(x_train, y_train)
        except ValueError as exc:
            if repr(self) in ["MvModel_LQR"]:  # revert to == "MvModel_LQR"
                # Check inputs for constant variables
                model_inputs = self.__check_model_inputs(x_train)
                x_train_original = x_train_original.loc[:, model_inputs]

                if self.SCALE_DATA:
                    (x_train, y_train), (
                        x_scaler, y_scaler) = self.__scale_train_data(
                        X=x_train_original,
                        y=y_train_original
                    )
                else:
                    x_train = x_train_original

                model.fit_model(x_train, y_train)
            else:
                raise exc

        train_preds = pd.DataFrame()
        if return_train_forecasts:
            # if asked to, returns forecasts for train dataset
            # (useful to predict Q later on)
            train_preds = self.__compute_forecasts_for_train(
                model=model,
                x_train=x_train,
                y_scaler=y_scaler
            )
        return model, train_preds, (x_scaler, y_scaler), model_inputs

    def __fill_with_weekday_profile(self,
                                    forecast_dates: pd.DatetimeIndex,
                                    avg_profile: List[float]):
        # -- Fill lag -7 with profile if it exists:
        _lag_col = f"{self.target}_-7_day"
        _required = self.inputs[_lag_col][forecast_dates].isnull().any()
        _profile_exists = len(avg_profile) == len(forecast_dates)
        new_inputs = self.inputs.copy()
        if _required and _profile_exists:
            new_inputs.loc[forecast_dates, _lag_col] = avg_profile
        return new_inputs

    def forecast_single_horizon(self,
                                model_list: list,
                                inputs_list: list,
                                forecast_dates: pd.DatetimeIndex,
                                avg_profile: list,
                                scalers_list: list,
                                ) -> pd.DataFrame:
        """
        Computes forecast for a specific horizon, identified by a string
        like "D" or "D+1".

        :param model_list: List of available models
        :param inputs_list: List of available inputs per model
        :param forecast_dates: Datetime Index of dates to be forecasted
        :param avg_profile: Average profile to use as input in case of missing
        data
        :param scalers_list: List of available scalers per model
        :return: DataFrame with predictions
        """
        # Fill NaN (in case of load forecasting)
        if self.inst_type == "load":
            inputs = self.__fill_with_weekday_profile(
                forecast_dates=forecast_dates,
                avg_profile=avg_profile
            )
        else:
            inputs = self.inputs.copy()
        # If multiple models are specified (e.g. backup_mix)
        # Then, multiple overlapping performances are created
        predictions = pd.DataFrame()
        for model, scalers, model_inputs in zip(model_list, scalers_list, inputs_list): # noqa
            # Separation of inputs and target variable
            x_operational, _ = self.split_dataset(
                target=self.target,
                dropna=False,
                inputs=inputs.loc[:, model_inputs],
                period=forecast_dates
            )
            # Interpolate few successive missing values (limit 2)
            x_operational.interpolate(
                method="linear",
                limit=2,
                limit_direction="both",
                inplace=True
            )
            # Scale data if needed
            x_scaler, y_scaler = scalers
            if self.SCALE_DATA and (x_scaler is not None):
                x_operational = self.__scale_operational_data(
                    X=x_operational,
                    x_scaler=x_scaler
                )
            # Generate forecasts:
            predictions_ = model.forecast(x_operational)
            # Inverse scaling for final predicted values
            if self.SCALE_DATA and (y_scaler is not None):
                predictions_ = pd.DataFrame(
                    data=y_scaler.inverse_transform(predictions_),
                    columns=predictions_.columns,
                    index=predictions_.index)
            # Final structure:
            if predictions.empty:
                # If DataFrame is empty, create new DataFrame.
                predictions = predictions_.copy()
            else:
                # Else just update predictions
                predictions.update(predictions_)

        return predictions

    def load_active_forecasts(self, predictions: pd.DataFrame, use_col="q50") -> NoReturn: # noqa
        """
        Loads active power forecasts to use as input
        for reactive power forecasts.

        :param predictions: DataFrame with predictions
        :param use_col: Name of column to get predictions from
        """
        forecasts_col = 'forecast_P'
        predictions = predictions.copy()
        if forecasts_col not in self.dataset.columns:
            predictions.rename(columns={use_col: forecasts_col}, inplace=True)
            self.dataset = self.dataset.join(predictions[forecasts_col].astype(float)) # noqa
