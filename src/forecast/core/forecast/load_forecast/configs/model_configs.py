"""
Models Configs

"""

import os
from core.forecast.helpers.model_helpers import (
    QUANTILES,
    QUANTILES_TESTS
)
__ROOT_DIR__ = os.path.dirname(os.path.abspath(__name__))


class LQRConfig:
    def __init__(self, nwp_variables, target):
        self.quantiles = QUANTILES

        self.predictors = {
            "season": ['hour_sin', 'hour_cos',
                       'week_day_sin', 'week_day_cos',
                       'month_sin', 'month_cos',
                       "business_day"],
            "forecasts": nwp_variables,
            "lags": {target: [('day', [x for x in range(-1, -8, -1)])]}
        }
        self.est_params = dict(
            quantiles=self.quantiles
        )
        self.qreg_params = dict(
            quantiles=self.quantiles,
            vcov='robust',
            kernel='epa',
            bandwidth='hsheather',
            max_iter=1000,
            p_tol=1e-6,
            verbose=0
        )
        self.scaler_params = dict(
            method="StandardScaler",
            # init_kwargs={"feature_range": (0, 1)}
        )

    def activate_unit_tests_configs(self):
        pass


class GBTConfig:
    def __init__(self, nwp_variables, target):
        self.quantiles = QUANTILES
        self.predictors = {
            "season": ["hour", "week_day", "month", "business_day"],
            "forecasts": nwp_variables,
            "lags": {target: [('day', [x for x in range(-1, -8, -1)])]}
        }
        # ---- Forecast Model Parameters ----
        self.est_params = {'learning_rate': 0.015,
                           'min_samples_leaf': 0.01,
                           'n_estimators': 600,
                           'max_depth': 7.2891474691238294,
                           'min_samples_split': 0.015,
                           'max_features': 'sqrt',
                           'random_state': 1,
                           'subsample': 0.8,
                           "verbose": 0,
                           "loss": "quantile",
                           "quantiles": QUANTILES}

    def activate_unit_tests_configs(self):
        self.quantiles = QUANTILES_TESTS
        # ---- Forecast Model Parameters ----
        self.est_params['n_estimators'] = 5
        self.est_params['quantiles'] = QUANTILES_TESTS


class NaiveConfig:
    def __init__(self, nwp_variables, target):
        self.naive_lag_dict = {
            "Monday": 7,
            "Tuesday": 1,
            "Wednesday": 1,
            "Thursday": 1,
            "Friday": 1,
            "Saturday": 7,
            "Sunday": 7
        }
        self.quantiles = QUANTILES
        self.target = target
        self.nwp_variables = nwp_variables

    def activate_unit_tests_configs(self):
        self.quantiles = QUANTILES_TESTS
