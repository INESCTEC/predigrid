import os
from core.forecast.helpers.model_helpers import (
    QUANTILES,
    QUANTILES_TESTS
)

__ROOT_DIR__ = os.path.dirname(os.path.abspath(__name__))


class SolarConfig:
    def __init__(self, nwp_variables, **kwargs):
        self.quantiles = QUANTILES
        self.predictors = dict(
            season=['month', 'hour'],
            forecasts=nwp_variables,
            lags={
                'swflx': ('hour', [-1, -2, -3, +1, +2, +3]),
                'cfl': ('hour', [-1, -2, -3, +1, +2, +3]),
            }
        )
        self.est_params = {'learning_rate': 0.025,
                           'min_samples_leaf': 0.01,
                           'n_estimators': 800,
                           'max_depth': 7,
                           'min_samples_split': 0.015,
                           'max_features': 'sqrt',
                           'random_state': 1,
                           'subsample': 0.8,
                           'quantiles': QUANTILES}

    def activate_unit_tests_configs(self):
        self.quantiles = QUANTILES_TESTS
        # ---- Forecast Model Parameters ----
        self.est_params['n_estimators'] = 6
        self.est_params['quantiles'] = QUANTILES_TESTS


class WindConfig:
    def __init__(self, **kwargs):
        self.quantiles = QUANTILES
        self.predictors = dict(
            season=['month', 'hour'],
            forecasts=['modlev1', 'modlev2', 'modlev3',
                       'dirlev1', 'dirlev2', 'dirlev3'],
            lags={
                'modlev1': ('hour', [-1, -2, -3, +1, +2, +3]),
                'modlev2': ('hour', [-1, -2, -3, +1, +2, +3]),
                'modlev3': ('hour', [-1, -2, -3, +1, +2, +3]),
            }
        )
        self.est_params = {'min_samples_leaf': 0.01,
                           'n_estimators': 800,
                           'learning_rate': 0.025,
                           'min_samples_split': 0.01,
                           'max_depth': 8,
                           'max_features': 'sqrt',
                           'random_state': 1,
                           'subsample': 0.8,
                           'quantiles': QUANTILES}

    def activate_unit_tests_configs(self):
        self.quantiles = QUANTILES_TESTS
        # ---- Forecast Model Parameters ----
        self.est_params['n_estimators'] = 6
        self.est_params['quantiles'] = QUANTILES_TESTS


class PVBackupConfig:
    def __init__(self, nwp_variables, target):
        self.predictors = {
            "season": ['month', 'hour_sin', 'hour_cos'],
            "forecasts": ["swflx", "cfl"],
            "lags": None
        }
        self.quantiles = QUANTILES
        self.target = target
        self.nwp_variables = nwp_variables

    def activate_unit_tests_configs(self):
        self.quantiles = QUANTILES_TESTS
