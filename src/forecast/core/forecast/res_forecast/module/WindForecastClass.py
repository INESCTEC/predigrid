from forecast_api.models import GradientBoostingTrees

from core.forecast.MvModelClass import MvModel
from core.forecast.res_forecast.configs.model_configs import WindConfig
from core.forecast.res_forecast.inputs_creation.temporal_features import (  # noqa
    mod,
    direction
)


class WindForecast(MvModel):
    """
    API for the Gradient Boosting Trees (GBT) algorithm.

    This Class contains the necessary functions to perform a
    Wind probabilistic forecasting task for a specific Wind park/installation.
    The NWP spatial variables were previously computed and should be
    already available inthe database.

    """
    CONFIG_CLASS = WindConfig
    MODEL_CLASS = GradientBoostingTrees
    SCALE_DATA = False
    MODEL_ID = 'gbt'
    INFER_DST_LAGS = False

    def create_extra_features_phase1(self):
        # --- Calculate mod/dir for all different levels:
        # Calculate wind module at diferent heights
        self.dataset['modlev1'] = self.dataset.apply(
            lambda row: mod(row['ulev1'], row['vlev1']), axis=1)
        self.dataset['modlev2'] = self.dataset.apply(
            lambda row: mod(row['ulev2'], row['vlev2']), axis=1)
        self.dataset['modlev3'] = self.dataset.apply(
            lambda row: mod(row['ulev3'], row['vlev3']), axis=1)
        self.dataset['dirlev1'] = self.dataset.apply(
            lambda row: direction(row['ulev1'], row['vlev1']), axis=1)
        self.dataset['dirlev2'] = self.dataset.apply(
            lambda row: direction(row['ulev2'], row['vlev2']), axis=1)
        self.dataset['dirlev3'] = self.dataset.apply(
            lambda row: direction(row['ulev3'], row['vlev3']), axis=1)
        _ignore_cols = ["ulev1", "ulev2", "ulev3", "vlev1", "vlev2", "vlev3"]
        self.features_to_ignore['forecasts'].extend(_ignore_cols)
