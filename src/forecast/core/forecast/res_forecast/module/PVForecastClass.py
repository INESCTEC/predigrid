from forecast_api.models import GradientBoostingTrees

from core.forecast.MvModelClass import MvModel
from core.forecast.res_forecast.configs.model_configs import SolarConfig


class PVForecast(MvModel):
    """
    API for the Gradient Boosting Trees (GBT) algorithm.

    This Class contains the necessary functions to perform a
    Solar probabilistic forecasting task for a specific PV installation.
    The NWP spatial variables were previously computed and should be
    already available in the database.

    """
    CONFIG_CLASS = SolarConfig
    MODEL_CLASS = GradientBoostingTrees
    SCALE_DATA = False
    MODEL_ID = 'gbt'
    INFER_DST_LAGS = False

    def create_extra_features_phase2(self):
        pass
