from forecast_api.models import QuantileReg

from core.forecast.MvModelClass import MvModel
from core.forecast.load_forecast.configs.model_configs import LQRConfig
from core.forecast.helpers.feature_eng_funcs import create_temp_inputs_df  # noqa

import warnings
warnings.filterwarnings("ignore")


class MvModelLQR(MvModel):
    """
    API for the Linear Quantile Regression algorithm.

    This Class contains the necessary functions to perform a active or
    reactive power probabilistic forecasting task for a
    specific MV Consumer (also referred as installation).


    """
    CONFIG_CLASS = LQRConfig
    MODEL_CLASS = QuantileReg
    SCALE_DATA = True
    MODEL_ID = 'lqr'
    INFER_DST_LAGS = True

    def create_extra_features_phase2(self):

        if (self.register_type == "Q") and ("forecast_P" in self.inputs.columns): # noqa
            new_features, self.inputs = create_temp_inputs_df(
                inputs_df=self.inputs,
                temp_col_id="forecast_P",
                daily_min_max=True,
                roll_ema=True)

            self.extra_features['forecasts'].extend(new_features)
            self.features_to_ignore['forecasts'].append("forecast_P")
