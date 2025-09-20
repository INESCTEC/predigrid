from forecast_api.models import GradientBoostingTrees

from core.forecast.MvModelClass import MvModel
from core.forecast.load_forecast.configs.model_configs import GBTConfig
from core.forecast.helpers.feature_eng_funcs import create_temp_inputs_df  # noqa


class MvModelGBT(MvModel):
    """
    API for the Gradient Boosting Trees Regression algorithm.

    This Class contains the necessary functions to perform a active or
    reactive power probabilistic forecasting task for a
    specific MV Consumer (also referred as installation).


    """
    CONFIG_CLASS = GBTConfig
    MODEL_CLASS = GradientBoostingTrees
    SCALE_DATA = False
    MODEL_ID = 'gbt'
    INFER_DST_LAGS = True

    def create_extra_features_phase2(self):
        """
        Replaces __create_extra_features method from MvModelClass.py

        Note:
            - Names of new features -> add to self.extra_features
            - Names of features to ignore -> add to self.features_to_ignore

        Returns:

        """
        # -- Extra Seasonal Features:
        # --- Day Sections:
        sections = {
            "section_a": [0, 5],
            "section_b": [5, 10],
            "section_c": [10, 15],
            "section_d": [15, 20],
            "section_e": [20, 24]
        }
        for s in sections:
            mask = (self.inputs.hour >= sections[s][0]) & \
                   (self.inputs.hour < sections[s][1])
            self.inputs.loc[:, s] = 0
            self.inputs.loc[mask, s] = 1
        self.extra_features["season"].extend(sections.keys())

        # -- Extra Forecasts Features created from "temp" NWP time-series:
        if "temp" in self.inputs.columns:
            new_cols, self.inputs = create_temp_inputs_df(
                inputs_df=self.inputs,
                temp_col_id="temp",
                daily_min_max=True,
                power_3=True)
            self.extra_features['forecasts'].extend(new_cols)
            self.features_to_ignore['forecasts'].append('temp')
