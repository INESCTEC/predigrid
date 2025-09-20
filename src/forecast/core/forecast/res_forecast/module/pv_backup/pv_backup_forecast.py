import os
import pickle
import pandas as pd

from core.forecast.MvModelClass import MvModel
from core.forecast.res_forecast.configs.model_configs import PVBackupConfig # noqa


class PVBackupForecast(MvModel):
    """
    API for a PV backup model.

    This Class contains the necessary functions to perform an active or
    reactive power probabilistic forecasting task for a
    specific MV Consumer (also referred as installation).


    """
    CONFIG_CLASS = PVBackupConfig
    MODEL_CLASS = None
    SCALE_DATA = False
    MODEL_ID = 'pv_backup'
    INFER_DST_LAGS = True

    loaded_models = {
        "model_1": None,
        "model_2": None,
        "model_3": None
    }

    def forecast_single_horizon(self,
                                model_list: list,
                                inputs_list: list,
                                forecast_dates: pd.DatetimeIndex,
                                avg_profile: list,
                                scalers_list: list,
                                ) -> pd.DataFrame:

        """

        **Operational Forecast Call WITHOUT or SHORT historical data**

        Method designed to simplify the response interface to a forecast
        request without or short historical data.
        Processes and prepare the required pre-forecast conditions and
        launches the forecast algorithms in order to
        provide energy generation forecasts for a given time horizon, with a
        specific temporal resolution.

        Extra:
            * Records the computational time of the forecasting process.
            * Reports the quality of the forecast (ModelClass.f_quality)

        The following quality indexes should be expected:
        * **Quality index "problem"** - Some problem happened during the
            forecast process and the predicted values container is empty.
        * **Quality index "no_pv_data"** - Forecast computed without historical
            PV data (transfer learning models used)
        * **Quality index "short_historical"** - Forecast computed with short
            historical PV data (transfer learning models used)

        Args:
            * horizon: (:obj:`int`) **Defined in system/settings.ini file.**
                Forecast horizon in hours ahead of launchtime.
            * quantiles: (:obj:`list`) List of quantiles to forecast.
            * res_temp_min: (:obj:`int`) **Defined in system/settings.ini file.
                ** Resolution of the forecast output,
                in minutes. Different options can be chosen, if resolution is
                lower than 60 minutes, interpolation is
                used as filling method, else the average is used.
            * inst_capacity: (:obj:`list`) List of forecast quantiles.

        Returns: (:obj:`pandas.DataFrame`) Forecast data.

        """

        # -- Forecast inputs
        x_operational, _ = self.split_dataset(period=forecast_dates,
                                              dropna=False)

        available_inputs = [k for k, v in
                            x_operational.isnull().any().items() if v is False]

        model_basepath = os.path.dirname(os.path.abspath(__file__))
        model_dict = {
            "model_1": {
                "features": ['cfl', 'hour_cos', 'hour_sin', 'month', 'swflx'],
                "path": os.path.join(model_basepath, "resources", f"{self.register_type}_bkp_model_with_weather.pkl") # noqa
            },
            "model_2": {
                "features": ['hour_cos', 'hour_sin', 'month', 'swflx'],
                "path": os.path.join(model_basepath, "resources", f"{self.register_type}_bkp_model_with_swflx.pkl") # noqa
            },
            "model_3": {
                "features": ["hour_sin", "hour_cos", "month"],
                "path": os.path.join(model_basepath, "resources", f"{self.register_type}_bkp_model_no_weather.pkl") # noqa
            },
        }

        if ("swflx" in available_inputs) and ("cfl" in available_inputs):
            model_id = "model_1"
        elif "swflx" in available_inputs:
            model_id = "model_2"
        else:
            model_id = "model_3"

        # -- Prepare model and features:
        model_path = model_dict[model_id]["path"]
        x_operational = x_operational[model_dict[model_id]["features"]]

        if self.loaded_models[model_id] is None:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
                self.loaded_models[model_id] = model
        else:
            model = self.loaded_models[model_id]

        predictions = model.forecast(x=x_operational)
        predictions.index.rename("datetime", inplace=True)

        # ----------- Corrections in Final Forecasts -------------
        # Remove negative values in PV Forecasts:
        if self.register_type == 'P':
            for col in predictions:
                predictions.loc[predictions[col] < 0, col] = 0

        qt_filter = ['q' + str(int(col * 100)).zfill(2) for col in
                     self.config.quantiles]
        predictions = predictions[qt_filter] * self.inst_capacity

        del x_operational

        return predictions
