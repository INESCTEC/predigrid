import pandas as pd

from core.forecast.MvModelClass import MvModel
from core.forecast.load_forecast.configs.model_configs import NaiveConfig
from core.forecast.custom_exceptions import ModelForecastException


class MvModelNaive(MvModel):
    """
    API for the Naive model.

    This Class contains the necessary functions to perform an active or
    reactive power deterministic forecasting task for a
    specific MV Consumer (also referred as installation).


    """
    CONFIG_CLASS = NaiveConfig
    MODEL_CLASS = None
    SCALE_DATA = False
    MODEL_ID = 'naive'
    INFER_DST_LAGS = True

    def feature_engineering(self):
        pass

    @staticmethod
    def __date_reference_replace(ref_date, target_dates):
        refyear = ref_date.year
        refmonth = ref_date.month
        refday = ref_date.day
        ref_dates = [d.replace(year=refyear, month=refmonth, day=refday)
                     for d in target_dates]
        return ref_dates

    def _get_date_reference(self, dates_dict, dataset, lags=None):
        """
        Routine to check if desired target label lags for naive model
        forecast have available values. If not, tries to get values from
        previous week (-7 days).
        Optionally, the list of lags can be provided, overriding the
        default behaviour.
        """
        final_ref_dates = list()
        for norm_date, dates in dates_dict.items():
            lag_offset = self.config.naive_lag_dict[norm_date.day_name()]
            possible_lags = [lag_offset, 7] if lags == None else lags # noqa
            for lag in possible_lags:
                ref_norm_date = norm_date - pd.DateOffset(**{"days": lag})
                ref_dates = self.__date_reference_replace(ref_norm_date,
                                                          dates)
                if dataset.reindex(ref_dates).notnull().all():
                    final_ref_dates.extend(ref_dates)
                    if lags is not None:
                        lags = lags[lags.index(lag):]
                    break
            else:
                final_ref_dates.extend(ref_dates)

        return final_ref_dates

    def forecast_single_horizon(self,
                                model_list: list,
                                inputs_list: list,
                                forecast_dates: pd.DatetimeIndex,
                                avg_profile: list,
                                scalers_list: list,
                                ) -> pd.DataFrame:
        last_week_start = self.launch_time_tz - pd.DateOffset(**{"days": 7})
        last_week_end = self.launch_time_tz
        last_weeks_dates = pd.date_range(last_week_start,
                                         last_week_end,
                                         freq="H",
                                         tz=self.dataset.index.tz,
                                         closed='left')
        dataset = self.dataset.reindex(last_weeks_dates)[self.target]
        date_refs = self._get_date_reference(dates_dict=forecast_dates.groupby(forecast_dates.date), # noqa
                                             dataset=dataset)
        if dataset.reindex(date_refs).isnull().any():
            lags = range(1, 7)
            dates_dict = forecast_dates.groupby(forecast_dates.date)
            date_refs = self._get_date_reference(dates_dict=dates_dict,
                                                 dataset=dataset,
                                                 lags=lags)
            if dataset.reindex(date_refs).isnull().any():
                raise ModelForecastException("No available data for naive forecast.") # noqa

        predictions = pd.DataFrame(index=forecast_dates,
                                   columns=["q50"],
                                   data=dataset.reindex(date_refs).values)

        return predictions
