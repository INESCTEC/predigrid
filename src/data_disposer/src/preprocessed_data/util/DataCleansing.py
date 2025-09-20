
import math
import numpy as np
import pandas as pd

from src.preprocessed_data.util.ts_decompose import astl_decomposition
from src.preprocessed_data.util.aux_functions import (
    remove_nan_periods,
    find_outliers
)


class DataCleansing:
    def __init__(self, freq):
        if not isinstance(freq, int):
            exit("Must specify an expected data frequency as an integer "
                 "(in minutes). E.g. freq=15 for 15 minutes.")
        if freq not in [15, 60]:
            exit("Frequency must be either 15 or 60 minutes.")

        self.freq = freq
        # Adapts smoothing window to the specified frequency
        self.sm_window = 800 if freq == 60 else 4000
        self.raw_data = None
        self.clean_data = None
        self.estimated_indexes = None

    def load_raw_dataset(self, raw_data):
        if not isinstance(raw_data, (pd.Series, pd.DataFrame)):
            exit("Raw data must be a pandas.Series or pandas.DataFrame.")

        if not isinstance(raw_data.index, pd.DatetimeIndex):
            exit("DataFrame Index must be a pd.DateTimeIndex")

        raw_data.sort_index(ascending=True, inplace=True)
        self.raw_data = raw_data.copy()

    def remove_large_missing_periods(self, hour_threshold=24.0):
        # ReSamples and Removes large missing periods
        # (over the specified threshold)
        self.clean_data = remove_nan_periods(df=self.raw_data, freq=self.freq,
                                             hour_threshold=hour_threshold)
        self.estimate_missing()

    def estimate_missing(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        itpl_lim = 3

        # First, interpolates the short missing periods:
        self.clean_data = self.clean_data.interpolate(method='time', limit=itpl_lim)
        missing_idx = self.clean_data.loc[self.clean_data.isnull(), ].index

        if self.estimated_indexes is None:
            self.estimated_indexes = missing_idx
        else:
            self.estimated_indexes = self.estimated_indexes.append(missing_idx)
            self.estimated_indexes = self.estimated_indexes.unique()

        # If any NaNs still exist, uses regression trained on the available data:
        if self.clean_data.isnull().values.any():
            nan_index = self.clean_data[self.clean_data.isnull()].index
            reg_data = pd.DataFrame(index=self.clean_data.index, columns=["value"])
            reg_data.loc[:, 'value'] = self.clean_data
            lag_list = [336, 168, 96, 72, 48, 24]
            for i, lag in enumerate(lag_list):
                lag_idx = reg_data.index - pd.DateOffset(hours=lag)
                lag_id = "lag_{}".format(lag)
                try:
                    reg_data.loc[:, lag_id] = reg_data.loc[lag_idx, "value"].values
                    reg_data.loc[:, lag_id].fillna(reg_data["value"], inplace=True)

                    for j in lag_list[:i]:
                        try:
                            # print(j)
                            prev_lag_id = "lag_{}".format(j)
                            reg_data.loc[:, lag_id].fillna(reg_data[prev_lag_id], inplace=True)
                            if reg_data[lag_id].isnull().values.any():
                                # if, by any chance, there is no way to fill the values with previous lags
                                # uses the week profile for that specific weekday
                                normal_idx = reg_data.index
                                week_ref_idx = reg_data.index.strftime("%w-%H:%M")
                                week_profile = reg_data["value"].groupby(week_ref_idx).mean()
                                reg_data.index = week_ref_idx
                                reg_data.loc[reg_data[lag_id].isnull(), lag_id] = week_profile
                                reg_data.index = normal_idx
                        except Exception as e:
                            pass

                        if reg_data[lag_id].isnull().values.any():
                            # if, by any chance, there is no way to fill the values with previous lags
                            # uses the week profile for that specific weekday
                            normal_idx = reg_data.index
                            week_ref_idx = reg_data.index.strftime("%w-%H:%M")
                            week_profile = reg_data["value"].groupby(week_ref_idx).mean()
                            reg_data.index = week_ref_idx
                            reg_data.loc[reg_data[lag_id].isnull(), lag_id] = week_profile
                            reg_data.index = normal_idx
                except Exception as e:
                    continue

            reg_data.loc[:, "hour_sin"] = np.sin((reg_data.index.hour * 2 * math.pi) / 24)
            reg_data.loc[:, "hour_cos"] = np.cos((reg_data.index.hour * 2 * math.pi) / 24)
            reg_data.loc[:, "wday_sin"] = np.sin((reg_data.index.month * 2 * math.pi) / 7)
            reg_data.loc[:, "wday_cos"] = np.cos((reg_data.index.month * 2 * math.pi) / 7)
            usable_cols = [x for x in reg_data.columns if (x != "value") and (reg_data[x].isnull().sum() == 0)]
            reg_data = reg_data[usable_cols + ["value"]]

            train_features = [x for x in reg_data if "value" not in x]
            train_data = reg_data.loc[~reg_data["value"].isnull(), ]
            test_data = reg_data.loc[reg_data["value"].isnull(), ]
            train_X, train_y = train_data[train_features], train_data["value"]
            test_X = test_data[train_features]

            scaler_x, scaler_y = StandardScaler().fit(X=train_X), StandardScaler().fit(X=train_y.values.reshape(-1, 1))
            train_X, train_y = scaler_x.transform(X=train_X), scaler_y.transform(X=train_y.values.reshape(-1, 1))
            test_X = scaler_x.transform(X=test_X)
            reg = LinearRegression().fit(X=train_X, y=train_y)
            pred = scaler_y.inverse_transform(reg.predict(test_X))
            pred = pd.DataFrame(index=test_data.index, columns=["estimated"], data=pred)
            self.clean_data.loc[nan_index, ] = pred["estimated"]

    def remove_outliers(self):
        # First, remove zero's as it was conditioning some distributions:
        complete_index = self.clean_data.index
        zeros_index = None
        ones_index = None

        if self.clean_data.shape[0] != 0:
            if (sum(self.clean_data == 0) / self.clean_data.shape[0]) > 0.05:  # more than 1% of values are zero - removes
                zeros_index = self.clean_data.loc[self.clean_data == 0, ].index
                self.clean_data = self.clean_data.loc[self.clean_data != 0, ]

        if self.clean_data.shape[0] != 0:
            if (sum(self.clean_data == 1) / self.clean_data.shape[0]) > 0.05:  # more than 1% of values are zero - removes
                ones_index = self.clean_data.loc[self.clean_data == 1,].index
                self.clean_data = self.clean_data.loc[self.clean_data != 1, ]

        if self.clean_data.shape[0] == 0:
            return

        # Apply STL decomposition (we will work with the residual and the trend):
        data_stl = astl_decomposition(
            data=self.clean_data,
            season_strftime="%w",
            smoothing_window=self.sm_window,
            centered_smoothing_window=True
        )

        # data_stl.plot(subplots=True)
        # plt.show()

        # Find Trend / Residual Outliers:
        is_outlier = find_outliers(
            data=data_stl,
            res_dist_kwargs={"max_threshold": 4.5, "min_threshold": 4.5},
            res_hl_kwargs={"max_thresh": 4.5, "min_thresh": 4.5},
            trend_dist_kwargs={"max_threshold": 1.7, "min_threshold": 1.7}  # antes estava a 0.9, mas removia alguns periodos necessarios.
        )

        # Replace Outliers by NaNs
        self.clean_data[is_outlier] = np.nan
        self.clean_data = remove_nan_periods(df=self.clean_data, freq=self.freq, hour_threshold=24.0)

        # Estimate values for outliers:
        self.estimate_missing()

        self.clean_data = self.clean_data.reindex(complete_index)
        if zeros_index is not None:
            self.clean_data.loc[zeros_index, ] = 0
        if ones_index is not None:
            self.clean_data.loc[ones_index, ] = 1












