import pandas as pd

from src.preprocessed_data.util.outliers_detection import high_low_based_outliers
from src.preprocessed_data.util.outliers_detection import distribuction_based_outliers


def find_outliers(data, res_dist_kwargs={}, res_hl_kwargs={}, trend_dist_kwargs={}):
    """
    Function for large outliers detection
    Searches abnormal values in time series:
        * Trend
        * Normalized/Stationary Time Series representation (referred as residual)

    :param data: (:obj:`pandas.DataFrame`) Raw time-series Trend and Stationary (residual) component
    :param res_dist_kwargs: (:obj:`dict`) Parameters for residual distribution based outliers algorithms.
    :param res_hl_kwargs: (:obj:`dict`) Parameters for residual percentile/extreme based outliers algorithms.
    :param trend_dist_kwargs: (:obj:`dict`) Parameters for trend distribution based outliers algorithms.

    :return: (:obj:`bool`) List of booleans with True for indexes with outliers and False for normal values.
    """
    residual_data = data.residual.copy()
    trend_data = data.trend.copy()
    if (residual_data.quantile(0.95) != 0) and (residual_data.quantile(0.05) != 0):
        res_dist_outliers = distribuction_based_outliers(values=residual_data.values.reshape(-1, 1), **res_dist_kwargs)
        res_hl_outliers = high_low_based_outliers(values=residual_data.values.reshape(-1, 1), **res_hl_kwargs)
        res_is_outlier = [(x or y) for x, y in zip(res_dist_outliers, res_hl_outliers)]
    else:
        res_is_outlier = high_low_based_outliers(values=residual_data.values.reshape(-1, 1), **res_hl_kwargs)

    if (trend_data.quantile(0.95) != 0) and (trend_data.quantile(0.05) != 0):
        trend_is_outlier = distribuction_based_outliers(values=trend_data.values.reshape(-1, 1), **trend_dist_kwargs)
        return [(x or y) for x, y in zip(res_is_outlier, trend_is_outlier)]
    else:
        return res_is_outlier


def remove_nan_periods(df, freq=15, hour_threshold=24.0):
    """
    Removes missing periods with range superior to a given hour threshold.

    Example:
        * With thresholds > 24, missing periods of more than 24 hours (independent of timestamp) will be removed.

    :param df: (:obj:`pandas.DataFrame` or :obj:`pd.Series`) Data to inspect and remove unwanted periods.
    :param freq: (:obj:`int`, default 15) Expected frequency of time-series (in minutes). Used to resample the data and show all the
    missing periods that were not removed.
    :param hour_threshold: (:obj:`float`, default 24.0) Reference maximum number of consecutive hours to keep.

    :return: (:obj:`pandas.DataFrame` or :obj:`pd.Series`) Processed data (with unwanted periods removed)
    """

    assert isinstance(df.index, pd.DatetimeIndex), "Error! Index must be a pandas.DateTimeIndex"

    if isinstance(df, pd.Series):
        series_name = df.name
        dfcopy = df.to_frame().copy()
    else:
        dfcopy = df.copy()

    freq = "{}H".format(freq/60.0) if freq % 60 == 0 else "{}T".format(freq)
    dfcopy = dfcopy.sort_index().asfreq(freq=freq)

    dfcopy["datetime"] = dfcopy.index
    cols_to_analyze = [x for x in dfcopy.columns if x != "datetime"]

    inputdiffs = pd.DataFrame(index=dfcopy.index)
    for inputcol in cols_to_analyze:
        inputdf = dfcopy[['datetime', inputcol]].copy()
        inputdf.dropna(inplace=True)
        inputdf['datetime'] = (inputdf['datetime'] - inputdf['datetime'].shift(1))
        inputdf['datetime'] = inputdf['datetime'].apply(lambda x: x.total_seconds() / 3600)
        inputdf['datetime'] -= float(freq[:-1])/(60 if freq[-1]=='T' else 1)
        inputdiffs[inputcol] = inputdf['datetime']

    inputdiffs = inputdiffs[inputdiffs.max(axis=1) > hour_threshold].max(axis=1)
    if not inputdiffs.empty:
        dates_to_clean = inputdiffs.index.map(lambda x: x - pd.DateOffset(hours=inputdiffs[x]))
        for date_start, date_end in zip(dates_to_clean, inputdiffs.index):
            todel = pd.date_range(date_start, date_end, freq=freq)[1:-1]
            dfcopy.drop(todel, inplace=True)
        dfcopy.drop(['datetime'], axis=1, inplace=True)
    else:
        dfcopy.drop(['datetime'], axis=1, inplace=True)

    if isinstance(df, pd.Series):
        return pd.Series(index=dfcopy.index, data=dfcopy.values.flatten(), name=series_name)
    else:
        return dfcopy
