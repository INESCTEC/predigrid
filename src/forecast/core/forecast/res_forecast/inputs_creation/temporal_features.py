# flake8: noqa

import pandas as pd
import numpy as np

from math import atan2, sqrt, atan


def forecast_runs(full_data, variable_id):

    # Sort dataset and get split more recent data to other obj.
    full_data.sort_values(by=['datetime', 'request'], ascending=True, inplace=True)
    recent_data = full_data[['datetime', variable_id]].drop_duplicates('datetime', keep='last')

    # Calculate time differences between timestamp and timerequest so we distinguish the forecasts:
    full_data['time_dif'] = (full_data['datetime'] - full_data['request']).astype('timedelta64[h]')

    full_data.set_index('datetime', inplace=True)
    recent_data.set_index('datetime', inplace=True)

    prev_data = pd.DataFrame(index=recent_data.index)

    for i in range(0, 4):
        df = full_data[(full_data['time_dif'] >= 24*i) & (full_data['time_dif'] < 24*(i+1))][variable_id]
        df = df[~df.index.duplicated(keep='last')]
        df = df.reindex(recent_data.index).fillna(recent_data[variable_id])  # Reindex and fills gaps with recent data
        df.name = variable_id + '_{}d'.format(i)
        prev_data = prev_data.join(df)

    prev_data = prev_data.resample('H').mean()
    prev_data.loc[:, '{}_w_ld'.format(variable_id)] = np.average(prev_data.values, weights=[80, 60, 45, 30], axis=1)

    return prev_data['{}_w_ld'.format(variable_id)]


def temporal_indexes(full_data, variable_id, window_list):
    # Work with most recent data for each timestamp
    recent_data = full_data.drop_duplicates('datetime', keep='last')
    recent_data.set_index('datetime', inplace=True)

    # For each timestamp, get a dataframe with most recent available data:
    recent_df = recent_data[variable_id]

    temporal_var = pd.DataFrame()
    for window in window_list:
        t_variable_name = "{}_var_t_{}h".format(variable_id, window)
        temporal_var[t_variable_name] = recent_df.rolling(window=window, min_periods=1, center=True).var()

    return temporal_var


def create_past_runs_weighted_inputs(data, variables):
    data = data.copy()
    data.index.rename("datetime", inplace=True)

    # DataFrame to store past run inputs data
    past_runs_data = pd.DataFrame(index=data.index.drop_duplicates(keep='last'))

    # Reset index to calculate diff between datetime and request in forecast_runs()
    data.reset_index(drop=False, inplace=True)
    data.sort_values(by=['datetime', 'request'], ascending=True, inplace=True)

    for variable_id in variables:
        try:
            aux = data[['datetime', 'request', variable_id]].copy()
        except KeyError:
            # logging.logWarn("Dataset {} ignored for past_runs_variables creation. Reason: Not contained in weather forecasts.".format(variable_id))
            print("Dataset {} ignored for past_runs_variables creation. Reason: Not contained in weather forecasts.".format(variable_id))
            continue

        aux = forecast_runs(full_data=aux, variable_id=variable_id)
        past_runs_data = past_runs_data.join(aux)
    return past_runs_data


def create_temporal_inputs(data, variables):
    data = data.copy()
    data.index.rename("datetime", inplace=True)

    # DataFrame to store past run inputs data
    temporal_data = pd.DataFrame(index=data.index.drop_duplicates(keep='last'))

    # Reset index to calculate diff between datetime and request in forecast_runs()
    data.reset_index(drop=False, inplace=True)
    data.sort_values(by=['datetime', 'request'], ascending=True, inplace=True)

    for variable_id in variables:
        try:
            aux = data[['datetime', 'request', variable_id]].copy()
        except KeyError:
            # logging.logWarn("Dataset {} ignored for temporal_variables creation. Reason: Not contained in weather forecasts.".format(variable_id))
            print("Dataset {} ignored for temporal_variables creation. Reason: Not contained in weather forecasts.".format(variable_id))
            continue

        aux = temporal_indexes(full_data=aux, variable_id=variable_id, window_list=[3, 7, 11])
        temporal_data = temporal_data.join(aux)

    return temporal_data


def mod(u, v):
    return sqrt((u**2) + (v**2))


def direction(u, v):
    return (atan2(u, v)*(45.0/atan(1.0))) + 180