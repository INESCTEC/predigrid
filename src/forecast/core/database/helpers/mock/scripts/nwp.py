import numpy as np
import pandas as pd


def create_mock_nwp(latitude, longitude, variables,
                    date_start_utc, date_end_utc):
    print(f"Retrieving mock data for {latitude, longitude}")
    _st = date_start_utc.strftime("%Y-%m-%d %H:%M:%S")
    _et = date_end_utc.strftime("%Y-%m-%d %H:%M:%S")
    if (latitude, longitude) == (999, 999):
        # Create pd dataframe for inst 1
        _ts = pd.date_range(_st, _et, freq='H', tz=None)
        _rq = [x.strftime("%Y-%m-%d 00:00:00") for x in _ts]
        df_dict = {"datetime": _ts, "request": _rq}
        for v in variables:
            df_dict[v] = np.random.uniform(low=0, high=1, size=len(_ts))
        return pd.DataFrame(df_dict)
    elif (latitude, longitude) == (999, 998):
        # Scenario where there is no weather data for last 48h
        _ts = pd.date_range(_st, _et, freq='H', tz=None)[:-48]
        _rq = [x.strftime("%Y-%m-%d 00:00:00") for x in _ts]
        df_dict = {"datetime": _ts, "request": _rq}
        for v in variables:
            df_dict[v] = np.random.uniform(low=0, high=1, size=len(_ts))
        return pd.DataFrame(df_dict)
    elif (latitude, longitude) == (999, 997):
        # Create pd dataframe for inst 3
        _ts = pd.date_range(_st, _et, freq='H', tz=None)[:-11]
        _rq = [x.strftime("%Y-%m-%d 00:00:00") for x in _ts]
        df_dict = {"datetime": _ts, "request": _rq}
        for v in variables:
            df_dict[v] = np.random.uniform(low=0, high=1, size=len(_ts))
        return pd.DataFrame(df_dict)
    elif (latitude, longitude) == (0, 0):
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)
        _rq = [x.strftime("%Y-%m-%d 00:00:00") for x in _ts]
        df_dict = {"datetime": _ts, "request": _rq}
        for v in variables:
            df_dict[v] = np.nan
        return pd.DataFrame(df_dict)
    else:
        raise ValueError("Invalid latitude longitude")  # noqa