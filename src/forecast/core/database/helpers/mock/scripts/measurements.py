import numpy as np
import pandas as pd


def create_mock_measurements(inst_id, date_start_utc, date_end_utc):
    print(f"Retrieving mock data for {inst_id}")
    _st = date_start_utc.strftime("%Y-%m-%d %H:%M:%S")
    _et = date_end_utc.strftime("%Y-%m-%d %H:%M:%S")
    if inst_id == "inst11":
        # Create pd dataframe for inst 1
        _ts = pd.date_range(_st, _et, freq='H', tz=None)
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst12":
        # Create pd dataframe for inst 2
        _ts = pd.date_range(_st, _et, freq='H', tz=None)[-744:]
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst13":
        # Create pd dataframe for inst 3
        return pd.DataFrame({"datetime": [], "value": []})
    elif inst_id == "inst14":
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)[:-48]
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst15":
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)[:-96]
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst16":
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst17":
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)[:-744]
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst21":
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst22":
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst23":
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)[-744:]
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst31":
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst32":
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    elif inst_id == "inst33":
        # Create pd dataframe for inst 4
        _ts = pd.date_range(_st, _et, freq='H', tz=None)[-744:]
        _v = np.random.uniform(low=0, high=0.5, size=len(_ts))
        return pd.DataFrame({"datetime": _ts, "value": _v})
    else:
        raise ValueError("Invalid mock installation")  # noqa
