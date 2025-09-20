import os
import re
import json
import gzip
import shutil
import pickle
import joblib
import numpy as np
import pandas as pd

from forecast_api.models.ModelClass import ModelClass

from core.database.helpers.cql_helpers import (
    save_obj_to_cql,
    load_obj_from_cql
)

QUANTILES = [0.5]
QUANTILES_TESTS = [0.5]


def make_tree_dirs(path_list):
    for dp in path_list:
        if not os.path.exists(dp):
            os.makedirs(dp)


def clean_models_dir(inst_id, register_type, models_list, path):
    """
    This is used at the end of a training method when models are saved locally.
    Removes any models that were not listed as having been trained.
    """
    path_model_folder = _generate_path_models(inst_id=inst_id,
                                              algorithm="dummy",
                                              register_type=register_type,
                                              name="dummy",
                                              path=path)
    path_model_folder = os.path.dirname(os.path.dirname(path_model_folder))
    refs_list = list(os.walk(path_model_folder))[0][1]
    refs_list = [ref.replace("\\", "_").replace("/", "_") for ref in refs_list]
    models_list_ = [m.replace("\\", "_").replace("/", "_") for m in models_list] # noqa
    for ref in refs_list:
        if ref not in models_list_:
            shutil.rmtree(os.path.join(path_model_folder, ref),
                          ignore_errors=True)


def save_load_train_dates(con, network_id, pt_id, model_id, model_obj=None,
                          action='save'):
    if action == 'save':
        if model_obj is None:
            raise ValueError("ERROR: No dates obj available!")
        save_obj_to_cql(con, network_id, pt_id, "train_dates", model_id,
                        model_obj)
        return model_obj
    elif action == 'load':
        model_obj = load_obj_from_cql(con, network_id, pt_id, "train_dates",
                                      model_id)
        return model_obj


def save_scalers(con, network_id, pt_id, model_id, x_scaler, y_scaler):
    save_obj_to_cql(con, network_id, pt_id, "x_scaler", model_id, x_scaler)
    save_obj_to_cql(con, network_id, pt_id, "y_scaler", model_id, y_scaler)


def load_scalers(con, network_id, pt_id, model_id):
    x_scaler = load_obj_from_cql(con, network_id, pt_id, "x_scaler", model_id)
    y_scaler = load_obj_from_cql(con, network_id, pt_id, "y_scaler", model_id)

    if x_scaler is None:
        raise FileNotFoundError("Error: X scaler saved obj. not found.")
    if y_scaler is None:
        raise FileNotFoundError("Error: Y scaler saved obj. not found.")

    return x_scaler, y_scaler


def save_scalers_to_file(path, x_scaler, y_scaler):
    joblib.dump(x_scaler, filename=os.path.join(path, "x_scaler.pkl"))
    joblib.dump(y_scaler, filename=os.path.join(path, "y_scaler.pkl"))


def load_scalers_from_file(path):
    path_x = os.path.join(path, "x_scaler.pkl")
    path_y = os.path.join(path, "y_scaler.pkl")

    # Check if path exists (file exists?)
    if os.path.exists(path_x) is False:
        raise FileNotFoundError("Error: X scaler saved obj. not found.")

    if os.path.exists(path_y) is False:
        raise FileNotFoundError("Error: Y scaler saved obj. not found.")

    # load Sklearn scaler objs.
    x_scaler = joblib.load(path_x)
    y_scaler = joblib.load(path_y)

    return x_scaler, y_scaler


def save_load_train_dates_to_file(path, dates=None, action='save'):
    if action == 'save':
        if dates is None:
            exit("ERROR! Dates obj must be specified to be saved.")
        joblib.dump(dates, filename=path)
        return dates
    elif action == 'load':
        dates = joblib.load(path)
        return dates


def verify_data_statistics(con, network_id, pt_id, model_id, data):
    data_stats = data.describe()
    old_stats = load_obj_from_cql(con, network_id, pt_id, "stats", model_id)

    if old_stats is None:
        save_obj_to_cql(con, network_id, pt_id, "stats", model_id, data_stats)
        return False

    to_update = []
    data_stats = data_stats.to_dict()
    for k, v in data_stats.items():
        prev_stats = old_stats.get(k)
        _prevmax = prev_stats.get("max", np.nan)
        _prevmin = prev_stats.get("min", np.nan)

        new_stats = data_stats.get(k)
        _newmax = new_stats.get("max", np.nan)
        _newmin = new_stats.get("min", np.nan)

        # if any element of comparison (_newmax, _prevmax, etc.) is nan,
        # the condition is false always. Tries to update
        # but crashes later on train phase with exception. so it's ok.
        if (_newmax > _prevmax) or (_newmin < _prevmin):
            if _newmax > _prevmax:
                old_stats[k]["max"] = _newmax
                # print("NEW {} bigger! {} > {}".format(k, _newmax, _prevmax))
            if _newmin < _prevmin:
                old_stats[k]["min"] = _newmin
                # print("NEW {} smaller! {} < {}".format(k, _newmin, _prevmin))
            to_update.append(False)
        else:
            to_update.append(True)

    # if one element of to_update is False, updatable is False, else it is True
    updatable = all(to_update)
    if not updatable:
        save_obj_to_cql(con, network_id, pt_id, "stats", model_id,
                        old_stats)  # overwrites previous file with new stats

    return updatable


def _generate_path_models(inst_id, algorithm, register_type, name, path):
    _ref = name.replace("\\", "_").replace("/", "_")
    if path is None:
        path_model_folder = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            ), 'files', 'models', inst_id, register_type, _ref, algorithm)
    else:
        path_model_folder = os.path.join(path, 'models', inst_id,
                                         register_type, _ref, algorithm)
    return path_model_folder


def _generate_path_stats(inst_id, path):
    if path is None:
        path_model_folder = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            ), 'files', 'models', inst_id)
    else:
        path_model_folder = os.path.join(path, 'models', inst_id)
    return path_model_folder

####################################
# Methods for local model management
####################################


def load_local_model(inst_id, algorithm, register_type, name, path):
    path_model_folder = _generate_path_models(inst_id, algorithm,
                                              register_type, name, path)
    os.makedirs(path_model_folder, exist_ok=True)
    filename = "model.gz"
    path_model_file = os.path.join(path_model_folder, filename)
    try:
        with gzip.open(path_model_file, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        model = None
    filename = "inputs.txt"
    path_model_file = os.path.join(path_model_folder, filename)
    try:
        with open(path_model_file, 'r') as f:
            inputs = eval(f.readline().strip())
    except FileNotFoundError:
        inputs = None
    return model, inputs


def save_local_model(obj, inputs, inst_id, algorithm,
                     register_type, name, path):
    path_model_folder = _generate_path_models(inst_id, algorithm,
                                              register_type, name, path)
    os.makedirs(path_model_folder, exist_ok=True)
    filename = "model.gz"
    path_model_file = os.path.join(path_model_folder, filename)
    with gzip.open(path_model_file, 'wb') as f:
        pickle.dump(obj, f)
    filename = "inputs.txt"
    path_model_file = os.path.join(path_model_folder, filename)
    with open(path_model_file, 'w') as f:
        f.write(str(inputs))


def save_local_scalers(x_scaler, y_scaler, inst_id, algorithm, register_type,
                       name, path):
    path_model_folder = _generate_path_models(inst_id, algorithm,
                                              register_type, name, path)
    os.makedirs(path_model_folder, exist_ok=True)
    for scaler, filename in zip([x_scaler, y_scaler],
                                ['x_scaler', 'y_scaler']):
        path_model_file = os.path.join(path_model_folder, filename + '.gz')
        if scaler is not None:
            with gzip.open(path_model_file, 'wb') as f:
                pickle.dump(scaler, f)


def load_local_scalers(inst_id, algorithm, register_type, name, path):
    path_model_folder = _generate_path_models(inst_id, algorithm,
                                              register_type, name, path)
    os.makedirs(path_model_folder, exist_ok=True)

    # x_scaler
    path_model_file = os.path.join(path_model_folder, 'x_scaler.gz')
    try:
        with gzip.open(path_model_file, 'rb') as f:
            x_scaler = pickle.load(f)
    except FileNotFoundError:
        x_scaler = None

    # y_scaler
    path_model_file = os.path.join(path_model_folder, 'y_scaler.gz')
    try:
        with gzip.open(path_model_file, 'rb') as f:
            y_scaler = pickle.load(f)
    except FileNotFoundError:
        y_scaler = None

    return x_scaler, y_scaler


def save_local_stats(stats_dict, inst_id, path):
    processed_stats = {
        reg_type: {
            stat_name: float(stat) for stat_name, stat in stats.items()
        }
        for reg_type, stats in stats_dict.items()
    }
    path_model_folder = _generate_path_stats(inst_id, path)
    os.makedirs(path_model_folder, exist_ok=True)
    path_model_file = os.path.join(path_model_folder, 'stats.json')
    with open(path_model_file, 'w') as f:
        json.dump(processed_stats, f)


def load_local_stats(inst_id, path):
    path_model_folder = _generate_path_stats(inst_id, path)
    os.makedirs(path_model_folder, exist_ok=True)
    path_model_file = os.path.join(path_model_folder, 'stats.json')
    with open(path_model_file, 'rb') as f:
        stats = json.loads(f.read())
    return stats


#############################################
# Methods for database level model management
#############################################

def save_db_model(con, obj, inputs, inst_id, algorithm,
                  register_type, name):
    if obj is None:
        raise ValueError("ERROR: No model available!")
    model_id = f"{name}_{algorithm}_{register_type}"
    save_obj_to_cql(con,
                    inst_id,
                    "model",
                    model_id,
                    obj,
                    split_serial=True,
                    split_serial_size=70)
    save_obj_to_cql(con,
                    inst_id,
                    "inputs",
                    model_id,
                    inputs,
                    split_serial=False)


def load_db_model(con, inst_id, algorithm, register_type, name):
    model_id = f"{name}_{algorithm}_{register_type}"
    model_obj = load_obj_from_cql(con,
                                  inst_id,
                                  "model",
                                  model_id,
                                  split_serial=True,
                                  split_serial_size=70)
    inputs = load_obj_from_cql(con,
                               inst_id,
                               "inputs",
                               model_id,
                               split_serial=False)
    return model_obj, inputs


def save_db_scalers(x_scaler, y_scaler, con, inst_id, algorithm, register_type,
                    name):
    for scaler, scaler_name in zip([x_scaler, y_scaler],
                                   ['x_scaler', 'y_scaler']):
        if scaler is not None:
            model_id = f"{name}_{algorithm}_{register_type}"
            save_obj_to_cql(con,
                            inst_id,
                            scaler_name,
                            model_id,
                            scaler,
                            split_serial=False)


def load_db_scalers(con, inst_id, algorithm, register_type, name):
    model_id = f"{name}_{algorithm}_{register_type}"
    x_scaler = load_obj_from_cql(con,
                                 inst_id,
                                 "x_scaler",
                                 model_id,
                                 split_serial=False)

    y_scaler = load_obj_from_cql(con,
                                 inst_id,
                                 "y_scaler",
                                 model_id,
                                 split_serial=False)
    return x_scaler, y_scaler


def save_db_stats(con, stats_dict, inst_id):
    processed_stats = {
        reg_type: {
            stat_name: float(stat) for stat_name, stat in stats.items()
        }
        for reg_type, stats in stats_dict.items()
    }
    save_obj_to_cql(con,
                    inst_id,
                    "stats",
                    "any",
                    processed_stats,
                    split_serial=False)


def load_db_stats(con, inst_id):
    processed_stats = load_obj_from_cql(con,
                                        inst_id,
                                        "stats",
                                        "any",
                                        split_serial=False)
    return processed_stats


########
# Legacy
########


def save_load_model(con, network_id, pt_id, model_id,
                    model_obj=None, action='save', split_serial=False,
                    split_serial_size=10):
    if action == 'save':
        if model_obj is None:
            raise ValueError("ERROR: No model available!")
        save_obj_to_cql(con, network_id, pt_id, "model", model_id,
                        model_obj, split_serial, split_serial_size)
        return model_obj
    elif action == 'load':
        model_obj = load_obj_from_cql(con, network_id, pt_id,
                                      "model", model_id, split_serial,
                                      split_serial_size)
        return model_obj


def save_load_data_stats(con, network_id, pt_id, model_id,
                         model_obj=None, action='save'):
    if action == 'save':
        if model_obj is None:
            raise ValueError("ERROR: No stats available!")
        save_obj_to_cql(con, network_id, pt_id,
                        "stats", model_id, model_obj)
        return model_obj
    elif action == 'load':
        model_obj = load_obj_from_cql(con, network_id, pt_id,
                                      "stats", model_id)
        return model_obj


def save_load_wdays_ref(con, network_id, pt_id, model_id,
                        model_obj=None, action='save'):
    if action == 'save':
        if model_obj is None:
            raise ValueError("ERROR: No stats available!")
        save_obj_to_cql(con, network_id, pt_id, "wdays_ref", model_id,
                        model_obj)
        return model_obj
    elif action == 'load':
        model_obj = load_obj_from_cql(con, network_id, pt_id, "wdays_ref",
                                      model_id)
        return model_obj


def search_similar_days(target_data, lookback_days=None):
    from scipy.spatial.distance import pdist, squareform

    def fetch_lag_ref(wday, similar_day):
        if wday <= similar_day:
            lag_step = wday - similar_day
            lag_step = [6, 1, 2, 3, 4, 5, 6][lag_step]
        else:
            lag_step = wday - similar_day

        return lag_step

    target_data = target_data.copy().dropna().to_frame()

    if lookback_days is not None:
        target_data = target_data[:target_data.index.max() - pd.DateOffset(
            days=lookback_days)]  # noqa

    nr_avail_wdays = len(target_data.index.weekday.unique())
    nr_avail_days = len(np.unique(target_data.dropna().index.date))

    if (nr_avail_wdays == 7) and (nr_avail_days > 14) and (
    not target_data.empty):  # noqa
        # Find Most Correlated days:
        pivot_wday = pd.pivot_table(data=target_data, values="real",
                                    index=target_data.index.hour,
                                    columns=target_data.index.weekday)

        find_best_matches = {}
        find_best_lags = {}
        distances = pdist(pivot_wday.transpose().values, metric='euclidean')
        dist_matrix = squareform(distances)
        dist_matrix = pd.DataFrame(dist_matrix,
                                   index=np.arange(0, 7),
                                   columns=np.arange(0, 7))

        for wday in dist_matrix.index:
            correl_arr = np.array(dist_matrix.loc[wday, ])
            find_best_matches[wday] = list(np.argsort(correl_arr)[1:])
            find_best_lags[wday] = [fetch_lag_ref(wday, x)
                                    for x in find_best_matches[wday]]

        return find_best_matches, find_best_lags


def search_best_lag(target_series, inputs_dframe, lookback_days=None):
    inputs_dframe = inputs_dframe.copy()[[x for x in inputs_dframe.columns
                                          if "real" in x]]
    target_series = target_series.copy()
    target_series_old_name = target_series.name
    target_series.name = "real"

    # -- Find Inputs With
    _dist_df = inputs_dframe.join(target_series).dropna()

    # -- Filter for
    if lookback_days is not None:
        _dist_df = _dist_df[:(_dist_df.index.max() - pd.DateOffset(
            days=lookback_days))]  # noqa

    if _dist_df.empty:
        return dict((wday, []) for wday in range(0, 7))

    try:
        new_cols = []
        for c in inputs_dframe.columns:
            col_ref = c.split('_')[1]
            _dist_df.loc[:, col_ref] = abs(
                _dist_df.loc[:, c] - _dist_df.loc[:, "real"])  # noqa
            new_cols.append(col_ref)

        _dist_df = _dist_df[new_cols].dropna()
        _dist_df_week = _dist_df.groupby(_dist_df.index.weekday).mean()

        best_lags_dict = {}
        for wday in range(0, 7):
            best_lags_dict[wday] = [int(x) for x in (_dist_df_week
                                                     .loc[wday, :]
                                                     .rank()
                                                     .sort_values()
                                                     .index)]
    except Exception as ex:
        print(f"Error searching best lags ({target_series_old_name}):",
              repr(ex))
        return dict((wday, []) for wday in range(0, 7))

    return best_lags_dict


def postprocessing_layer(data, max_obs, min_obs):
    a = ModelClass()
    p = re.compile('q[0-9]{2}$')
    quantile_col = [col for col in data.columns if p.match(col)]
    # If there is only one quantile column of forecasts, no processing is needed. # noqa
    if len(quantile_col) == 1:
        return data

    max_obs_data_p = max_obs
    min_obs_data_p = min_obs

    for i in data.index:
        val = data.loc[i, quantile_col].values.astype(np.float64)
        val = np.where(val > min_obs_data_p, val, min_obs_data_p)
        val = np.where(val < max_obs_data_p * 1.5, val, max_obs_data_p)
        val_idx = np.arange(0, len(val))
        modified_z_score = lambda x: (x - np.median(x)) / np.std(x)
        q50_idx = int(np.floor(len(val) / 2))
        val_lower = val[:q50_idx + 1]
        val_higher = val[q50_idx:]

        val_lower_diff = val_lower[1:] - val_lower[:-1]
        val_higher_diff = val_higher[1:] - val_higher[0:-1]
        d = np.append(val_lower_diff, val_higher_diff)
        sc = abs(modified_z_score(d))
        to_remove = sc > 1.4
        to_remove = np.insert(to_remove, q50_idx, False)

        p = np.polyfit(x=val_idx[~to_remove], y=val[~to_remove], deg=3)
        v = np.polyval(p=p, x=val_idx[to_remove])
        val[val_idx[to_remove]] = v
        val = np.where(val > min_obs_data_p, val, min_obs_data_p)
        data.loc[i, quantile_col] = val

    data = a.reorder_quantiles(data)

    return data


def forecasts_limiter(max_limit, min_limit, predictions,
                      f_col="forecast", interpolate_faulty=False):
    max_limit_ref = max_limit * 2.5 if max_limit >= 0 else max_limit * -2.5
    min_limit_ref = min_limit  # * 1.5 if min_limit <= 0 else min_limit * -1.5

    mask_high = predictions[f_col] <= max_limit_ref
    mask_low = predictions[f_col] > min_limit_ref

    if interpolate_faulty:
        # -- Flag to detect missing interpolations:
        _fflag = 999999

        # -- Replace by NaN and try interpolation
        predictions.loc[~mask_low, :] = np.nan
        predictions = (predictions
                       .interpolate(method="linear",
                                    limit=4,
                                    limit_direction="backward")
                       .fillna(-_fflag))
        predictions.loc[predictions[f_col] == -_fflag,
        :] = min_limit  # Replace non-interpolatable by minimum # noqa

        # -- Replace by NaN and try interpolation
        mask_high = predictions[f_col] <= max_limit_ref
        predictions.loc[~mask_high, :] = np.nan
        predictions = (predictions
                       .interpolate(method="linear",
                                    limit=4,
                                    limit_direction="backward")
                       .fillna(_fflag))
        predictions.loc[predictions[f_col] == _fflag,
        :] = max_limit  # Replace non-interpolatable by maximum # noqa

        return predictions
    else:
        return predictions.loc[mask_high & mask_low, :]
