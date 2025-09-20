import numpy as np
import pandas as pd


def create_temp_inputs_df(inputs_df: pd.DataFrame,
                          temp_col_id: str,
                          daily_min_max: bool = False,
                          roll_ema: bool = False,
                          roll_std: bool = False,
                          roll_diff: bool = False,
                          power_3: bool = False):
    """
    Function used to apply feature-engineering techniques to the temperature
    time series in order to extract new features.


    :param inputs_df: (:obj:`pandas.DataFrame`) DataFrame with all the input
    data for forecasting models.
    :param temp_col_id: (:obj:`str`) Name of the temperature column.
    :param daily_min_max: (:obj:`bool`)
    :param roll_ema: (:obj:`bool`)
    :param roll_std: (:obj:`bool`)
    :param roll_diff: (:obj:`bool`)
    :param power_3: (:obj:`bool`)
    :return: (:obj:`pandas.DataFrame`) Original inputs_df + new temperature
    features.
    """
    created_data = inputs_df[
        temp_col_id].copy()  # select specific temperature column to create the variables with # noqa
    created_data = created_data.to_frame()  # convert series to dataframe
    new_inputs = []
    if daily_min_max:
        # Calculate Max daily based values:
        dmax_name = f"dmax_{temp_col_id}_day"
        created_data.loc[:, dmax_name] = created_data[temp_col_id].resample(
            "D").max()  # noqa
        created_data.loc[:, dmax_name] = created_data[dmax_name].fillna(
            method="ffill", limit=24)  # noqa

        # Calculate Min daily based values:
        dmin_name = f"dmin_{temp_col_id}_day"
        created_data.loc[:, dmin_name] = created_data[temp_col_id].resample(
            "D").min()  # noqa
        created_data.loc[:, dmin_name] = created_data[dmin_name].fillna(
            method="ffill", limit=24)  # noqa

        # Calculate Min daily based values:
        davg_name = f"davg_{temp_col_id}_day"
        created_data.loc[:, davg_name] = created_data[temp_col_id].resample(
            "D").mean()  # noqa
        created_data.loc[:, davg_name] = created_data[davg_name].fillna(
            method="ffill", limit=24)  # noqa

        new_inputs.extend([dmax_name, dmin_name, davg_name])

    if roll_ema:
        # ---------------- Exponential Moving Averages (EMA) -----------------
        ema_w12_name = f"ema_{temp_col_id}_12"
        ema_w24_name = f"ema_{temp_col_id}_24"
        ema_w48_name = f"ema_{temp_col_id}_48"
        created_data.loc[:, ema_w12_name] = created_data.loc[:, temp_col_id].ewm(span=12, min_periods=1, adjust=False).mean() # noqa
        created_data.loc[:, ema_w24_name] = created_data.loc[:, temp_col_id].ewm(span=24, min_periods=1, adjust=False).mean() # noqa
        created_data.loc[:, ema_w48_name] = created_data.loc[:, temp_col_id].ewm(span=48, min_periods=1, adjust=False).mean() # noqa

        new_inputs.extend([ema_w12_name, ema_w24_name, ema_w48_name])

    if roll_std:
        # ---------------- Simple Moving Averages (std) ---------------------
        std_w3_name = f"std_{temp_col_id}_3"
        std_w6_name = f"std_{temp_col_id}_6"
        std_w12_name = f"std_{temp_col_id}_12"
        created_data.loc[:, std_w3_name] = created_data.loc[:, temp_col_id].rolling(window=3, center=False, min_periods=1).apply(np.std) # noqa
        created_data.loc[:, std_w6_name] = created_data.loc[:, temp_col_id].rolling(window=6, center=False, min_periods=1).apply(np.std) # noqa
        created_data.loc[:, std_w12_name] = created_data.loc[:, temp_col_id].rolling(window=12, center=False, min_periods=1).apply(np.std) # noqa

        new_inputs.extend([std_w3_name, std_w6_name, std_w12_name])

    if roll_diff:
        roll_dif_2 = f"diff_{temp_col_id}_2"
        roll_dif_4 = f"diff_{temp_col_id}_4"
        roll_dif_6 = f"diff_{temp_col_id}_6"
        created_data.loc[:, roll_dif_2] = created_data.loc[:, temp_col_id].diff(1) # noqa
        created_data.loc[:, roll_dif_4] = created_data.loc[:, temp_col_id].diff(3) # noqa
        created_data.loc[:, roll_dif_6] = created_data.loc[:, temp_col_id].diff(5) # noqa

        new_inputs.extend([roll_dif_2, roll_dif_4, roll_dif_6])

    if power_3:
        power_3_col = f"power3_{temp_col_id}"
        created_data.loc[:, power_3_col] = created_data.loc[:, temp_col_id] ** 3 # noqa

        new_inputs.extend([power_3_col])

    # Join created data to original inputs df:
    created_data.drop(temp_col_id, 1, inplace=True)
    inputs_df = inputs_df.join(created_data, how="left")
    del created_data

    return new_inputs, inputs_df
