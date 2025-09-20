import os
import pandas as pd

__MOCK_PATH__ = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "csv"
)


def create_mock_inst_metadata(inst_id):
    print(f"Retrieving mock installation data for {inst_id}")
    _fp = os.path.join(__MOCK_PATH__, "installations.csv")
    _df = pd.read_csv(_fp)
    if inst_id in _df["id"].unique():
        # Create pd dataframe for inst 1
        return _df.loc[_df["id"] == inst_id, :]
    else:
        raise ValueError("Invalid mock inst_id.")


def create_mock_models_metadata(inst_id):
    print(f"Retrieving mock models data for {inst_id}")
    _fp = os.path.join(__MOCK_PATH__, "models_info.csv")
    _df = pd.read_csv(_fp)
    if inst_id in _df["id"].unique():
        # Create pd dataframe for inst 1
        _df = _df.loc[_df["id"] == inst_id, :]
        df_dict = _df.to_dict(orient="records")[0]
        df_dict["model_ref"] = eval(df_dict["model_ref"])
        df_dict["model_in_use"] = eval(df_dict["model_in_use"])
        return df_dict
    else:
        raise ValueError("Invalid mock inst_id.")
