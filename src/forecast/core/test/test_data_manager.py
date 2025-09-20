# flake8: noqa

import pytest
import pandas as pd

from core.database.DataManager import DataManager


def test_init_class_attr():
    """
    Check values of initial attributes
    """
    db_manager = DataManager(db_con=None, inst_id="inst11")
    assert db_manager.engine is None
    assert db_manager.launch_time is None
    assert db_manager.launch_time_hour_utc is None
    assert db_manager.f_horizon is None
    assert db_manager.forecast_range is None
    assert db_manager.mode is None
    assert db_manager.use_full_historical is None
    assert db_manager.last_date_horizon_utc is None
    assert db_manager.inst_id == "inst11"
    assert db_manager.inst_metadata == {}
    assert db_manager.inst_type is None
    assert db_manager.country is None
    assert db_manager.forecast_tz is None
    assert db_manager.f_targets == []
    assert db_manager.data_tracker == {"P": False, "Q": False, "NWP": False}
    assert db_manager.nwp_vars is None
    assert db_manager.models_configs == {}


def test_init_launch_time():
    """
    Check valid and invalid launch times
    """
    lt_list_ok = [
        "2020-01-01 00:00:00",
        "2020-01-01 03:00:00",
        "2020-01-01 03:15:00",
        "2020-01-01 03:15:34",
    ]
    lt_list_error = [
        "12-01-2020",
        "2020-01-01 25",
        "2020-01-01 23:60",
        "2020-01-01 23:59:60",
    ]

    fmt = "%Y-%m-%d %H:%M:%S"
    db_manager = DataManager(db_con=None, inst_id="inst11")
    # -- Fail conditions expected:
    for lt in lt_list_error:
        with pytest.raises(ValueError):
            db_manager.set_launch_time(launch_time=lt)

    # -- Correct conditions expected:
    for lt in lt_list_ok:
        db_manager.set_launch_time(launch_time=lt)
        assert db_manager.launch_time == pd.to_datetime(lt,
                                                        format=fmt,
                                                        utc=True)
        assert db_manager.launch_time_hour_utc == pd.to_datetime(
            lt,
            format=fmt,
            utc=True
        ).replace(minute=0, second=0, microsecond=0)


def test_init_forecast_horizon():
    """
    Check valid and invalid forecast_horizons
    """
    db_manager = DataManager(db_con=None, inst_id="inst11")
    # -- Fail conditions expected:
    with pytest.raises(ValueError):
        db_manager.set_forecast_horizon(200)  # max horizon = 168
    with pytest.raises(AttributeError):
        db_manager.set_forecast_horizon(168)  # max horizon = 168

    # -- Correct conditions expected:
    horizon_list = [24, 48, 72, 96, 168]
    launch_time = "2020-01-01 00:00:00"
    for horizon in horizon_list:
        db_manager.set_launch_time(launch_time=launch_time)
        db_manager.set_forecast_horizon(horizon)
        assert db_manager.f_horizon == horizon
        assert type(db_manager.forecast_range) == pd.DatetimeIndex
        assert len(db_manager.forecast_range) == horizon
        assert str(db_manager.forecast_range.freq) == "<Hour>"
        _1st_dt = db_manager.forecast_range[0].strftime("%Y-%m-%d %H:%M:%S")
        assert _1st_dt == launch_time


def test_init_mode():
    """
    Check valid and invalid modes
    """
    db_manager = DataManager(db_con=None, inst_id="inst11")
    # -- Fail conditions expected:
    with pytest.raises(ValueError):
        db_manager.set_mode(mode="bob")  # valid modes == ["train", "forecast"]

    # -- Correct conditions expected:
    db_manager.set_mode(mode="train")
    assert db_manager.mode == "train"
    assert db_manager.use_full_historical is True
    db_manager.set_mode(mode="forecast")
    assert db_manager.use_full_historical is False


def test_configs_load_forecast():
    """
    Check if all configs are loaded as expected for load inst type
    """
    db_manager = DataManager(db_con=None, inst_id="inst11")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    # -- Fail conditions expected:
    with pytest.raises(NameError):
        # Fails to load without declaring launch_time:
        db_manager.load_configs()
    with pytest.raises(NameError):
        # Fails to load without declaring launch_time:
        db_manager.set_mode("train")
        db_manager.load_configs()
    with pytest.raises(NameError):
        # Fails to load without declaring forecast horizon:
        db_manager.set_mode("train")
        db_manager.set_launch_time(launch_time=launch_time)
        db_manager.load_configs()

    # -- Correct conditions expected:
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="train")
    db_manager.load_configs()

    assert db_manager.country == "portugal"
    assert db_manager.inst_type == "load"
    assert db_manager.source_nwp == "meteogalicia"
    assert db_manager.latitude_nwp == 999
    assert db_manager.longitude_nwp == 999
    assert db_manager.has_generation is False
    assert db_manager.net_power_types == 'PQ'
    assert isinstance(db_manager.models_metadata, dict)
    assert list(db_manager.models_metadata.keys()) == ['id', 'last_train', 'last_updated', 'model_in_use', 'model_ref', 'to_train']
    assert db_manager.forecast_tz == "Europe/Lisbon"
    assert db_manager.register_types == ["P", "Q"]
    assert db_manager.f_targets == [f"real_{x}" for x in ["P", "Q"]]
    assert isinstance(db_manager.nwp_vars, list)
    assert isinstance(db_manager.tables, dict)
    assert list(db_manager.tables.keys()) == ['measurements', 'forecasts', 'nwp']
    assert db_manager.holyclass is not None
    assert db_manager.nwp_vars == ["temp"]


def test_configs_load_gen_forecast():
    """
    Check if all configs are loaded as expected for load inst type
    """
    db_manager = DataManager(db_con=None, inst_id="inst12")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    # -- Fail conditions expected:
    with pytest.raises(NameError):
        # Fails to load without declaring launch_time:
        db_manager.load_configs()
    with pytest.raises(NameError):
        # Fails to load without declaring forecast horizon:
        db_manager.set_launch_time(launch_time=launch_time)
        db_manager.load_configs()
    with pytest.raises(NameError):
        # Fails to load without declaring mode:
        db_manager.set_launch_time(launch_time=launch_time)
        db_manager.set_forecast_horizon(forecast_horizon=horizon)
        db_manager.load_configs()

    # -- Correct conditions expected:
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="train")
    db_manager.load_configs()

    assert db_manager.country == "portugal"
    assert db_manager.inst_type == "load"
    assert db_manager.source_nwp == "meteogalicia"
    assert db_manager.latitude_nwp == 999
    assert db_manager.longitude_nwp == 998
    assert db_manager.has_generation is True
    assert db_manager.net_power_types == 'PQ'
    assert isinstance(db_manager.models_metadata, dict)
    assert list(db_manager.models_metadata.keys()) == ['id', 'last_train', 'last_updated', 'model_in_use', 'model_ref', 'to_train']
    assert db_manager.forecast_tz == "Europe/Lisbon"
    assert db_manager.register_types == ["P", "Q"]
    assert db_manager.f_targets == [f"real_{x}" for x in ["P", "Q"]]
    assert isinstance(db_manager.nwp_vars, list)
    assert isinstance(db_manager.tables, dict)
    assert list(db_manager.tables.keys()) == ['measurements', 'forecasts', 'nwp']
    assert db_manager.holyclass is not None
    assert db_manager.nwp_vars == ["temp", "swflx", "cfl", "cfm"]


def test_configs_wind_forecast():
    """
    Check if all configs are loaded as expected for wind inst type
    """
    db_manager = DataManager(db_con=None, inst_id="inst21")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    # -- Fail conditions expected:
    with pytest.raises(NameError):
        # Fails to load without declaring launch_time:
        db_manager.load_configs()
    with pytest.raises(NameError):
        # Fails to load without declaring forecast horizon:
        db_manager.set_launch_time(launch_time=launch_time)
        db_manager.load_configs()
    with pytest.raises(NameError):
        # Fails to load without declaring mode:
        db_manager.set_launch_time(launch_time=launch_time)
        db_manager.set_forecast_horizon(forecast_horizon=horizon)
        db_manager.load_configs()

    # -- Correct conditions expected:
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="train")
    db_manager.load_configs()

    assert db_manager.country == "portugal"
    assert db_manager.inst_type == "wind"
    assert db_manager.source_nwp == "meteogalicia"
    assert db_manager.latitude_nwp == 999
    assert db_manager.longitude_nwp == 999
    assert db_manager.has_generation is True
    assert db_manager.net_power_types == 'PQ'
    assert isinstance(db_manager.models_metadata, dict)
    assert list(db_manager.models_metadata.keys()) == ['id', 'last_train',
                                                       'last_updated',
                                                       'model_in_use',
                                                       'model_ref', 'to_train']
    assert db_manager.forecast_tz == "Europe/Lisbon"
    assert db_manager.register_types == ["P", "Q"]
    assert db_manager.f_targets == [f"real_{x}" for x in ["P", "Q"]]
    assert isinstance(db_manager.nwp_vars, list)
    assert isinstance(db_manager.tables, dict)
    assert list(db_manager.tables.keys()) == ['measurements', 'forecasts',
                                              'nwp']
    assert db_manager.holyclass is not None
    assert db_manager.nwp_vars == [
        'u', 'ulev1', 'ulev2', 'ulev3',
        'v', 'vlev1', 'vlev2', 'vlev3'
    ]


def test_configs_solar_forecast():
    """
    Check if all configs are loaded as expected for solar inst type
    """
    db_manager = DataManager(db_con=None, inst_id="inst31")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    # -- Fail conditions expected:
    with pytest.raises(NameError):
        # Fails to load without declaring launch_time:
        db_manager.load_configs()
    with pytest.raises(NameError):
        # Fails to load without declaring forecast horizon:
        db_manager.set_launch_time(launch_time=launch_time)
        db_manager.load_configs()
    with pytest.raises(NameError):
        # Fails to load without declaring mode:
        db_manager.set_launch_time(launch_time=launch_time)
        db_manager.set_forecast_horizon(forecast_horizon=horizon)
        db_manager.load_configs()

    # -- Correct conditions expected:
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="train")
    db_manager.load_configs()

    assert db_manager.country == "portugal"
    assert db_manager.inst_type == "solar"
    assert db_manager.source_nwp == "meteogalicia"
    assert db_manager.latitude_nwp == 999
    assert db_manager.longitude_nwp == 999
    assert db_manager.has_generation is True
    assert db_manager.net_power_types == 'PQ'
    assert isinstance(db_manager.models_metadata, dict)
    assert list(db_manager.models_metadata.keys()) == ['id', 'last_train',
                                                       'last_updated',
                                                       'model_in_use',
                                                       'model_ref', 'to_train']
    assert db_manager.forecast_tz == "Europe/Lisbon"
    assert db_manager.register_types == ["P", "Q"]
    assert db_manager.f_targets == [f"real_{x}" for x in ["P", "Q"]]
    assert isinstance(db_manager.nwp_vars, list)
    assert isinstance(db_manager.tables, dict)
    assert list(db_manager.tables.keys()) == ['measurements', 'forecasts',
                                              'nwp']
    assert db_manager.holyclass is not None
    assert db_manager.nwp_vars == ["swflx", "cfl", "cfm", "cfh", "cft"]


def test_get_data_train_mode():
    """
    Check if correct number of historical timestamps are loaded for both
    train or forecast modes

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - For 'train' mode -> Queries 2 years of data

    """
    db_manager = DataManager(db_con=None, inst_id="inst11")
    db_manager.activate_unit_tests_mode()
    horizon = 168
    launch_time = "2020-01-05 00:00:00"

    # -- Correct conditions expected:
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    # -- Correct conditions expected:
    db_manager.set_mode(mode="train")
    dataset, model_config = db_manager.get_dataset()

    # Check if first date in historical is as expected
    expected_first_date = pd.to_datetime(launch_time, utc=True) - pd.DateOffset(years=2)  # noqa
    expected_first_date = expected_first_date.tz_convert("Europe/Lisbon")
    assert dataset.index[0] == expected_first_date


def test_get_data_forecast_mode():
    """
    Check if correct number of historical timestamps are loaded for both
    train or forecast modes

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - For 'forecast' mode -> Queries 2 years of data

    """
    db_manager = DataManager(db_con=None, inst_id="inst11")
    db_manager.activate_unit_tests_mode()
    horizon = 168
    launch_time = "2020-01-05 00:00:00"

    # -- Correct conditions expected:
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    # -- Correct conditions expected:
    db_manager.set_mode(mode="forecast")
    dataset, model_config = db_manager.get_dataset()

    # Check if first date in historical is as expected
    expected_first_date = pd.to_datetime(launch_time, utc=True) - pd.DateOffset(days=31)  # noqa
    expected_first_date = expected_first_date.tz_convert("Europe/Lisbon")
    assert dataset.index[0] == expected_first_date


def test_check_model_config_dict():
    """
    Check if model_config dictionaries are created as expected
    - Normal behaviour
    """
    db_manager = DataManager(db_con=None, inst_id="inst11")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="forecast")
    dataset, model_config = db_manager.get_dataset()

    expected_horizon = {'D', 'D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6'}
    expected_refs = {
        "P": {'D-7', 'D-1/D-7', 'D-2/D-7', 'D-3/D-7'},
        "Q": {'D-7'}
    }

    for rtype in ["P", "Q"]:
        assert set(model_config[rtype]['models'].keys()) == expected_horizon
        assert set(model_config[rtype]['models'].values()) == expected_refs[rtype]  # noqa


def test_no_last_2_days_historical():
    """
    Check if model_config dictionaries are created as expected.

    Scenario:
    - Removed last 2 days of measurements data

    Expected:
    - Model adapts and considers exclusively combination of lags D-3 and D-7

    """
    db_manager = DataManager(db_con=None, inst_id="inst14")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="forecast")
    dataset, model_config = db_manager.get_dataset()

    expected_horizon = {'D', 'D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6'}
    expected_refs = {
        "P": {'D-7', 'D-3/D-7'},
        "Q": {'D-7'}
    }

    for rtype in ["P", "Q"]:
        assert set(model_config[rtype]['models'].keys()) == expected_horizon
        assert set(model_config[rtype]['models'].values()) == expected_refs[rtype]  # noqa

def test_no_last_4_days_historical():
    """
    Check if model_config dictionaries are created as expected.

    Scenario:
    - Removed last 4 days of measurements data

    Expected:
    - Model adapts and considers exclusively lags on D-7

    """
    db_manager = DataManager(db_con=None, inst_id="inst15")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="forecast")
    dataset, model_config = db_manager.get_dataset()

    expected_horizon = {'D', 'D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6'}
    expected_refs = {
        "P": {'D-7'},
        "Q": {'D-7'}
    }

    for rtype in ["P", "Q"]:
        assert set(model_config[rtype]['models'].keys()) == expected_horizon
        assert set(model_config[rtype]['models'].values()) == expected_refs[rtype]  # noqa


def test_no_last_2_days_nwp():
    """
    Check if model_config dictionaries are created as expected.

    Scenario:
    - Removed last 2 days of NWP in forecast horizon

    Expected:
    - Defines model = 'D-7/no_weather' for last 2 days in horizon

    """
    db_manager = DataManager(db_con=None, inst_id="inst12")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="forecast")
    dataset, model_config = db_manager.get_dataset()

    expected_horizon = {'D', 'D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6'}
    expected_refs = {
        "P": {'D-2/D-7', 'D-7/no_weather',
                       'D-3/D-7', 'D-1/D-7', 'D-7'},
        "Q": {'D-7', 'D-7/no_weather'}
    }

    for rtype in ["P", "Q"]:
        assert set(model_config[rtype]['models'].keys()) == expected_horizon
        assert set(model_config[rtype]['models'].values()) == expected_refs[rtype]  # noqa


def test_no_measurements_train():
    """
    Check if model_config dictionaries are created as expected.

    Scenario:
    - Removed historical data

    Expected:
    - Model configs hours_in_hist field = 0
    - DropNA on dataset -> empty (no rows with data in every variable)
    - DataManager.data_tracker should have 'P' and 'Q' fields set to False

    """
    db_manager = DataManager(db_con=None, inst_id="inst13")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="train")
    dataset, model_config = db_manager.get_dataset()
    assert dataset.dropna().empty
    assert db_manager.data_tracker["P"] is False
    assert db_manager.data_tracker["Q"] is False
    assert db_manager.data_tracker["NWP"] is True

    expected_horizon = set()
    expected_refs = {
        "P": set(),
        "Q": set()
    }

    for rtype in ["P", "Q"]:
        assert set(model_config[rtype]['models'].keys()) == expected_horizon
        assert set(model_config[rtype]['models'].values()) == expected_refs[rtype]  # noqa
        assert model_config[rtype]["hours_in_hist"] == 0


def test_no_measurements_forecast():
    """
    Check if model_config dictionaries are created as expected.

    Scenario:
    - Removed historical data

    Expected:
    - Model configs hours_in_hist field = 0
    - DropNA on dataset -> empty (no rows with data in every variable)
    - DataManager.data_tracker should have 'P' and 'Q' fields set to False

    """
    db_manager = DataManager(db_con=None, inst_id="inst13")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="forecast")
    dataset, model_config = db_manager.get_dataset()
    assert dataset.dropna().empty
    assert db_manager.data_tracker["P"] is False
    assert db_manager.data_tracker["Q"] is False
    assert db_manager.data_tracker["NWP"] is True

    expected_horizon = {'D', 'D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6'}
    expected_refs = {
        "P": {"backup_weather", "backup_no_weather"},
        "Q": {"backup_weather", "backup_no_weather"}
    }

    for rtype in ["P", "Q"]:
        assert set(model_config[rtype]['models'].keys()) == expected_horizon
        assert set(model_config[rtype]['models'].values()) == expected_refs[rtype]  # noqa
        assert model_config[rtype]["hours_in_hist"] == 0


def test_no_nwp():
    """
    Check if model_config dictionaries are created as expected.

    Scenario:
    - Removed NWP data

    Expected:
    - Model configs hours_in_hist field = 0
    - DropNA on dataset -> empty (no rows with data in every variable)
    - DataManager.data_tracker should have 'NWP' field set to False

    """
    db_manager = DataManager(db_con=None, inst_id="inst16")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="train")
    dataset, model_config = db_manager.get_dataset()
    assert db_manager.data_tracker["P"] is True
    assert db_manager.data_tracker["Q"] is True
    assert db_manager.data_tracker["NWP"] is False

    expected_horizon = {'D', 'D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6'}
    expected_refs = {
        "P": {"D-7/no_weather"},
        "Q": {"D-7/no_weather"}
    }

    for rtype in ["P", "Q"]:
        assert set(model_config[rtype]['models'].keys()) == expected_horizon
        assert set(model_config[rtype]['models'].values()) == expected_refs[rtype]  # noqa
        assert model_config[rtype]["hours_in_hist"] == 0


def test_backup_mix():
    """
    Check if model_config dictionaries are created as expected.

    Scenario:
    - Removed NWP data for last 12 hours in horizon
    - Removed measurements data for last month (to avoid creation of avg_profile)

    Expected:
    - Model for last horizon should be "backup_mix"
    - Remaining horizon model's should be "backup_weather"

    """
    db_manager = DataManager(db_con=None, inst_id="inst17")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="forecast")
    dataset, model_config = db_manager.get_dataset()

    expected_horizon = {'D', 'D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6'}
    expected_refs = {
        "P": {'backup_weather', 'backup_mix'},
        "Q": {'backup_weather', 'backup_mix'},
    }

    for rtype in ["P", "Q"]:
        assert set(model_config[rtype]['models'].keys()) == expected_horizon
        assert set(model_config[rtype]['models'].values()) == expected_refs[rtype]  # noqa
        # last model should be backup mix
        assert model_config[rtype]['models']["D+6"] == "backup_mix"


def test_holidays():
    """
    Check if DataManager sets correct attributes when holidays are present
    in the horizon

    Scenario:
    - Horizon has holidays

    Expected:
    - Attribute "has_holidays" is True
    - Dataset generated has two years of data

    """
    db_manager = DataManager(db_con=None, inst_id="inst11")
    db_manager.activate_unit_tests_mode()

    horizon = 168
    launch_time = "2020-01-01 00:00:00"

    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=horizon)
    db_manager.set_mode(mode="forecast")
    dataset, model_config = db_manager.get_dataset()

    assert db_manager.has_holidays

    # Check if first date in historical is as expected
    expected_first_date = pd.to_datetime(launch_time, utc=True) - pd.DateOffset(years=2)  # noqa
    expected_first_date = expected_first_date.tz_convert("Europe/Lisbon")
    assert dataset.index[0] == expected_first_date