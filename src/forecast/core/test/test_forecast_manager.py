import os
import pytest

from core.forecast.ForecastManager import ForecastManager
from core.database.DataManager import DataManager


def test_init_class_attr():
    """
    Check initial attributes
    """
    f_manager = ForecastManager()

    assert f_manager.configs == {}
    assert f_manager.db_manager is None
    assert f_manager.model_ref is None
    assert f_manager.model_in_use is None
    assert f_manager.dataset.empty
    assert f_manager.P_forecasts.empty
    assert f_manager.algorithms == {}
    assert f_manager.inst_id is None
    assert f_manager.inst_type is None
    assert f_manager.db_con is None
    assert f_manager.models_path is None


def test_assign_db_manager():
    """
    Check DataManager assignment
    """
    db_manager = DataManager(db_con=None, inst_id="inst11")
    db_manager.activate_unit_tests_mode()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    with pytest.raises(ValueError):
        f_manager.assign_dbmanager(db_manager)

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, _ = db_manager.get_dataset()

    f_manager.assign_dbmanager(db_manager)
    assert f_manager.inst_id == "inst11"
    assert f_manager.inst_type == "load"
    for rtype in ["P", "Q"]:
        assert rtype in f_manager.model_ref
        assert rtype in f_manager.model_in_use
        assert f_manager.model_ref[rtype] != {}
        assert f_manager.model_in_use[rtype] != {}


def test_db_manager_train_mode():
    """
    Assures that the correct mode "train" is set before training methods are
    invoked.

    Scenario:
    - "forecast" mode is triggered

    Expected:
    - Assertion errors when invoking training methods due to incorrect mode
    being set.

    """
    db_manager = DataManager(db_con=None, inst_id="inst11")
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    # Mode is set to "forecast" to force Exception
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, _ = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    with pytest.raises(AssertionError):
        f_manager.train_active_power_models(save=False)
    with pytest.raises(AssertionError):
        f_manager.train_reactive_power_models(save=True)


def test_load_train_available_data(tmp_path_factory):
    """
    Check if models are correctly trained when data is available.

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Existence of model references
    - Backup models reference
    - Locally saved files (models, scalers, ...) are indeed generated.

    """
    inst_id = "inst11"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, _ = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    for rtype, models in zip(['P', 'Q'], [modelsP, modelsQ]):
        assert len(models) > 0
        assert "backup_weather" in models
        assert "backup_no_weather" in models
        for model in models:
            model_ = '_'.join(model.split('/'))
            path = os.path.join(folder_models, "models", inst_id, rtype,
                                model_, models[model])
            for file in ['inputs.txt', 'model.gz']:
                assert os.path.exists(os.path.join(path, file))
            if models[model] == 'lqr':
                assert os.path.exists(os.path.join(path, "x_scaler.gz"))
                assert os.path.exists(os.path.join(path, "y_scaler.gz"))


def test_wind_train_available_data(tmp_path_factory):
    """
    Check if models are correctly trained when data is available for a
    wind power generation installation.

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Existence of model references
    - Locally saved files (models, scalers, ...) are indeed generated.

    """
    inst_id = "inst21"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, _ = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    for rtype, models in zip(['P', 'Q'], [modelsP, modelsQ]):
        assert len(models) > 0

        assert "weather" in models
        assert "no_weather" in models
        assert set(models.values()) == {"gbt_wind"}
        for model in models:
            model_ = '_'.join(model.split('/'))
            path = os.path.join(folder_models, "models", inst_id, rtype,
                                model_, models[model])
            for file in ['inputs.txt', 'model.gz']:
                assert os.path.exists(os.path.join(path, file))


def test_solar_train_available_data(tmp_path_factory):
    """
    Check if models are correctly trained when data is available for a
    PV generation installation.

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Existence of model references
    - Locally saved files (models, scalers, ...) are indeed generated.

    """
    inst_id = "inst31"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, _ = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    for rtype, models in zip(['P', 'Q'], [modelsP, modelsQ]):
        assert len(models) > 0

        assert "weather" in models
        assert "no_weather" in models
        assert set(models.values()) == {"gbt_solar"}
        for model in models:
            model_ = '_'.join(model.split('/'))
            path = os.path.join(folder_models, "models", inst_id, rtype,
                                model_, models[model])
            for file in ['inputs.txt', 'model.gz']:
                assert os.path.exists(os.path.join(path, file))


def test_load_train_short_data(tmp_path_factory):
    """
    Check if models are correctly trained when measurements data is short.

    Scenario:
    - Short measurement history + Full NWP data available

    Expected:
    - Existence of model references
    - Backup models reference
    - Locally saved files (models, scalers, ...) are indeed generated.
    - All models were trained with algorithm referenced as "lqr"

    """
    inst_id = "inst12"
    folder_models = tmp_path_factory.getbasetemp() / "short_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    for rtype, models in zip(['P', 'Q'], [modelsP, modelsQ]):
        assert len(models) > 0
        assert "backup_weather" in models
        assert "backup_no_weather" in models
        assert set(models.values()) == {"lqr"}
        for model in [m for m in models if models[m] != "not_available"]:
            model_ = '_'.join(model.split('/'))
            path = os.path.join(folder_models, "models", inst_id, rtype,
                                model_, models[model])
            for file in ['inputs.txt', 'model.gz']:
                assert os.path.exists(os.path.join(path, file))
            if models[model] == 'lqr':
                if model != "backup_no_weather":
                    assert os.path.exists(os.path.join(path, "x_scaler.gz"))
                assert os.path.exists(os.path.join(path, "y_scaler.gz"))


def test_wind_train_short_data(tmp_path_factory):
    """
    Check if models are correctly trained when measurements data is short, for
    a wind power generation installation.

    Scenario:
    - Short measurement history + Full NWP data available

    Expected:
    - Existence of model references
    - Backup models reference
    - Locally saved files (models, scalers, ...) are indeed generated.

    """
    inst_id = "inst23"
    folder_models = tmp_path_factory.getbasetemp() / "short_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    for rtype, models in zip(['P', 'Q'], [modelsP, modelsQ]):
        assert len(models) > 0
        assert set(models.values()) == {"gbt_wind"}
        for model in [m for m in models if models[m] != "not_available"]:
            model_ = '_'.join(model.split('/'))
            path = os.path.join(folder_models, "models", inst_id, rtype,
                                model_, models[model])
            for file in ['inputs.txt', 'model.gz']:
                assert os.path.exists(os.path.join(path, file))


def test_solar_train_short_data(tmp_path_factory):
    """
    Check if models are correctly trained when measurements data is short,
    for a PV generation installation.

    Scenario:
    - Short measurement history + Full NWP data available

    Expected:
    - Existence of model references
    - Backup models reference
    - Locally saved files (models, scalers, ...) are indeed generated.

    """
    inst_id = "inst33"
    folder_models = tmp_path_factory.getbasetemp() / "short_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    for rtype, models in zip(['P', 'Q'], [modelsP, modelsQ]):
        assert len(models) > 0
        assert set(models.values()) == {"gbt_solar"}
        for model in [m for m in models if models[m] != "not_available"]:
            model_ = '_'.join(model.split('/'))
            path = os.path.join(folder_models, "models", inst_id, rtype,
                                model_, models[model])
            for file in ['inputs.txt', 'model.gz']:
                assert os.path.exists(os.path.join(path, file))


def test_load_train_only_q(tmp_path_factory):
    """
    Check if reactive models are properly trained without previous active
    power training.

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Existence of model references
    - Backup models reference
    - Locally saved files (models, scalers, ...) are indeed generated.
    - Models with reference "forecast" included are listed with "not_available"

    """
    inst_id = "inst11"
    folder_models = tmp_path_factory.getbasetemp() / "only_Q"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    assert len(modelsQ) > 0
    assert "backup_weather" in modelsQ
    assert "backup_no_weather" in modelsQ

    keys_with_forecast = [k for k in modelsQ.keys() if "forecast" in k]
    for k in keys_with_forecast:
        assert modelsQ[k] == 'not_available'

    for model in [m for m in modelsQ if modelsQ[m] != "not_available"]:
        model_ = '_'.join(model.split('/'))
        path = os.path.join(folder_models, "models", inst_id, "Q", model_,
                            modelsQ[model])
        for file in ['inputs.txt', 'model.gz']:
            assert os.path.exists(os.path.join(path, file))
        if modelsQ[model] == 'lqr':
            if model != "backup_no_weather":
                assert os.path.exists(os.path.join(path, "x_scaler.gz"))
            assert os.path.exists(os.path.join(path, "y_scaler.gz"))


def test_wind_train_only_q(tmp_path_factory):
    """
    Check if reactive models are properly trained without previous active
    power training, for a wind power generation installation.

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Existence of model references
    - Backup models reference
    - Locally saved files (models, scalers, ...) are indeed generated.
    - Models with reference "forecast" included are listed with "not_available"

    """
    inst_id = "inst21"
    folder_models = tmp_path_factory.getbasetemp() / "only_Q"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    assert len(modelsQ) > 0
    assert "weather" in modelsQ
    assert "no_weather" in modelsQ

    keys_with_forecast = [k for k in modelsQ.keys() if "forecast" in k]
    for k in keys_with_forecast:
        assert modelsQ[k] == 'not_available'

    for model in [m for m in modelsQ if modelsQ[m] != "not_available"]:
        model_ = '_'.join(model.split('/'))
        path = os.path.join(folder_models, "models", inst_id, "Q", model_,
                            modelsQ[model])
        for file in ['inputs.txt', 'model.gz']:
            assert os.path.exists(os.path.join(path, file))


def test_solar_train_only_q(tmp_path_factory):
    """
    Check if reactive models are properly trained without previous active
    power training, for a PV generation installation.

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Existence of model references
    - Backup models reference
    - Locally saved files (models, scalers, ...) are indeed generated.
    - Models with reference "forecast" included are listed with "not_available"

    """
    inst_id = "inst31"
    folder_models = tmp_path_factory.getbasetemp() / "only_Q"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    assert len(modelsQ) > 0
    assert "weather" in modelsQ
    assert "no_weather" in modelsQ

    keys_with_forecast = [k for k in modelsQ.keys() if "forecast" in k]
    for k in keys_with_forecast:
        assert modelsQ[k] == 'not_available'

    for model in [m for m in modelsQ if modelsQ[m] != "not_available"]:
        model_ = '_'.join(model.split('/'))
        path = os.path.join(folder_models, "models", inst_id, "Q", model_,
                            modelsQ[model])
        for file in ['inputs.txt', 'model.gz']:
            assert os.path.exists(os.path.join(path, file))


def test_load_train_no_nwp(tmp_path_factory):
    """
    Check training of models without NWP data. Almost all should become not
    available.

    Scenario:
    - Full measurements + No NWP data available

    Expected:
    - Existence of model references
    - Locally saved files (models, scalers, ...) are indeed generated.
    - All models except "D-7/forecast" and "backup_no_weather"
        are listed with "not_available"

    """
    inst_id = "inst16"
    folder_models = tmp_path_factory.getbasetemp() / "no_nwp"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, _ = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    expected_model_refs = {
        'P': {'D-7/no_weather': 'lqr', 'backup_no_weather': 'lqr',
              'D-1/D-7': 'not_available', 'D-2/D-7': 'not_available',
              'D-3/D-7': 'not_available', 'D-7': 'not_available',
              'backup_weather': 'not_available'},
        'Q': {'D-7/no_weather': 'lqr', 'backup_no_weather': 'lqr',
              'D-7': 'not_available', 'D-7/forecast': 'not_available',
              'backup_weather': 'not_available'}
    }

    for rtype, models in zip(["P", "Q"], [modelsP, modelsQ]):
        assert len(models) > 0
        assert models == expected_model_refs[rtype]

        for model in [m for m in models if models[m] != "not_available"]:
            model_ = '_'.join(model.split('/'))
            path = os.path.join(folder_models, "models", inst_id, rtype,
                                model_, models[model])
            for file in ['inputs.txt', 'model.gz']:
                assert os.path.exists(os.path.join(path, file))
            if models[model] == 'lqr':
                if model != "backup_no_weather":
                    assert os.path.exists(os.path.join(path, "x_scaler.gz"))
                assert os.path.exists(os.path.join(path, "y_scaler.gz"))


# -- Forecasting


def test_db_manager_forecast_mode(tmp_path_factory):
    """
    Assures that the correct mode "train" is set before training methods are
    invoked.

    Scenario:
    - "train" mode is triggered

    Expected:
    - Assertion errors when invoking training methods due to incorrect mode
    being set.

    """
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id="inst11")
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    # Wrong DataManager mode
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    for rtype in model_config:
        with pytest.raises(AssertionError):
            f_manager.forecast(
                trained_models={},
                trained_scalers={},
                trained_inputs={},
                model_configs=model_config,
                register_type=rtype
            )


def test_load_forecast_available_data(tmp_path_factory):
    """
    Check if predictions are correctly computed when data is available

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Predictions for active and reactive cases are generated
    - Expected horizons have predictions
    - Expected model references are listed

    """
    inst_id = "inst11"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(
            register_type=rtype)
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    assert set(predictions_dict.keys()) == {'P', 'Q'}
    assert set(predictions_dict['P']['model_info'].unique()) == {'D-1/D-7',
                                                                 'D-2/D-7',
                                                                 'D-3/D-7',
                                                                 'D-7'}
    assert set(predictions_dict['Q']['model_info'].unique()) == {'D-7/forecast'} # noqa
    for rtype in ['P', 'Q']:
        assert predictions_dict[rtype] is not None
        assert not predictions_dict[rtype].empty
        assert not predictions_dict[rtype].isnull().any().any()
        assert set(predictions_dict[rtype]['horizon'].unique()) == {'D',
                                                                    'D+1',
                                                                    'D+2',
                                                                    'D+3',
                                                                    'D+4',
                                                                    'D+5',
                                                                    'D+6'}


def test_load_forecast_holiday(tmp_path_factory):
    """
    Check if holiday/bridge predictions are correctly computed.

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Predictions for active and reactive cases are generated
    - Expected horizons have predictions
    - Expected model references are listed (model with "+holiday" added)

    """
    inst_id = "inst11"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-12-20 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(
            register_type=rtype)
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    assert set(predictions_dict.keys()) == {'P', 'Q'}
    assert set(predictions_dict['P']['model_info'].unique()) == {'D-1/D-7',
                                                                 'D-2/D-7',
                                                                 'D-3/D-7',
                                                                 'D-7',
                                                                 'D-7+holiday'} # noqa
    assert set(predictions_dict['Q']['model_info'].unique()) == {'D-7/forecast', # noqa
                                                                 'D-7/forecast+holiday'} # noqa
    for rtype in ['P', 'Q']:
        assert predictions_dict[rtype] is not None
        assert not predictions_dict[rtype].empty
        assert not predictions_dict[rtype].isnull().any().any()
        assert set(predictions_dict[rtype]['horizon'].unique()) == {'D',
                                                                    'D+1',
                                                                    'D+2',
                                                                    'D+3',
                                                                    'D+4',
                                                                    'D+5',
                                                                    'D+6'}


def test_wind_forecast_available_data(tmp_path_factory):
    """
    Check if predictions are correctly computed when data is available,
    for a wind power generation installation

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Predictions for active and reactive cases are generated
    - Horizons from D to D+6 in predictions
    - Appropriate model references are listed

    """
    inst_id = "inst21"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(
            register_type=rtype)
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    assert set(predictions_dict.keys()) == {'P', 'Q'}
    assert set(predictions_dict['P']['model_info'].unique()) == {'weather'}
    assert set(predictions_dict['Q']['model_info'].unique()) == {'weather/forecast'} # noqa
    for rtype in ['P', 'Q']:
        assert predictions_dict[rtype] is not None
        assert not predictions_dict[rtype].empty
        assert not predictions_dict[rtype].isnull().any().any()
        assert set(predictions_dict[rtype]['horizon'].unique()) == {'D+X'}


def test_wind_forecast_holiday(tmp_path_factory):
    """
    Check if no holiday processing is performed on the predictions.

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Predictions for active and reactive cases are generated
    - Horizons from D to D+6 in predictions
    - No model reference with holiday

    """
    inst_id = "inst21"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-06-07 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(
            register_type=rtype)
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    assert set(predictions_dict.keys()) == {'P', 'Q'}
    assert set(predictions_dict['P']['model_info'].unique()) == {'weather'}
    assert set(predictions_dict['Q']['model_info'].unique()) == {'weather/forecast'} # noqa
    for rtype in ['P', 'Q']:
        assert predictions_dict[rtype] is not None
        assert not predictions_dict[rtype].empty
        assert not predictions_dict[rtype].isnull().any().any()
        assert set(predictions_dict[rtype]['horizon'].unique()) == {'D+X'}


def test_solar_forecast_available_data(tmp_path_factory):
    """
    Check if predictions are correctly computed when data is available,
    for a PV power generation installation

    Scenario:
    - Full measurements and NWP data available

    Expected:
    - Predictions for active and reactive cases are generated
    - Horizons from D to D+6 in predictions
    - Expected model references are listed

    """
    inst_id = "inst21"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(
            register_type=rtype)
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    assert set(predictions_dict.keys()) == {'P', 'Q'}
    assert set(predictions_dict['P']['model_info'].unique()) == {'weather'}
    assert set(predictions_dict['Q']['model_info'].unique()) == {'weather/forecast'} # noqa
    for rtype in ['P', 'Q']:
        assert predictions_dict[rtype] is not None
        assert not predictions_dict[rtype].empty
        assert not predictions_dict[rtype].isnull().any().any()
        assert set(predictions_dict[rtype]['horizon'].unique()) == {'D+X'}


def test_load_forecast_short_data(tmp_path_factory):
    """
    Check if predictions are correctly computed when data is short

    Scenario:
    - Short measurement history + Full NWP data available

    Expected:
    - Predictions for active and reactive cases are generated
    - Horizons from D to D+6 in predictions
    - Appropriate model references are listed

    """
    inst_id = "inst12"
    folder_models = tmp_path_factory.getbasetemp() / "short_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(
            register_type=rtype)
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    assert set(predictions_dict.keys()) == {'P', 'Q'}
    assert set(predictions_dict['P']['model_info'].unique()) == {'D-1/D-7',
                                                                 'D-2/D-7',
                                                                 'D-3/D-7',
                                                                 'D-7',
                                                                 'D-7/no_weather'} # noqa
    assert set(predictions_dict['Q']['model_info'].unique()) == {'D-7/forecast', # noqa
                                                                 'D-7/no_weather'} # noqa
    for rtype in ['P', 'Q']:
        assert predictions_dict[rtype] is not None
        assert not predictions_dict[rtype].empty
        assert not predictions_dict[rtype].isnull().any().any()
        assert set(predictions_dict[rtype]['horizon'].unique()) == {'D',
                                                                    'D+1',
                                                                    'D+2',
                                                                    'D+3',
                                                                    'D+4',
                                                                    'D+5',
                                                                    'D+6'}


def test_load_train_backup_mix(tmp_path_factory):
    """
    Dummy test to train models for specific installation that triggers
    "backup_mix" models.

    Scenario:
    - Last month historical measurements unavailable
             and last 11h hours in horizon missing
      + NWP data available

    Expected:
    - Existence of model references
    - Backup models reference
    - Locally saved files (models, scalers, ...) are indeed generated.

    """
    inst_id = "inst17"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, _ = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    for rtype, models in zip(['P', 'Q'], [modelsP, modelsQ]):
        assert len(models) > 0
        assert "backup_weather" in models
        assert "backup_no_weather" in models
        for model in models:
            model_ = '_'.join(model.split('/'))
            path = os.path.join(folder_models, "models", inst_id, rtype,
                                model_, models[model])
            for file in ['inputs.txt', 'model.gz']:
                assert os.path.exists(os.path.join(path, file))
            if models[model] == 'lqr':
                assert os.path.exists(os.path.join(path, "x_scaler.gz"))
                assert os.path.exists(os.path.join(path, "y_scaler.gz"))


def test_load_forecast_backup_mix(tmp_path_factory):
    """
    Check if predictions are correctly computed when backup_mix models are
    activated.

    Scenario:
    - Last month historical measurements unavailable
             and last 11h hours in horizon missing
      + NWP data available

    Expected:
    - Predictions for active and reactive cases are generated
    - Expected horizons have predictions
    - Expected model references are listed ("backup_mix" and "backup_weather")

    """
    inst_id = "inst17"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(
            register_type=rtype)
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    assert set(predictions_dict.keys()) == {'P', 'Q'}
    for rtype in ['P', 'Q']:
        assert set(predictions_dict[rtype]['model_info'].unique()) == {'backup_mix', 'backup_weather'} # noqa
        assert predictions_dict[rtype] is not None
        assert not predictions_dict[rtype].empty
        assert not predictions_dict[rtype].isnull().any().any()
        assert set(predictions_dict[rtype]['horizon'].unique()) == {'D',
                                                                    'D+1',
                                                                    'D+2',
                                                                    'D+3',
                                                                    'D+4',
                                                                    'D+5',
                                                                    'D+6'}


def test_wind_train_backup_mix(tmp_path_factory):
    """
    Dummy test to have models for specific installation that triggers
    "mix" models for a wind power generation installation.

    Scenario:
    - Full measurements historical
      + NWP data available but last 11h hours in horizon missing

    Expected:
    - Existence of model references
    - Locally saved files (models, scalers, ...) are indeed generated.

    """
    inst_id = "inst22"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, _ = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    for rtype, models in zip(['P', 'Q'], [modelsP, modelsQ]):
        assert len(models) > 0

        assert "weather" in models
        assert "no_weather" in models
        assert set(models.values()) == {"gbt_wind"}
        for model in models:
            model_ = '_'.join(model.split('/'))
            path = os.path.join(folder_models, "models", inst_id, rtype,
                                model_, models[model])
            for file in ['inputs.txt', 'model.gz']:
                assert os.path.exists(os.path.join(path, file))


def test_wind_forecast_backup_mix(tmp_path_factory):
    """
    Check if predictions are correctly computed when "mix" models are
    activated, for a wind power generation installation.

    Scenario:
    - Full measurements historical
      + NWP data available but last 11h hours in horizon missing

    Expected:
    - Predictions for active and reactive cases are generated
    - Expected horizons have predictions
    - Expected model references are listed ("mix")

    """
    inst_id = "inst22"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(
            register_type=rtype)
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    assert set(predictions_dict.keys()) == {'P', 'Q'}
    for rtype in ['P', 'Q']:
        assert set(predictions_dict[rtype]['model_info'].unique()) == {'mix'} # noqa
        assert predictions_dict[rtype] is not None
        assert not predictions_dict[rtype].empty
        assert not predictions_dict[rtype].isnull().any().any()
        assert set(predictions_dict[rtype]['horizon'].unique()) == {'D+X'}


def test_solar_train_backup_mix(tmp_path_factory):
    """
    Dummy test to have models for specific installation that triggers
    "mix" models for a PV power generation installation.

    Scenario:
    - Full measurements historical
      + NWP data available but last 11h hours in horizon missing

    Expected:
    - Existence of model references
    - Locally saved files (models, scalers, ...) are indeed generated.

    """
    inst_id = "inst32"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, _ = db_manager.get_dataset()

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)

    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    for rtype, models in zip(['P', 'Q'], [modelsP, modelsQ]):
        assert len(models) > 0

        assert "weather" in models
        assert "no_weather" in models
        assert set(models.values()) == {"gbt_solar"}
        for model in models:
            model_ = '_'.join(model.split('/'))
            path = os.path.join(folder_models, "models", inst_id, rtype,
                                model_, models[model])
            for file in ['inputs.txt', 'model.gz']:
                assert os.path.exists(os.path.join(path, file))


def test_solar_forecast_backup_mix(tmp_path_factory):
    """
    Check if predictions are correctly computed when backup_mix models are
    activated, for a PV power generation installation.

    Scenario:
    - Full measurements historical
      + NWP data available but last 11h hours in horizon missing

    Expected:
    - Predictions for active and reactive cases are generated
    - Expected horizons have predictions
    - Expected model references are listed ("mix")

    """
    inst_id = "inst32"
    folder_models = tmp_path_factory.getbasetemp() / "avail_data"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(
            register_type=rtype)
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    assert set(predictions_dict.keys()) == {'P', 'Q'}
    assert set(predictions_dict['P']['model_info'].unique()) == {'mix'}
    assert set(predictions_dict['Q']['model_info'].unique()) == {'mix'} # noqa
    for rtype in ['P', 'Q']:
        assert predictions_dict[rtype] is not None
        assert not predictions_dict[rtype].empty
        assert not predictions_dict[rtype].isnull().any().any()
        assert set(predictions_dict[rtype]['horizon'].unique()) == {'D+X'}


def test_load_forecast_no_nwp(tmp_path_factory):
    """
    Check if predictions are correctly computed when NWP data is not available.

    Scenario:
    - Full measurements + No NWP data available

    Expected:
    - Predictions for active and reactive cases are generated
    - Expected horizons have predictions
    - Expected model references are listed (only "D-7/weather")

    """

    inst_id = "inst16"
    folder_models = tmp_path_factory.getbasetemp() / "no_nwp"
    db_manager = DataManager(db_con=None, inst_id=inst_id)
    db_manager.activate_unit_tests_mode()

    launch_time = "2020-01-15 00:00:00"
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    f_manager = ForecastManager()
    f_manager.set_model_location("local")
    f_manager.activate_unit_tests_mode()
    f_manager.set_local_models_path(folder_models)
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)

    # -- Train models for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(
            register_type=rtype)
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    assert set(predictions_dict.keys()) == {'P', 'Q'}
    for rtype in ['P', 'Q']:
        assert set(predictions_dict[rtype]['model_info'].unique()) == {'D-7/no_weather'} # noqa
        assert predictions_dict[rtype] is not None
        assert not predictions_dict[rtype].empty
        assert not predictions_dict[rtype].isnull().any().any()
        assert set(predictions_dict[rtype]['horizon'].unique()) == {'D',
                                                                    'D+1',
                                                                    'D+2',
                                                                    'D+3',
                                                                    'D+4',
                                                                    'D+5',
                                                                    'D+6'}
