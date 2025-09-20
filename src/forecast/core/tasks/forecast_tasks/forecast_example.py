import pandas as pd

from core.database.DataManager import DataManager
from core.forecast.ForecastManager import ForecastManager


def run_forecast(db_con, inst_id, launch_time):
    #####################################
    #           Data Manager            #
    #####################################
    db_manager = DataManager(
        db_con=db_con,
        inst_id=inst_id,
    )
    # -- Correct conditions expected:
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=48)
    db_manager.set_mode(mode="forecast")
    # -- Fetch dataset (measurements + nwp):
    dataset, model_config = db_manager.get_dataset()

    #####################################
    #         Forecast Manager          #
    #####################################
    f_manager = ForecastManager()
    f_manager.set_model_location("database")
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    dataset_stats = f_manager.load_dataset_stats()
    f_manager.assign_stats(stats=dataset_stats)

    # -- Create forecasts for active & reactive power:
    predictions_dict = {}
    for rtype in model_config:
        trained_models, trained_inputs = f_manager.load_forecast_models(register_type=rtype) # noqa
        trained_scalers = f_manager.load_forecast_scalers(register_type=rtype)
        predictions = f_manager.forecast(
            trained_models=trained_models,
            trained_scalers=trained_scalers,
            trained_inputs=trained_inputs,
            model_configs=model_config,
            register_type=rtype
        )
        predictions_dict[rtype] = predictions

    predictions_df = f_manager.prepare_final_output(predictions_dict)
    print(predictions_df)
    f_manager.insert_in_database(conn=db_con, predictions_df=predictions_df)


if __name__ == '__main__':
    import datetime as dt
    from forecast_api.util.databases import CassandraDB
    CASSANDRA_DB_HOST = "CASSANDRA_DB_HOST"
    CASSANDRA_DB_KEYSPACE = "CASSANDRA_DB_KEYSPACE"
    CASSANDRA_DB_PORT = 0
    db_con = CassandraDB(host=CASSANDRA_DB_HOST, keyspace=CASSANDRA_DB_KEYSPACE, port=CASSANDRA_DB_PORT)  # noqa

    # Query to get available Pt's in the DB
    query = "select * from installations;"
    installations = db_con.read_query(query=query)
    installations = list(installations["id"].values.ravel())
    # init launch times:
    launch_time = pd.to_datetime(dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")) # noqa
    launch_time = launch_time.tz_localize("UTC").tz_convert("Europe/Lisbon")
    print("working on launch_time {}".format(launch_time))
    # for each inst - create train obj:
    for pt_counter, inst_id in enumerate(installations):
        try:
            print("Client {} out of {}".format(pt_counter, len(installations)))
            run_forecast(
                db_con=db_con,
                inst_id=inst_id,
                launch_time=launch_time,
            )
        except Exception as ex:
            print(inst_id, repr(ex))
    db_con.disconnect()
