import pandas as pd


from core.database.DataManager import DataManager
from core.forecast.ForecastManager import ForecastManager


def run_train(db_con, inst_id, launch_time):

    #####################################
    #           Data Manager            #
    #####################################
    db_manager = DataManager(
        db_con=db_con,
        inst_id=inst_id,
    )
    # -- Correct conditions expected:
    db_manager.set_launch_time(launch_time=launch_time)
    db_manager.set_forecast_horizon(forecast_horizon=168)
    db_manager.set_mode(mode="train")
    # -- Fetch dataset (measurements + nwp):
    dataset, _ = db_manager.get_dataset()
    dataset_stats = db_manager.get_statistics(dataset=dataset)

    #####################################
    #         Forecast Manager          #
    #####################################
    f_manager = ForecastManager()
    # f_manager.set_model_location("local")
    # Assign data manager & dataset to forecast manager:
    f_manager.assign_dbmanager(db_manager=db_manager)
    f_manager.assign_dataset(dataset=dataset)
    f_manager.assign_stats(stats=dataset_stats)
    # -- Train models for active & reactive power:
    modelsP = f_manager.train_active_power_models(save=True)
    modelsQ = f_manager.train_reactive_power_models(save=True)

    # Update metadata in database (info. about models effectively trained)
    f_manager.upload_models_metadata({"P": modelsP, "Q": modelsQ})

    # Save dataset statistics for forecasting task
    f_manager.save_dataset_stats(stats=dataset_stats)


if __name__ == '__main__':
    from forecast_api.util.databases import CassandraDB
    CASSANDRA_DB_HOST = "CASSANDRA_DB_HOST"
    CASSANDRA_DB_KEYSPACE = "CASSANDRA_DB_KEYSPACE"
    CASSANDRA_DB_PORT = 0
    db_con = CassandraDB(host=CASSANDRA_DB_HOST, keyspace=CASSANDRA_DB_KEYSPACE, port=CASSANDRA_DB_PORT)  # noqa

    import os
    import datetime as dt
    # Query to get available Pt's in the DB
    query = "select * from installations;"
    installations = db_con.read_query(query=query)
    installations = list(installations["id"].values.ravel())
    # init launch times:
    launch_time = pd.to_datetime(dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")) # noqa
    launch_time = launch_time.tz_localize("Europe/Lisbon")
    print("working on launch_time {}".format(launch_time))
    # for each inst - create train obj:
    for pt_counter, inst_id in enumerate(installations):
        try:
            print("Client {} out of {}".format(pt_counter, len(installations)))
            # active power forecast
            run_train(
                db_con=db_con,
                inst_id=inst_id,
                launch_time=launch_time,
            )
        except Exception as ex:
            print(inst_id, repr(ex))
    db_con.disconnect()
