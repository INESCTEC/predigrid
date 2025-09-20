import datetime as dt

from time import time
from configs import SystemConfig
from joblib import Parallel, delayed
from configs.logs.Log import MyLoggerAdapter

from src.netload_data import (
    insert_active_power_netload,
    insert_reactive_power_netload
)
from src.database.CassandraDB import CassandraDB

# A+: Energia ativa consumida da rede: [número do registo]=1
# A-: Energia ativa injetada da rede: [número do registo]=4
# Q: Energia indutiva consumida da rede: [número do registo]=2 + Energia indutiva injetada da rede: [número do registo]=5 - Energia capacitiva consumida da rede: [número do registo]=3 - Energia capacitiva injetada da rede: [número do registo]=6. Em modo operacional são estas três variáveis que vamos prever. Alguns ficheiros estão vazios, por isso consideras 0.


def calculate_netload(config: SystemConfig,
                      conn: CassandraDB,
                      logger: MyLoggerAdapter,
                      start_date: dt.datetime,
                      end_date: dt.datetime,
                      save_to_csv: bool = False) -> None:
    """
    Main routine to perform netload computation.

    For active power, the netload corresponds to the subtraction of values for
    power types "A+" and "A-"
    For reactive power, the netload corresponds to adding values for
    power types "Qi+" and "Qi-" and substracting values
    for power types "Qc+" and "Qc-"

    Finally, new computed series is stored in a database.

    :param config: System configuration
    :type config: SystemConfig
    :param conn: Database connection
    :type conn: CassandraDB
    :param logger: Logger object for producing system messages
    :type logger: Log
    :param start_date: Starting date for computing data
    :type start_date: dt.datetime
    :param end_date: Ending date for computing data
    :type end_date: dt.datetime
    :param save_to_csv: Flag for saving local copies of data
    :type save_to_csv: bool
    """
    raw_table = config.TABLES["raw"]
    netload_table = config.TABLES[f"netload"]
    # -- Date Range to Parse (reset start date seconds to avoid inserting
    # wrong avg data to db):
    start_date = start_date.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")  # noqa
    end_date = end_date.strftime("%Y-%m-%d %H:%M:%S")

    # -- Initialize Loggers:
    logger.info("-" * 70)
    logger.info("-" * 70)
    logger.info(f"NETLOAD -- Period: {start_date}-{end_date}")

    for pt_type in ["load", "solar", "wind"]:
        # check clients for each pt_type:
        pt_type = pt_type.lower()
        available_pts_query = f"SELECT id as pt_id " \
                              f"from {config.TABLES['metadata']} " \
                              f"where id_type='{pt_type}' " \
                              f"allow filtering;"
        available_pts = conn.read_query(available_pts_query)

        if available_pts.empty:
            logger.error(f"No id's returned for {pt_type} pt_type.")
            continue

        pts = list(available_pts["pt_id"])

        logger.info("-" * 70)
        logger.info("Processing active power ...")
        logger.info(f"PT Type: {pt_type}")
        logger.info(f"No. of series to insert: {len(pts)}")

        # -- Timer and counters:
        process_time = time()

        # -- Calculate and Insert Netload Active Power Data:
        n_processed = Parallel(n_jobs=config.N_JOBS,
                               backend="threading")(
            delayed(insert_active_power_netload)(
                logger,
                conn,
                raw_table,
                netload_table,
                pt_id,
                pt_type,
                start_date,
                end_date,
                save_to_csv
            ) for pt_id in pts)

        try:
            msg = f"Finished inserting data for " \
                  f"{sum(n_processed)}/{len(pts)} installations."
            if sum(n_processed) != len(pts):
                logger.warning(msg=msg)
            else:
                logger.info(msg=msg)
        except BaseException as ex:
            logger.exception(msg=repr(ex))

        logger.info(msg=f"Elapsed time: {(time() - process_time):.2f}s.")
        logger.info("Processing active power ... Ok!")

        logger.info("-" * 70)
        logger.info("Processing reactive power ...")
        logger.info(f"PT Type: {pt_type}")
        logger.info(f"No. of series to insert: {len(pts)}")

        # -- Timer and counters:
        process_time = time()

        # -- Calculate and Insert Netload Reactive Power Data:
        n_processed = Parallel(n_jobs=config.N_JOBS,
                               backend="threading")(
            delayed(insert_reactive_power_netload)(
                logger,
                conn,
                raw_table,
                netload_table,
                pt_id,
                pt_type,
                start_date,
                end_date,
                save_to_csv
            ) for pt_id in pts)

        try:
            msg = f"Finished inserting data for " \
                  f"{sum(n_processed)}/{len(pts)} installations."
            if sum(n_processed) != len(pts):
                logger.warning(msg=msg)
            else:
                logger.info(msg=msg)
        except BaseException as ex:
            logger.exception(msg=repr(ex))

        logger.info(msg=f"Elapsed time: {(time() - process_time):.2f}s.")

