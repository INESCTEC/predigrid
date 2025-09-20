import datetime as dt

from time import time
from configs import SystemConfig
from configs.logs.Log import MyLoggerAdapter
from joblib import Parallel, delayed

from src.database.CassandraDB import CassandraDB
from src.preprocessed_data.helpers import clean_netload_data


def process_netload_data(config: SystemConfig,
                         conn: CassandraDB,
                         logger: MyLoggerAdapter,
                         start_date: dt.datetime,
                         end_date: dt.datetime,
                         save_to_csv:bool = False) -> None:
    """
    Main routine to perform netload processing.

    For each register type (P or Q), this method takes the time series
    with netload data and performs the following tasks:

    - Remove periods longer than 24 hours from the time series
    - Estimate (by means of interpolation) short periods with not data (up to 3 hours)
    - Identify and remove outliers from the data series

    The processed set of data is then inserted in the database.

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
    # -- Date Range to Cleanse:
    netload_table = config.TABLES["netload"]
    processed_table = config.TABLES["processed"]
    start_date = start_date.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    end_date = end_date.strftime("%Y-%m-%d %H:%M:%S")

    # -- Initialize Loggers:
    logger.info("-" * 70)
    logger.info("-" * 70)
    logger.info(f"CLEANSING -- Period: {start_date}-{end_date}")

    # -- Clean Data:
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

        for register_type in ['P', 'Q']:
            logger.info("-" * 70)
            logger.info(f"PT Type: {pt_type}")
            logger.info(f"No. of series to insert: {len(pts)}")

            # -- Timer and counters:
            process_time = time()

            n_processed = Parallel(n_jobs=config.N_JOBS,
                                   backend="threading")(
                delayed(clean_netload_data)(
                    logger,
                    conn,
                    netload_table,
                    processed_table,
                    pt_id,
                    pt_type,
                    start_date,
                    end_date,
                    register_type,
                    save_to_csv
                )
                for pt_id in pts)

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

