import numpy as np
import pandas as pd

from cassandra.connection import ConnectionException


def fetch_from_database(conn, table, inst_id, register_type,
                        date_start_utc, date_end_utc):
    query = f"select datetime, value from {table} " \
            f"where id='{inst_id}' " \
            f"and register_type='{register_type}' " \
            f"and datetime >= '{date_start_utc}' " \
            f"and datetime <= '{date_end_utc}' " \
            f"order by datetime asc;"
    try:
        df = conn.read_query(query=query)
    except AttributeError as ex:
        raise ConnectionException(repr(ex))
    return df


def query_measurements(conn,
                       table,
                       inst_id,
                       register_type,
                       date_start_utc,
                       date_end_utc,
                       use_mock_data
                       ):
    """
    Method used to retrieve measurements data from the database.

    :return:
    """

    # Prepare container (dataset) for expected historical data
    measurements_df = pd.DataFrame(
        index=pd.date_range(start=date_start_utc,
                            end=date_end_utc,
                            freq='H', tz='UTC')
    )
    if use_mock_data:
        # If in unit tests mode, creates Mock Dataset
        from .mock.scripts.measurements import create_mock_measurements
        df = create_mock_measurements(
            inst_id=inst_id,
            date_start_utc=date_start_utc,
            date_end_utc=date_end_utc
        )
    else:
        # If in normal mode, fetches data from DB
        df = fetch_from_database(
            conn=conn,
            table=table,
            inst_id=inst_id,
            register_type=register_type,
            date_start_utc=date_start_utc,
            date_end_utc=date_end_utc
        )
    # Set DateTimeIndex:
    df.loc[:, 'datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S").dt.tz_localize("UTC")  # noqa
    df.set_index('datetime', inplace=True)
    # Rename target column:
    df.rename(columns={"value": f"real_{register_type}"}, inplace=True)
    # Join with expected DataFrame (missing ts = NaN):
    if not df.empty:
        measurements_df = measurements_df.join(df, how="left")
        # fill small gaps of missing values with interpolation
        measurements_df.interpolate(method="linear", limit=6,
                                    limit_direction="backward",
                                    inplace=True)
    else:
        measurements_df[f"real_{register_type}"] = np.nan

    return measurements_df


def query_measurements_last_date(conn, table, network_id, inst_id,
                                 register_type, launch_time_utc):
    query = f"select max(datetime) as max_datetime from {table} " \
            f"where network_id='{network_id}' " \
            f"and inst_id='{inst_id}' " \
            f"and register_type='{register_type}' " \
            f"and datetime <= '{launch_time_utc}';"
    try:
        df = conn.read_query(query=query)
    except AttributeError as ex:
        raise ConnectionException(repr(ex))
    return df["max_datetime"][0]
