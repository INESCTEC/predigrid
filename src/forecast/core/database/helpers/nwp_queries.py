import numpy as np
import pandas as pd

from cassandra.connection import ConnectionException


def fetch_from_database(conn, source_nwp, table, latitude, longitude,
                        variables, date_start_utc, date_end_utc):
    query = f"select datetime, request, {variables} from {table} " \
            f"where source='{source_nwp}' " \
            f"and latitude={latitude} and longitude={longitude} " \
            f"and datetime >= '{date_start_utc}' " \
            f"and datetime <= '{date_end_utc}';"
    try:
        df = conn.read_query(query=query)
    except AttributeError as ex:
        raise ConnectionException(repr(ex))
    return df


def query_nwp_data(conn,
                   source_nwp,
                   table,
                   latitude,
                   longitude,
                   variables,
                   date_start_utc,
                   date_end_utc,
                   use_mock_data,
                   ):
    """
    Method used to retrieve NWP variables from the database.

    :return:
    """
    # -- Prepare container (dataset) for expected historical data
    nwp_dataset = pd.DataFrame(
        index=pd.date_range(start=date_start_utc,
                            end=date_end_utc,
                            freq='H', tz='UTC')
    )
    # -- Fetch data:
    if use_mock_data:
        # If in unit tests mode, creates Mock Dataset
        from .mock.scripts.nwp import create_mock_nwp
        df = create_mock_nwp(
            latitude=latitude,
            longitude=longitude,
            variables=variables,
            date_start_utc=date_start_utc,
            date_end_utc=date_end_utc
        )
    else:
        # If in normal mode, fetches data from DB
        df = fetch_from_database(
            conn=conn,
            source_nwp=source_nwp,
            table=table,
            latitude=latitude,
            longitude=longitude,
            variables=','.join(variables),
            date_start_utc=date_start_utc,
            date_end_utc=date_end_utc
        )
    # -- Parse dates:
    df.loc[:, 'datetime'] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S").dt.tz_localize("UTC")  # noqa
    df.loc[:, 'request'] = pd.to_datetime(df["request"], format="%Y-%m-%d %H:%M:%S").dt.tz_localize("UTC")   # noqa
    # -- Organize data:
    df.sort_values(by=["datetime", "request"], ascending=True, inplace=True)
    df.drop_duplicates(subset=["datetime"], keep="last", inplace=True)
    df.set_index("datetime", inplace=True)
    df.drop("request", 1, inplace=True)

    # Join retrieved data with expected container:
    nwp_dataset = nwp_dataset.join(df, how="left")
    # fill small gaps of missing values with interpolation
    nwp_dataset.interpolate(method="linear", limit=6,
                            limit_direction="backward",
                            inplace=True)
    return nwp_dataset
