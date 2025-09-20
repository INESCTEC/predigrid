import numpy as np
import datetime as dt

from time import sleep

from src.preprocessed_data.util.DataCleansing import DataCleansing


def clean_netload_data(logger,
                       cass_con,
                       netload_table,
                       processed_table,
                       pt_id,
                       pt_type,
                       start_date,
                       end_date,
                       register_type,
                       to_csv):

    try:
        # -- Query data from Cassandra DB:
        query = f"SELECT datetime, value, units " \
                f"from {netload_table} " \
                f"where id='{pt_id}' " \
                f"and register_type='{register_type}' " \
                f"and datetime >= '{start_date}' " \
                f"and datetime <= '{end_date}';"

        df = cass_con.read_query(query)
        df.set_index("datetime", inplace=True)

        if not df.empty:
            # -- Data Cleansing (outliers removal/replacement):
            cc = DataCleansing(freq=60)
            cc.load_raw_dataset(raw_data=df["value"])
            cc.remove_large_missing_periods(hour_threshold=24)
            cc.remove_outliers()
            df_clean = cc.clean_data.to_frame()

            if not df_clean.empty:
                df_clean.loc[np.isin(df_clean.index, cc.estimated_indexes), "estimated"] = 1
                df_clean.loc[:, "estimated"].fillna(0, inplace=True)

                # -- Replace old "value" column by new:
                df_clean.dropna(inplace=True)
                df = df.drop(["value"], 1).join(df_clean, how="outer")
                df['last_updated'] = dt.datetime.utcnow()
                df["id"] = pt_id
                df["register_type"] = register_type
                df.dropna(inplace=True)
                df.reset_index(drop=False, inplace=True)
                df = df[["id", "register_type", "datetime", "estimated",
                         "last_updated", "units", "value"]]

                result = False
                while result == False:
                    try:
                        if to_csv:
                            df.to_csv(f"data/{pt_type}_processed_60/{register_type}_{pt_id}.csv", index=False)
                        else:
                            cass_con.insert_query(df=df,
                                                  table=processed_table,
                                                  exec_async=False)
                        result = True
                    except Exception as e:
                        result = False
                        logger.exception(repr(e))
                        sleep(5)
                        pass

                return 1
        else:
            logger.warning("Skipped installation {} - No values.".format(pt_id))
            return 0
    except BaseException as e:
        logger.exception(msg="Failed for installation {}".format(pt_id))
        return 0
