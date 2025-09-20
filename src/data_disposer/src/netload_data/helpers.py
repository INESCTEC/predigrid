import datetime as dt

from time import sleep


def insert_active_power_netload(logger,
                                cass_con,
                                raw_table,
                                netload_table,
                                pt_id,
                                pt_type,
                                start_date,
                                end_date,
                                to_csv):

    try:
        # -- Active Power Data
        # Active power consumed from the grid
        query_Ac = f"SELECT datetime, value, " \
                   f"units " \
                   f"from {raw_table} " \
                   f"where id='{pt_id}' " \
                   f"and register_type='A+' " \
                   f"and datetime >= '{start_date}' " \
                   f"and datetime <= '{end_date}';"

        data_Ac = cass_con.read_query(query=query_Ac)
        data_Ac.set_index("datetime", inplace=True)
        data_Ac.sort_index(ascending=True, inplace=True)
        data_Ac.rename(columns={"value": "A+"}, inplace=True)
        # Convert all "W" records to "kW"
        data_Ac.loc[data_Ac["units"].str.lower() == "w", "A+"] /= 1000
        data_Ac.drop("units", axis=1, inplace=True)

        # Active Power Injected on the grid
        query_Ai = f"SELECT datetime, value, " \
                   f"units " \
                   f"from {raw_table} " \
                   f"where id='{pt_id}' " \
                   f"and register_type='A-' " \
                   f"and datetime >= '{start_date}' " \
                   f"and datetime <= '{end_date}';"

        data_Ai = cass_con.read_query(query=query_Ai)
        data_Ai.set_index("datetime", inplace=True)
        data_Ai.sort_index(ascending=True, inplace=True)
        data_Ai.rename(columns={"value": "A-"}, inplace=True)
        # Convert all "W" records to "kW"
        data_Ai.loc[data_Ai["units"].str.lower() == "w", "A-"] /= 1000
        data_Ai.drop("units", axis=1, inplace=True)

        # Active Power DataFrame:
        data_A = data_Ac.join(data_Ai, how="outer")
        data_A.dropna(how="all", inplace=True)  # drop rows with all elements NaN

        # If Active Power DataFrame is not empty, calculate netload:
        if not data_A.empty:
            if data_A["A+"].isnull().values.all():
                data_A["A+"].fillna(0, inplace=True)
            data_A["A-"].fillna(0, inplace=True)

            # -- Netload Formula (A+ Avg. power consumed - A- Avg power injected):
            data_A["value"] = data_A["A+"] - data_A["A-"]
            _dfr = data_A["value"].resample("60T").mean()
            _dfr = _dfr.to_frame()
            _dfr.loc[:, 'register_type'] = "P"
            _dfr.loc[:, 'id'] = pt_id
            _dfr.loc[:, 'units'] = "kW"
            _dfr.reset_index(inplace=True)
            _dfr.dropna(inplace=True)
            _dfr.loc[:, 'last_updated'] = dt.datetime.utcnow()
            df = _dfr[["id", "register_type", "datetime", "last_updated",
                       "units", "value"]]

            result = False
            while result == False:
                try:
                    if to_csv:
                        df.to_csv(f"data/mv_data_netload_60/{pt_type.lower()}_P_{pt_id}.csv", index=False)
                    else:
                        cass_con.insert_query(df=df, table=netload_table, exec_async=False)
                    result = True
                except Exception as e:
                    result = False
                    print(repr(e))
                    sleep(5)
                    pass

            return 1
        else:
            logger.warning(f"Skipped installation {pt_id} - No values.")
            return 0

    except BaseException as e:
        logger.exception(msg=f"Failed for pt {pt_id}.")
        return 0


def insert_reactive_power_netload(logger,
                                  cass_con,
                                  raw_table,
                                  netload_table,
                                  pt_id,
                                  pt_type,
                                  start_date,
                                  end_date,
                                  to_csv):

    try:
        # -------------------- Reactive Power --------------------------
        # -- Inductive
        # Reactive (inductive) power consumed from the grid
        query_Qic = f"SELECT datetime, value, " \
                   f"units " \
                   f"from {raw_table} " \
                   f"where id='{pt_id}' " \
                   f"and register_type='Qi+' " \
                   f"and datetime >= '{start_date}' " \
                   f"and datetime <= '{end_date}';"
        data_Qic = cass_con.read_query(query=query_Qic)
        data_Qic.set_index("datetime", inplace=True)
        data_Qic.rename(columns={"value": "Qi+"}, inplace=True)
        # Convert all "W" records to "kW"
        data_Qic.loc[data_Qic["units"].str.lower() == "var", "Qi+"] /= 1000
        data_Qic.drop("units", axis=1, inplace=True)

        # Reactive (inductive) power injected from the grid
        query_Qii = f"SELECT datetime, value, " \
                   f"units " \
                   f"from {raw_table} " \
                   f"where id='{pt_id}' " \
                   f"and register_type='Qi-' " \
                   f"and datetime >= '{start_date}' " \
                   f"and datetime <= '{end_date}';"
        data_Qii = cass_con.read_query(query=query_Qii)
        data_Qii.set_index("datetime", inplace=True)
        data_Qii.rename(columns={"value": "Qi-"}, inplace=True)
        # Convert all "var" records to "kvar"
        data_Qii.loc[data_Qii["units"].str.lower() == "var", "Qi-"] /= 1000
        data_Qii.drop("units", axis=1, inplace=True)
        data_Q = data_Qic.join(data_Qii, how="outer")

        # -- Capacitive
        # Reactive (capacitive) power consumed from the grid
        query_Qcc = f"SELECT datetime, value, " \
                   f"units " \
                   f"from {raw_table} " \
                   f"where id='{pt_id}' " \
                   f"and register_type='Qc+' " \
                   f"and datetime >= '{start_date}' " \
                   f"and datetime <= '{end_date}';"
        data_Qcc = cass_con.read_query(query=query_Qcc)
        data_Qcc.set_index("datetime", inplace=True)
        data_Qcc.rename(columns={"value": "Qc+"}, inplace=True)
        # Convert all "var" records to "kvar"
        data_Qcc.loc[data_Qcc["units"].str.lower() == "var", "Qc+"] /= 1000
        data_Qcc.drop("units", axis=1, inplace=True)
        data_Q = data_Q.join(data_Qcc, how="outer")

        # Reactive (capacitive) power injected from the grid
        query_Qci = f"SELECT datetime, value, " \
                   f"units " \
                   f"from {raw_table} " \
                   f"where id='{pt_id}' " \
                   f"and register_type='Qc-' " \
                   f"and datetime >= '{start_date}' " \
                   f"and datetime <= '{end_date}';"
        data_Qci = cass_con.read_query(query=query_Qci)
        data_Qci.set_index("datetime", inplace=True)
        data_Qci.rename(columns={"value": "Qc-"}, inplace=True)
        # Convert all "var" records to "kvar"
        data_Qci.loc[data_Qci["units"].str.lower() == "var", "Qc-"] /= 1000
        data_Qci.drop("units", axis=1, inplace=True)
        data_Q = data_Q.join(data_Qci, how="outer")
        data_Q.dropna(how="all", inplace=True)  # drop rows with all elements NaN

        # If Reactive Power DataFrame is not empty, calculate netload:
        if not data_Q.empty:
            data_Q.fillna(0, inplace=True)  # fill remaining rows with 0
            data_Q["value"] = data_Q["Qi+"] + data_Q["Qi-"] - data_Q["Qc+"] - data_Q["Qc-"]

            _dfr = data_Q["value"].resample("60T").mean()
            _dfr = _dfr.to_frame()
            _dfr.loc[:, 'register_type'] = "Q"
            _dfr.loc[:, 'id'] = pt_id
            _dfr.loc[:, 'units'] = "kvar"
            _dfr.reset_index(inplace=True)
            _dfr.dropna(inplace=True)
            _dfr.loc[:, 'last_updated'] = dt.datetime.utcnow()
            df = _dfr[["id", "register_type", "datetime", "last_updated",
                       "units", "value"]]

            result = False
            while result == False:
                try:
                    if to_csv:
                        df.to_csv(f"data/mv_data_netload_60/{pt_type.lower()}_Q_{pt_id}.csv", index=False)
                    else:
                        cass_con.insert_query(df=df,
                                              table=netload_table,
                                              exec_async=False)
                    result = True
                except Exception as e:
                    result = False
                    print(repr(e))
                    sleep(5)
                    pass
            return 1
        else:
            logger.warning(f"Skipped installation {pt_id} - No values.")
            return 0

    except BaseException as ex:
        logger.exception(msg=f"Failed for pt {pt_id}.")
        return 0

