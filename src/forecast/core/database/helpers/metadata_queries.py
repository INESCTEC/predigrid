from cassandra.connection import ConnectionException


####################################################
# Installation metadata
####################################################
def query_installation_metadata(conn, table, inst_id, use_mock_data):
    if use_mock_data:
        from .mock.scripts.metadata import create_mock_inst_metadata
        df = create_mock_inst_metadata(inst_id=inst_id)
    else:
        # -- Prepare query statement & query DB:
        query = f"select * from {table} where id='{inst_id}';"
        try:
            df = conn.read_query(query=query)
        except AttributeError as ex:
            raise ConnectionException(repr(ex))
    # Unpack models metadata:
    return df.to_dict(orient="records")[0]


####################################################
# Models metadata
####################################################
def query_models_metadata(conn, table, inst_id, use_mock_data):
    if use_mock_data:
        from .mock.scripts.metadata import create_mock_models_metadata
        df_dict = create_mock_models_metadata(inst_id=inst_id)
        return df_dict
    else:
        # -- Prepare query statement & query DB:
        query = f"select * from {table} where id='{inst_id}';"
        try:
            # Fetch models metadata (DataFrame):
            _df = conn.read_query(query=query)
            # Unpack models metadata:
            df_dict = _df.to_dict(orient="records")[0]
            return df_dict
        except AttributeError as ex:
            raise ConnectionException(repr(ex))
