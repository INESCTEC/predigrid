import os
import pandas as pd

from time import sleep

from cassandra.cluster import Cluster
from cassandra.query import tuple_factory
from cassandra.protocol import NumpyProtocolHandler


IP_ADDRESS = os.environ.get("CASSANDRA_CONTACT_POINTS")
if IP_ADDRESS is not None:
    IP_ADDRESS = IP_ADDRESS.split(",")

KEYSPACE = os.environ.get("CASSANDRA_KEYSPACE")
PORT = int(os.environ.get("CASSANDRA_PORT", default=8000))


class CassandraDB:

    instance = None

    def __init__(self, config):
        self.cluster = Cluster(**config.DATABASE_CLUSTER)
        self.session = self.cluster.connect(config.DATABASE_KEYSPACE)
        self.session.row_factory = tuple_factory  # Returns each row as a tuple
        self.session.client_protocol_handler = NumpyProtocolHandler
        CassandraDB.instance = self

    @staticmethod
    def get_cassandra_instance(config):
        if CassandraDB.instance:
            return CassandraDB.instance
        print("new connection to db")
        return CassandraDB(config=config)

    @staticmethod
    def disconnect():
        if CassandraDB.instance is not None:
            CassandraDB.instance.shutdown_cluster()

    def read_query(self, query):
        from cassandra.query import SimpleStatement

        # prepared_stmt = self.session.prepare(query)
        prepared_stmt = SimpleStatement(query, fetch_size=2000)
        rslt = self.session.execute(prepared_stmt)
        df = pd.DataFrame()
        for r in rslt:
            df = df.append(pd.DataFrame(r), ignore_index=True)

        return df

    def insert_query(self, df, table, fields=None, exec_async=True):  # noqa
        """
        This method inserts a Pandas Dataframe in a cassandradb rw_eb or
        rw_dtc tables
        :param df: pandas DataFrame
        :param table: Cassandra table
        :param fields: you can send custom fields
        :param exec_async: you execute query in async
        :return:
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError('ERROR! Data to be inserted into Cassandra DB is '
                            'not a pandas.DataFrame')

        fields = paste(df.columns) if not None else fields

        statement = "INSERT INTO " + table + "(" + fields + ") VALUES (" \
                    + paste(["?"] * len(df.columns)) + ");"

        prepared_stmt = self.session.prepare(statement)

        futures = []  # array to save async execution results.
        # Async execution with blocking wait for results (futures list)
        for i, row in enumerate(df.iterrows()):
            try:
                if exec_async:  # noqa
                    futures.append(self.session.execute_async(prepared_stmt,
                                                              row[1].values))
                else:
                    self.session.execute(prepared_stmt, row[1].values)
            except Exception as e:
                print(e)

        # wait for async inserts to complete complete and use the results
        # (iterates through all the results)
        for i, future in enumerate(futures):
            state = False
            try:
                while state == False:
                    future.result()
                    state = True
            except Exception as ex:
                print("Warning: The following exception occurred waiting for future {row}".format(row=i))
                print(ex)
                sleep(0.5)
                print("retring...")
                pass

    def execute_query(self, query):
        """
        This method sync execute a query statement
        :param query:
        :return:
        """
        return self.session.execute(query)

    def shutdown_cluster(self):
        """
        Closes all sessions and connection associated with this Cluster
        :return:
        """
        self.cluster.shutdown()


def paste(x, sep=", "):
    """
    Custom string formatting function to format (???) output.
    """
    out = ""
    for i in x:
        out += i + sep
    return out.strip(sep)
