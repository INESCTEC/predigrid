import pandas as pd

from cassandra.cluster import Cluster
from cassandra.query import tuple_factory, SimpleStatement

CQLENG_ALLOW_SCHEMA_MANAGEMENT = 'CQLENG_ALLOW_SCHEMA_MANAGEMENT'


def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)


def paste(x, sep=", "):
    """
    Custom string formatting function to format (???) output.
    """
    out = ""
    for i in x:
        out += i + sep
    return out.strip(sep)


class CassandraDB:

    instance = None

    def __init__(self, config):

        self.cluster = Cluster(**config.DATABASE_CLUSTER)
        self.session = self.cluster.connect(config.DATABASE_KEYSPACE)
        self.session.row_factory = tuple_factory
        CassandraDB.instance = self

    @staticmethod
    def get_cassandra_instance(config):
        if CassandraDB.instance:
            # print("reused connection")
            return CassandraDB.instance
        print("new connection to db")
        return CassandraDB(config=config)

    def read_query(self, query):
        self.session.row_factory = pandas_factory
        self.session.default_fetch_size = None
        prepared_stmt = SimpleStatement(query)
        result = self.session.execute(prepared_stmt)

        return result._current_rows

    def insert_query(self, df, table, fields=None, use_async=True, logger=None):
        """
        This method inserts a Pandas Dataframe in a cassandradb rw_eb or rw_dtc tables
        :param df: pandas DataFrame
        :param table: Cassandra table
        :param fields: you can send custom fields
        :param async: you execute query in async
        :return:
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError('ERROR! Data to be inserted into Cassandra DB is not a pandas dataframe')

        fields = paste(df.columns) if not None else fields

        statement = "INSERT INTO " + table + "(" + fields + ") VALUES (" + paste(
            ["?"] * len(df.columns)) + ");"

        prepared_stmt = self.session.prepare(statement)

        futures = []  # array to save async execution results.
        # Async execution with blocking wait for results (futures list)
        for i, row in enumerate(df.iterrows()):
            try:
                if use_async:
                    futures.append(self.session.execute_async(prepared_stmt, row[1].values))
                else:
                    self.session.execute(prepared_stmt, row[1].values)
            except BaseException:
                logger.exception(msg="Exception while inserting data into DB.")

        # wait for async inserts to complete complete and use the results (iterates through all the results)
        for i, future in enumerate(futures):
            try:
                future.result()
            except Exception as e:
                print("Warning: The following exception occurred when inserting row {row}".format(row=i))
                print(e)

    def execute_query(self, query):
        """
        This method sync execute a query statement
        :param query:
        :return:
        """
        return self.session.execute(query)

    def disconnect(self):
        self.session.shutdown()
        self.cluster.shutdown()
