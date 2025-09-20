import os


class SystemConfig:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.DATABASE_CLUSTER = None
        self.DATABASE_KEYSPACE = None
        self.TABLES = None
        self.USER_TABLES = None
        self.TOKEN_DURATION = None
        self.PRODUCTION = None
        self.SECRET_KEY = None
        self.load_settings()

    def load_settings(self):
        self.TABLES = dict(
            metadata=os.environ["TBL_METADATA"],
            raw=os.environ["TBL_RAW"],
            models_info=os.environ["TBL_MODELS"],
            netload=os.environ["TBL_NETLOAD"],
            forecast=os.environ["TBL_FORECAST"],
            users=os.environ["TBL_USERS"],
            metrics_longterm=os.environ["TBL_METRICS_LONGTERM"],
            metrics_shortterm=os.environ["TBL_METRICS_SHORTTERM"],
            nwp_grid_table=os.environ["TBL_NWP_GRID"],
        )
        self.DATABASE_CLUSTER = dict(
            contact_points=os.environ['CASSANDRA_CONTACT_POINTS'].split(','),
            port=int(os.environ['CASSANDRA_PORT']),
        )
        self.DATABASE_KEYSPACE = os.environ['CASSANDRA_KEYSPACE']
        self.TOKEN_DURATION = int(os.environ['TOKEN_DURATION'])
        self.SECRET_KEY = str(os.environ['SECRET_KEY'])

