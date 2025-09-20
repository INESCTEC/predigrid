import os


class SystemConfig:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.DATABASE_CLUSTER = None
        self.DATABASE_KEYSPACE = None
        self.TABLES = None
        self.PRODUCTION = None
        self.N_JOBS = None
        self.load_settings()

    def load_settings(self):
        self.N_JOBS = int(os.environ["N_JOBS"])
        self.TABLES = dict(
            metadata=os.environ["TBL_METADATA"],
            raw=os.environ["TBL_RAW"],
            netload=os.environ["TBL_NETLOAD"],
            processed=os.environ["TBL_PROCESSED"],
        )
        self.DATABASE_CLUSTER = dict(
            contact_points=os.environ['CASSANDRA_CONTACT_POINTS'].split(','),
            port=int(os.environ['CASSANDRA_PORT']),
        )
        self.DATABASE_KEYSPACE = os.environ['CASSANDRA_KEYSPACE']

