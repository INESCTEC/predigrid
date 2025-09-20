from time import sleep
from cassandra.cluster import NoHostAvailable

from ..database.CassandraDB import CassandraDB
from ..helpers.log.log import just_console_log

logger = just_console_log()


def get_database_conn(n_retries=5):
    n_retries_ = 0
    while True:
        try:
            conn = CassandraDB.get_cassandra_instance()
            return conn
        except NoHostAvailable as ex:
            logger.error(ex)
            sleep(2)
            logger.warning(f"DB reconnection attempt "
                           f"({n_retries_}/{n_retries}) ...")
            n_retries_ += 1
            if n_retries_ > n_retries:
                raise ex
