import os
import argparse

from configs import SystemConfig, LogConfig
from src.common.databases import CassandraDB
from src.util.generate_admin_token import register_admin_user

parser = argparse.ArgumentParser(description='Register new admin.')
parser.add_argument('--username', type=str, help='Admin username',
                    required=True)
parser.add_argument('--password', type=str, help='Admin password',
                    required=True)
args = parser.parse_args()

# -- Project ROOT directory:
__ROOT_DIR__ = os.path.dirname(os.path.abspath(__file__))
print("ROOT DIR:", __ROOT_DIR__)

# -- Init configs:
config_class = SystemConfig(root_dir=__ROOT_DIR__)

# -- Init Loggers:
loggers = LogConfig(root_dir=__ROOT_DIR__).get_loggers()
logger = loggers["users"]

try:
    # Connect to Cassandra DB:
    db_con = CassandraDB.get_cassandra_instance(config=config_class)
    register_admin_user(
        config_class=config_class,
        db_con=db_con,
        username=args.username,
        password=args.password,
        logger=logger
    )

except BaseException as ex:
    logger.exception(msg=repr(ex))
    exit("Error! Unable to perform operation. See error description in "
         "service logs (users_logs.log)")
