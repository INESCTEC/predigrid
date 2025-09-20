# -- Uncomment code bellow for debug purposes --
# from dotenv import load_dotenv
# load_dotenv(r"ENV\.env.dev")

import datetime as dt
from metrics.MetricsClass import Metrics
from metrics.database.helpers import get_database_conn

db_con = get_database_conn()
now = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

metric = Metrics()
metric.load_db(db_con)
metric.set_launch_time(now)
metric.run_calc_metrics(register_type='P', save=True)
metric.run_calc_metrics(register_type='Q', save=True)
