import os
import re
import datetime
import numpy as np
import pandas as pd
from typing import Union

from forecast_api import EvaluationClass
from metrics.helpers.log import create_log
from metrics.database.CassandraDB import CassandraDB

__ROOT_DIR__ = os.path.dirname(os.path.dirname(__file__))


def merge_real_forec(real_df: pd.DataFrame, forec_df: pd.DataFrame) -> pd.DataFrame:
    quantiles_re = re.compile('q[0-9]{2}$')
    quantile_cols = [col for col in forec_df.columns
                     if quantiles_re.match(col)]
    merged = (forec_df[['datetime', 'request',
                        *quantile_cols, 'horizon']]
              .merge(real_df[['datetime', 'real']],
                     on=['datetime'],
                     how='inner'))
    merged.sort_values(['request', 'datetime'], inplace=True)
    merged['datetime'] = merged['datetime'].dt.tz_localize('UTC')
    merged['request'] = merged['request'].dt.tz_localize('UTC')
    return merged

class Metrics:
    """
    Class to manage data for computing key performance indicators (KPI).

    """
    def __init__(self):
        self.installations = None
        self.db = None

        self._inst_table = None
        self._real_table = None
        self._forec_table = None
        self._launch_time = None

        self._inst_table = os.environ['TABLE_CLIENTS']
        self._real_table = os.environ['TABLE_REAL_DATA']
        self._forec_table = os.environ['TABLE_FORECASTS']
        self._metrics_year_table = os.environ['TABLE_METRICS_YEAR']
        self._metrics_2wk_table = os.environ['TABLE_METRICS_2WEEKS']

        log_path = os.path.join(__ROOT_DIR__, "logs", "metrics")
        self.logger = create_log(service='metrics',
                                 path=log_path)

    def _query_measurements(self,
                            inst_id: str,
                            register_type: str,
                            is_solar: bool) -> pd.DataFrame:
        launch_time_str = self._launch_time.strftime("%Y-%m-%d %H:%M:%S")

        query = f"select * from {self._real_table} " \
                f"where id='{inst_id}' " \
                f"and register_type='{register_type}' " \
                f"and datetime<'{launch_time_str}';"

        df = self.db.read_query(query)
        if is_solar:
            df = df[df['value'] != 0].copy()
        df.sort_values(by=['datetime'], inplace=True)

        return df

    def _query_forecasts(self, inst_id: str, register_type: str) -> pd.DataFrame:
        one_year_ago = self._launch_time - pd.DateOffset(years=1)
        one_year_ago_str = one_year_ago.strftime("%Y-%m-%d %H:%M:%S")

        launch_time_str = self._launch_time.strftime("%Y-%m-%d %H:%M:%S")

        query = f"select * from {self._forec_table} " \
                f"where id='{inst_id}' " \
                f"and register_type='{register_type}'" \
                f"and datetime>='{one_year_ago_str}' " \
                f"and datetime<'{launch_time_str}';"

        df = self.db.read_query(query)

        df.sort_values(by=['datetime', 'request', 'last_updated'],
                       inplace=True)

        df.drop_duplicates(subset=['datetime', 'request',
                                   'horizon', 'last_updated'],
                           keep='last',
                           inplace=True)
        return df

    def load_db(self, db_con: CassandraDB) -> None:
        """
        Stores reference to a database connection object (Cassandra).
        :param db_con: the object connecting to a database.
        :type db_con: CassandraDB
        """
        self.db = db_con

    def set_launch_time(self, launch_time: Union[str, pd.Timestamp, datetime.datetime]) -> None: # noqa
        """
        Defines upper limit for dates to calculate KPIs. Must be in UTC.

        :param launch_time: date to use as upper limit.
        :type launch_time: str, pd.Timestamp, datetime.datetime
        """
        self._launch_time = pd.to_datetime(launch_time).tz_localize('UTC')

    def _get_installations_data(self) -> None:
        inst_df = self.db.read_query(f"select * from {self._inst_table};")
        self.installations = inst_df

    def _min_max(self, real_df: pd.DataFrame) -> (float, float):
        # m = self.db.read_query(
        #     f"select min(value) as min, max(value) as max "
        #     f"from {self._real_table} "
        #     f"where id='{inst_id}' "
        #     f"and register_type='{register_type}';")

        m = real_df.agg(['min', 'max'])
        min, max = m.loc[['min', 'max'], 'value']

        if np.abs(min) > max:
            max = np.abs(min)

        return float(min), float(max)

    def _calc_errors(self, data: pd.DataFrame, max: float) -> pd.DataFrame:
        forec_col = 'q50' if 'q50' in data.columns else 'forecast'
        errors = pd.DataFrame(index=data.index)
        errors['datetime'] = data['datetime']
        errors['request'] = data['request']
        errors['horizon'] = data['horizon']

        errors['ae'] = (data['real']-data[forec_col]).abs()
        errors['se'] = (data['real']-data[forec_col])**2

        errors['nae'] = errors['ae'] / max
        errors['nse'] = errors['se'] / max
        return errors

    def _calc_metrics(self, data: pd.DataFrame, min: float, max: float) -> pd.DataFrame:
        opts = {
            'mae': True,
            'rmse': True,
            'crps': True,
            'y_min': min,
            'y_max': max,
        }
        eval = EvaluationClass()
        metrics = (data.dropna(axis=1, how='all')
                   .groupby(data['horizon'])
                   .apply(eval.calc_metrics, **opts))

        metrics = (metrics
                   .reset_index(level=1, drop=True)
                   .reset_index())

        metrics['nmae'] = metrics['mae'] / max
        metrics['nrmse'] = metrics['rmse'] / max
        return metrics

    def run_calc_metrics(self,
                         register_type: str,
                         calc_errors: bool = False,
                         save: bool = True) -> pd.DataFrame:
        """
        Main routine to perform KPI computation. For each installation it:

        - Loads measurements from the database for respective register type
        - Loads forecasts from the database for respective register type
        - Computes KPIs for the last year and for the last two weeks (considering the previously define launch time)
        - If calc_erros is True, generates DataFrame for each timestamp's error
        - If save is True, uploads KPIs to database

        :param register_type: Target register type. "P" or "Q".
        :type register_type: str
        :param calc_errors: Flag to generate error metrics per timestamp.
        :type calc_errors: bool
        :param save: Flag to upload computed KPIs to database.
        :type save: bool
        :return: DataFrame with errors per timestamp. Empty if calc_errors is False
        :rtype: pd.DataFrame
        """
        self.logger.info(f"Processing forecast metrics for register type: {register_type}") # noqa
        self._get_installations_data()

        metrics_yearly = pd.DataFrame()
        metrics_2wk = pd.DataFrame()
        errors = pd.DataFrame()
        for inst_id, id_type in self.installations[['id', 'id_type']].values:
            self.logger.debug(f"Processing metrics ({register_type}) for installation: {inst_id}") # noqa
            # Queries all data to get min and max
            real_df = self._query_measurements(inst_id=inst_id,
                                               register_type=register_type,
                                               is_solar=id_type == "solar")
            ymin, ymax = self._min_max(real_df)
            real_df = real_df.rename(columns={'value': 'real'})
            # -- Queries last year's forecasts
            forec_df = self._query_forecasts(inst_id, register_type)
            if real_df.empty or forec_df.empty:
                self.logger.debug(f"{inst_id}: Data for metrics computation unavailable. Skipping.")
                continue
            # "Horizon" processing for solar and wind installations
            if id_type in ["solar", "wind"]:
                forec_df['horizon'] = (forec_df['datetime'] - forec_df['request']).dt.days
                forec_df['horizon'] = "D+"+forec_df["horizon"].astype(str)
                forec_df.loc[forec_df['horizon'] == "D+0", "horizon"] = "D"
            data = merge_real_forec(real_df, forec_df)

            # -- Compute metrics for last year's data
            if data.empty:
                self.logger.debug(f"{inst_id}: Data for metrics computation unavailable. Skipping.")
                continue
            inst_metrics_yearly = self._calc_metrics(data=data,
                                                     min=ymin,
                                                     max=ymax)

            inst_metrics_yearly['id'] = inst_id
            inst_metrics_yearly['register_type'] = register_type
            inst_metrics_yearly['request'] = self._launch_time.tz_convert(None)
            inst_metrics_yearly['last_updated'] = pd.to_datetime(datetime.datetime.utcnow()) # noqa
            metrics_yearly = metrics_yearly.append(inst_metrics_yearly)

            # -- Compute metrics for last two weeks' data
            _2wk_ago = self._launch_time - pd.DateOffset(weeks=2)
            # - Masks to restrain data to last two weeks
            lower_bound = data['datetime'] >= _2wk_ago
            upper_bound = data['datetime'] < self._launch_time
            data_2k = data[lower_bound & upper_bound]
            if not data_2k.empty:
                inst_metrics_2wk = self._calc_metrics(data=data_2k,
                                                      min=ymin,
                                                      max=ymax)
                inst_metrics_2wk['id'] = inst_id
                inst_metrics_2wk['register_type'] = register_type
                inst_metrics_2wk['request'] = self._launch_time.tz_convert(None)
                inst_metrics_2wk['last_updated'] = pd.to_datetime(datetime.datetime.utcnow()) # noqa
                metrics_2wk = metrics_2wk.append(inst_metrics_2wk)

            if calc_errors:
                if not data.empty:
                    inst_errors = self._calc_errors(data, ymax)
                    inst_errors['id'] = inst_id
                    inst_errors['register_type'] = register_type
                    inst_errors['last_updated'] = pd.to_datetime(datetime.datetime.utcnow()) # noqa
                    errors = errors.append(inst_errors)

        if save:
            if metrics_yearly.empty:
                self.logger.debug("Yearly metrics DataFrame empty. Nothing to insert.")
            else:
                self.db.insert_query(metrics_yearly, self._metrics_year_table)
            if metrics_2wk.empty:
                self.logger.debug("Last 2 weeks metrics DataFrame empty. Nothing to insert.")
            else:
                self.db.insert_query(metrics_2wk, self._metrics_2wk_table)
        return errors
