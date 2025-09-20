# -- Uncomment code bellow for debug purposes --
# from dotenv import load_dotenv
# load_dotenv(r".\ENV\.env.dev")

# A+: Energia ativa consumida da rede: [número do registo]=1
# A-: Energia ativa injetada da rede: [número do registo]=4
# Q: Energia indutiva consumida da rede: [número do registo]=2 + Energia indutiva injetada da rede: [número do registo]=5 - Energia capacitiva consumida da rede: [número do registo]=3 - Energia capacitiva injetada da rede: [número do registo]=6. Em modo operacional são estas três variáveis que vamos prever. Alguns ficheiros estão vazios, por isso consideras 0.


if __name__ == '__main__':
    import os
    import argparse
    import datetime as dt

    from configs import SystemConfig, LogConfig
    from src.database import CassandraDB
    from src.netload_data import calculate_netload

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_days', type=int, default=30,
                        help='Number of preceeding days (to the current day) '
                             'considered at the cleansing process.')
    parser.add_argument('--last_date', type=str, default=None,
                        help='Last date to process (defaults to today). '
                             'Format: %Y-%m-%d (e.g. 2021-08-11)')
    args = parser.parse_args()
    N_LOOKBACK_DAYS = args.n_days
    if args.last_date is None:
        LAST_DATE = dt.datetime.utcnow()
    else:
        LAST_DATE = dt.datetime.strptime(args.last_date, "%Y-%m-%d")

    # -- Initialize Config Class:
    root_dir = os.path.dirname(__file__)
    config = SystemConfig(root_dir=root_dir)
    loggers = LogConfig(root_dir=root_dir).get_loggers()
    cass_con = CassandraDB.get_cassandra_instance(config=config)

    # Select logger:
    logger = loggers["netload"]
    logger.info(">" * 70)
    logger.info(f"Warning! Working on {os.environ['MODE']} mode!")
    logger.info("<" * 70)

    # -- Calculate and insert netload power data (Consumed - Injected)
    start_date = LAST_DATE - dt.timedelta(days=N_LOOKBACK_DAYS)
    calculate_netload(
        config=config,
        conn=cass_con,
        logger=logger,
        start_date=start_date,
        end_date=LAST_DATE
    )
