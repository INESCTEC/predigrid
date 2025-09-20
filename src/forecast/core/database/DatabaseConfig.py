
class DatabaseConfig:

    TABLES = dict(
        objects="mv_objects",
        installations="installations",
        models_info="models_info",

        load_main_logs="mv_main_logs",
        load_models_logs="mv_aux_logs",

        measurements='processed',
        forecasts='forecast',

        nwp_solar='nwp_solar',
        nwp_wind='nwp_wind',
    )
