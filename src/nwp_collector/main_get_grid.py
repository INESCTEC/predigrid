# -- Uncomment code bellow for debug purposes --
# from dotenv import load_dotenv
# load_dotenv(r"ENV\.env.dev")

import os

from forecast_weather.WeatherManager import WeatherManager


wm = WeatherManager()
wm.load_installations_metadata()
wm.load_grid_limits()
wm.set_source(os.environ['SOURCE'])

grid, installations = wm.process_grid_metadata()
wm.insert_in_database(grid, os.environ['GRID_TABLE'])
wm.update_installations_db(installations)
wm.save_grid_boundaries()