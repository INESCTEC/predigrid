# -- Uncomment code bellow for debug purposes --
# from dotenv import load_dotenv
# load_dotenv(r"ENV\.env.dev")

import os

from forecast_weather.WeatherManager import WeatherManager

wm = WeatherManager()
wm.load_installations_metadata()
wm.load_grid_limits()
wm.set_source(os.environ['SOURCE'])

# Check if any changes to the grid have been made
wm.grid, wm.installations = wm.process_grid_metadata()
wm.insert_in_database(wm.grid, os.environ['GRID_TABLE'])
wm.update_installations_db(wm.installations)
wm.save_grid_boundaries()

wm.clean_local_files()
nwp_solar, nwp_wind, grid = wm.get_weather()

wm.insert_in_database(nwp_solar, os.environ['SOLAR_TABLE'])
wm.insert_in_database(nwp_wind, os.environ['WIND_TABLE'])
wm.insert_in_database(grid, os.environ['GRID_TABLE'])
