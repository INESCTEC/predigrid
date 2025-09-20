import datetime
import os
import json
import numpy as np
import pandas as pd
import datetime as dt

from netCDF4 import Dataset
from forecast_weather.helpers.log import create_log
from forecast_weather.database.CassandraDB import CassandraDB
from forecast_weather.MeteogaliciaWeather import MeteogaliciaWeather
from forecast_weather.helpers.variable_helpers import (
    solar_variables,
    wind_variables
)

__ROOT_DIR__ = os.path.dirname(os.path.dirname(__file__))


def find_nearest_coord(coords, grid):
    distances = (np.sqrt((grid['latitude']-coords['latitude']) ** 2 + (grid['longitude']-coords['longitude'])**2)) # noqa
    min_coords = grid.loc[distances.idxmin()]

    return min_coords['latitude'], min_coords['longitude']


source_dict = {
    '_04km_history_': 'wrf_04km',
    '_12km_history_': 'wrf_12km',
    '_36km_history_': 'wrf_36km',
}

inv_source_dict = dict((y, x) for x, y in source_dict.items())


class WeatherManager:
    """
    This class entails the methods needed to manipulate data pertaining
    the Numerical Weather Predictions (NWP) needed for the forecasting serive.
    """
    def __init__(self):
        self.db_con = CassandraDB.get_cassandra_instance()
        self.installations = None
        self.grid = None
        self.grid_limits = None
        self.source = None
        self.weather_class = None
        self.installations_table = os.environ.get('INSTALLATIONS_TABLE', "installations")
        self.grid_table = os.environ.get('GRID_TABLE', "nwp_grid")
        self.solar_table = os.environ.get('SOLAR_TABLE', "nwp_solar")
        self.wind_table = os.environ.get('WIND_TABLE', "nwp_wind")

        self._log_file_path = os.path.join(__ROOT_DIR__, "files", "logs", "weather") # noqa
        self.logger = create_log(service="weather")

    def load_installations_metadata(self):
        """
        Simply run a query to load the metadata pertaining installations.
        :return:
        """
        self.logger.debug("Getting installation metadata from DB...")

        installations_df = self.db_con.read_query(
            f"select * from {self.installations_table};")

        if not installations_df.empty:
            installations_df['is_active'] = installations_df['is_active'].astype(bool) # noqa

        self.logger.debug("Getting installation metadata from DB... Ok!")
        self.installations = installations_df

    def load_grid_metadata(self):
        """
        Simply run a query to load the processed metadata pertaining the grid
        for the selected weather source.
        :return:
        """
        self.logger.debug(f"Getting grid metadata from DB "
                          f"for source {self.source}...")

        grid_df = self.db_con.read_query(
            f"select * from {self.grid_table} "
            f"where source = '{self.source}';")
        last_insertion = grid_df['inserted_at'].max()
        grid_df = grid_df[grid_df['inserted_at'] == last_insertion]

        self.logger.debug(f"Getting grid metadata from DB "
                          f"for source {self.source}... Ok!")

        self.grid = grid_df

    def clean_local_files(self, filespath=None):
        if filespath is None:
            filespath = os.path.join(__ROOT_DIR__, 'files', self.source)
        if self.weather_class is None:
            self.weather_class = self.__initialize_weather_class(filespath)
        most_recent_date = pd.to_datetime(self.weather_class.days[-1]).normalize() # noqa
        a_year_ago = most_recent_date - pd.DateOffset(years=1)
        range_dates = pd.date_range(a_year_ago, most_recent_date, freq='D')
        range_dates_str = [date.strftime("%Y%m%d") for date in range_dates]
        for file in [*os.walk(filespath)][0][2]:
            if not file.endswith(".nc4"):
                continue
            if file.split('.')[0] not in range_dates_str:
                os.remove(os.path.join(filespath, file))

    def load_grid_limits(self):
        """
        Load grid limits from environment variables.
        :return:
        """
        self.logger.debug("Setting grid limits:")
        self.logger.debug(f"West - {os.environ['WEST']}")
        self.logger.debug(f"South - {os.environ['SOUTH']}")
        self.logger.debug(f"East - {os.environ['EAST']}")
        self.logger.debug(f"North - {os.environ['NORTH']}")

        self.grid_limits = {
            'west': os.environ['WEST'],
            'south': os.environ['SOUTH'],
            'east': os.environ['EAST'],
            'north': os.environ['NORTH']
        }

    def set_source(self, source):
        self.source = source

    def __initialize_weather_class(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(__ROOT_DIR__, 'files', self.source)
            os.makedirs(filepath, exist_ok=True)
        self.logger.debug("Initializing MeteogaliciaWeather class... ")
        weather_class = MeteogaliciaWeather(dataset_path=filepath)
        weather_class.set_source(inv_source_dict[self.source])
        weather_class.subset = self.grid_limits
        self.logger.debug("Initializing MeteogaliciaWeather class... Ok!")
        return weather_class

    def process_grid_metadata(self):
        """
        Method to get NWP data and use it to build the grid metadata. Generates
        a DataFrame listing all the coordinates contained in the designated
        grid limits and retrieved variables. Also updates installations
        metadata by calculating which point in the grid is closer to which
        installation location.
        :return:
        """
        self.logger.info("Processing grid and installations metadata...")
        filepath = os.path.join(__ROOT_DIR__, 'files', self.source)
        os.makedirs(filepath, exist_ok=True)

        # Initialize WeatherManager class
        if self.weather_class is None:
            self.weather_class = self.__initialize_weather_class(filepath)

        grid_df = self.__process_grid(self.weather_class)
        grid_df, installations_df = self.__process_metadata(self.weather_class,
                                                            grid_df)
        self.logger.info("Processing grid and installations metadata... Ok!")

        return grid_df, installations_df

    def __process_grid(self, weather_class):
        change_last_cols = True
        same_grid = False

        # Load previously selected boundaries
        try:
            last_grid = self.__load_previous_grid_boundaries()
        except FileNotFoundError:
            # Use current grid boundaries
            self.logger.debug("Using current grid boundaries.")
            last_grid = self.grid_limits

        # Compare with current choice of grid boundaries
        if not self.__is_same_grid(self.grid_limits, last_grid):
            if self.__is_inside_grid(self.grid_limits, last_grid):
                self.logger.debug("Newly defined grid is contained in previous grid. " # noqa
                                  "No changes to previously downloaded files will be made.") # noqa
                change_last_cols = False
            else:
                self.logger.info("Newly defined grid either contains previous "
                                 "grid or intersects it. "
                                 "Replacing old NETCDF4 files.")
                filespath = os.path.join(__ROOT_DIR__, 'files', self.source)
                self.__empty_folder(filespath)
        else:
            same_grid = True

        # Check if DB already contains wanted metadata
        source_data = self.db_con.read_query(
            f"select * from {self.grid_table} "
            f"where source='{self.source}';")
        last_insertion = source_data['inserted_at'].max()
        source_data = source_data[source_data['inserted_at'] == last_insertion]

        if same_grid and not source_data.empty:
            return source_data

        # Download NETCDF4 file for present day. If fail, try previous days.
        date_ref = weather_class.days[-1]
        date_ref_incr = date_ref + datetime.timedelta(days=1)
        for i in range(4):
            date_ref_str = date_ref.strftime("%Y-%m-%d")
            date_ref_incr_str = date_ref_incr.strftime("%Y-%m-%d")

            # Date adjustment to start next loop
            weather_class.set_dates(date_ref_str,
                                    date_ref_incr_str)
            try:
                weather_class.download(force_download=True)
                break
            except Exception as e:
                date_ref_incr = date_ref
                date_ref = date_ref - datetime.timedelta(days=1)
                if i == 3:
                    self.logger.debug(
                        "Couldn't download recent NETCDF4 file "
                        "for grid processing. Please try again.")
                    raise e

        nc4file = "{}.nc4".format(weather_class.days[-1].strftime("%Y%m%d"))
        self.logger.debug(f"Using file {nc4file}")

        grid_df = pd.DataFrame(
            columns=['source', 'latitude', 'longitude', 'variables'])

        # Load most recent day's NWP data's coordinates information and variables # noqa
        ncdf4filepath = os.path.join(weather_class.dataset_path,
                                     nc4file)

        nc = Dataset(ncdf4filepath)
        grid_df['latitude'] = nc['lat'][:][:].ravel()
        grid_df['longitude'] = nc['lon'][:][:].ravel()
        grid_df['variables'] = ','.join(weather_class.variables)

        if change_last_cols:
            grid_df['last_request'] = pd.to_datetime('1970-01-01')  # Default
            grid_df['last_updated'] = pd.to_datetime('1970-01-01')  # Default
        else:
            # Means we're processing a subset of a previous grid, so the fields
            # "last_request" and "last_updated" should be kept
            source_data.set_index(['latitude', 'longitude'], inplace=True)
            grid_df.set_index(['latitude', 'longitude'], inplace=True)
            grid_df['last_request'] = source_data.loc[grid_df.index, 'last_request']
            grid_df['last_updated'] = source_data.loc[grid_df.index, 'last_updated']
            grid_df.reset_index(inplace=True)

        return grid_df

    def save_grid_boundaries(self):
        filepath = os.path.join(__ROOT_DIR__, 'files', self.source, "gridlimits.json") # noqa
        with open(filepath, "w") as f:
            json.dump(self.grid_limits, f)

    def __load_previous_grid_boundaries(self):
        filepath = os.path.join(__ROOT_DIR__, 'files', self.source, "gridlimits.json") # noqa
        try:
            with open(filepath, "r") as f:
                last_coords = json.load(f)
        except FileNotFoundError as fnt:
            self.logger.debug(f"Previous grid limits local file "
                              f"for source {self.source} not found.")
            raise fnt
        return last_coords

    @staticmethod
    def __empty_folder(path):
        for file in [*os.walk(path)][0][2]:
            os.remove(os.path.join(path, file))

    @staticmethod
    def __is_inside_grid(grid1, grid2):
        inside_north = float(grid1['north']) <= float(grid2['north'])
        inside_south = float(grid1['south']) >= float(grid2['south'])
        inside_west = float(grid1['west']) >= float(grid2['west'])
        inside_east = float(grid1['east']) <= float(grid2['east'])
        test = inside_north & inside_south & inside_west & inside_east
        return test

    @staticmethod
    def __is_same_grid(grid1, grid2):
        same_north = float(grid1['north']) == float(grid2['north'])
        same_south = float(grid1['south']) == float(grid2['south'])
        same_east = float(grid1['east']) == float(grid2['east'])
        same_west = float(grid1['west']) == float(grid2['west'])
        test = same_north & same_south & same_east & same_west
        return test

    def update_installations_db(self, installations):
        """
        Updates installations' metadata in the database.
        :param installations:
        :return:
        """
        # Check which installations have had their nearest coordinates recalculated # noqa
        mask_lat = installations['latitude_nwp'] != self.installations['latitude_nwp'] # noqa
        mask_lon = installations['longitude_nwp'] != self.installations['longitude_nwp'] # noqa
        subset_installations = installations[mask_lat & mask_lon].copy()
        subset_installations['last_updated'] = pd.to_datetime(dt.datetime.utcnow()) # noqa
        # If any installations were updated, update database
        if not subset_installations.empty:
            self.logger.info("Updating installations metadata in DB...")
            self.db_con.insert_query(subset_installations, self.installations_table) # noqa
            self.logger.info("Updating installations metadata in DB... Ok!")
        else:
            self.logger.info("Installations metadata not affected.") # noqa


    def __process_metadata(self, weather_class, grid_df):
        """
        Method to process metadata for a grid of coordinates. Also updates
        nearest grid coordinates in installations metadata.
        :param weather:
        :return:
        """

        # Checking if grid has column "inserted_at". If not, this means
        # It was not fetched from the database
        grid_changed = 'inserted_at' not in grid_df.columns

        # Create installations DataFrame with nearest NWP coordinates computed
        installations = self.__update_nearest_coords(self.installations,
                                                     grid_df)
        act_installations = installations.loc[installations['is_active'], :]
        # Determine which coordinates in the grid are going to be needed
        mask_lat = grid_df['latitude'].isin(act_installations['latitude_nwp'])
        mask_lon = grid_df['longitude'].isin(act_installations['longitude_nwp']) # noqa
        grid_df['is_active'] = mask_lat & mask_lon

        if grid_changed:
            grid_df['inserted_at'] = pd.to_datetime(dt.datetime.utcnow()).tz_localize(None)  # Default value # noqa
            grid_df['source'] = source_dict[weather_class.source]

        return grid_df, installations

    def __update_nearest_coords(self, installations, grid_df):
        """
        Calculates nearest grid coordinates for a set of installations.
        :param pd.DataFrame installations: DataFrame containing installation metadata
        :param pd.DataFrame grid_df: DataFrame containing grid metadata
        :return:
        """
        self.logger.debug(f"Computing nearest grid coordinates for installations...") # noqa
        inst_df = installations.copy()
        inst_df[['latitude_nwp', 'longitude_nwp']] = \
            installations[['latitude', 'longitude']].apply(find_nearest_coord,
                                                           grid=grid_df,
                                                           axis=1,
                                                           result_type='expand') # noqa
        self.logger.debug(f"Computing nearest grid coordinates for installations... Ok!") # noqa
        return inst_df

    def get_weather(self):
        """
        Method for retrieving weather data on a daily basis. NWP data is
        downloaded, processed and stored in the database. If needed,
        downloads and/or processes historical NWP data
        for recently activated coordinates.

        :return:
        """
        self.logger.info("Processing NWP data...")
        filepath = os.path.join(os.path.dirname(__file__), '..', 'files', self.source) # noqa
        os.makedirs(filepath, exist_ok=True)

        # Initialize WeatherManager class
        if self.weather_class is None:
            self.weather_class = self.__initialize_weather_class(filepath)

        try:
            nwp_solar, nwp_wind, grid = self.__process_weather_data(self.weather_class)
        except BaseException as exc:
            self.logger.error(f"Failed to process weather data: {repr(exc)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.logger.info("Processing NWP data... Ok!")
        return nwp_solar, nwp_wind, grid

    def __process_weather_data_day(self,
                                   day,
                                   weather_class,
                                   active_grid_coords):
        """
        Process NETCDF4 file for a specific file. Extracts NWP data for a given
        set of coordinates
        :param day:
        :param weather_class:
        :param active_grid_coords:
        :return:
        """
        nwp_solar = pd.DataFrame()
        nwp_wind = pd.DataFrame()

        filename = f"{day.strftime('%Y%m%d')}.nc4"
        path = os.path.join(weather_class.dataset_path, filename)
        try:
            nc = Dataset(path)
        except FileNotFoundError:
            self.logger.error(f"File for day {day.strftime('%Y-%m-%d')} "
                              f"not found. Skipping this day.")
            return nwp_solar, nwp_wind

        for latitude, longitude in active_grid_coords[['latitude', 'longitude']].values: # noqa
            this_nwp_solar = pd.DataFrame()
            this_nwp_wind = pd.DataFrame()
            # Find closest grid point
            ll_finder = np.sqrt(
                (nc['lat'][:][:] - latitude) ** 2 + (
                        nc['lon'][:][:] - longitude) ** 2)
            y, x = np.unravel_index(ll_finder.argmin(), ll_finder.shape)

            # Get request time from file "units" value
            request = pd.to_datetime(' '.join((nc['time'].units.split(' ')[2:]))) # noqa

            # List datetimes in file
            time = nc['time'][:].data
            time_unit = nc['time'].units[0]
            datetime_list = [request + pd.Timedelta(t, unit=time_unit)
                             for t in time]

            this_nwp_solar['datetime'] = datetime_list
            this_nwp_solar['source'] = self.source
            this_nwp_solar['request'] = request
            this_nwp_solar['latitude'] = latitude
            this_nwp_solar['longitude'] = longitude

            # Extract data for each variable
            for var in solar_variables():
                this_nwp_solar[var] = nc[var][:, y, x]

            nwp_solar = nwp_solar.append(this_nwp_solar)

            this_nwp_wind['datetime'] = datetime_list
            this_nwp_wind['source'] = self.source
            this_nwp_wind['request'] = request
            this_nwp_wind['latitude'] = latitude
            this_nwp_wind['longitude'] = longitude

            # Extract data for each variable
            for var in wind_variables():
                this_nwp_wind[var] = nc[var][:, y, x]

            nwp_wind = nwp_wind.append(this_nwp_wind)

        nc.close()

        return nwp_solar, nwp_wind

    def __process_weather_data(self, weather_class):
        """
        Routine to process NWP data. Processes data for most recent day and
        then checks which grid coordinates have been activated recently and
        need to have historical data retrieved.
        :param weather:
        :param grid:
        :return:
        """

        ll_cols = ['latitude', 'longitude']
        mask_grid_active = self.grid['is_active'].astype(bool)  # Active coordinates in grid # noqa

        active_grid_coords = self.grid.loc[mask_grid_active, [*ll_cols, 'last_request']] # noqa
        nwp_solar = pd.DataFrame()
        nwp_wind = pd.DataFrame()

        # Get date of oldest request
        start_date = active_grid_coords['last_request'].min().normalize()
        # If a grid pair of coordenates were previously deactivated, change start date to a year ago # noqa
        if start_date == pd.to_datetime('1970-01-01'):
            start_date = pd.to_datetime(weather_class.days[-1]) - pd.DateOffset(years=1) # noqa
        elif start_date != pd.to_datetime(weather_class.days[-1]).normalize():
            start_date = start_date + pd.DateOffset(days=1)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = (weather_class.days[-1]+pd.DateOffset(days=1)).strftime("%Y-%m-%d") # noqa
        # Change dates to retrieve data in weather service class
        weather_class.set_dates(start_date_str, end_date_str)
        self.logger.info("Downloading missing files...")
        weather_class.download(logger_path=self._log_file_path)
        self.logger.info("Downloading missing files... Ok!")
        last_updated = pd.to_datetime(dt.datetime.utcnow())
        # Process data for most recent day
        for day in weather_class.days:
            # Check which coordinates need data for this day
            mask_coords = active_grid_coords['last_request'] <= day
            day_activate_coords = active_grid_coords.loc[mask_coords, ll_cols] # noqa
            if not day_activate_coords.empty:
                self.logger.info(f"Processing NWP data for day: {day}. "
                                 f"{len(day_activate_coords)} grid point(s).")
                this_nwp_solar, this_nwp_wind = self.__process_weather_data_day( # noqa
                    day=day,
                    weather_class=weather_class,
                    active_grid_coords=day_activate_coords
                )
                if not this_nwp_wind.empty and not this_nwp_solar.empty:
                    nwp_solar = nwp_solar.append(this_nwp_solar)
                    nwp_wind = nwp_wind.append(this_nwp_wind)
                    self.grid.loc[day_activate_coords.index, 'last_request'] = pd.to_datetime(day) # noqa
                    self.grid.loc[day_activate_coords.index, 'last_updated'] = last_updated # noqa

        nwp_solar['last_updated'] = last_updated
        nwp_wind['last_updated'] = last_updated

        return nwp_solar, nwp_wind, self.grid

    def update_db_active_grid_points(self):
        """
        This method checks points in the grid that are close to any registered
        installations, checking if they are marked as active. If any previously
        active grid point becomes unused due to lack of any nearby registered
        installations (e.g., installations around a grid point being
        unregistered), it is marked as inactive.
        :return:
        """
        self.logger.info("Updating grid points status...")
        # Check if the needed metadata was previously fetched from database
        if self.installations is None:
            self.load_installations_metadata()
        if self.grid is None:
            self.load_grid_metadata()

        # Get active installations
        active_installs = self.installations.loc[self.installations['is_active'], :] # noqa

        # Get active grid points through the installations' metadata
        in_use_coords = active_installs[['latitude_nwp', 'longitude_nwp']]
        mask_lat = self.grid['latitude'].isin(in_use_coords['latitude_nwp'].values) # noqa
        mask_lon = self.grid['longitude'].isin(in_use_coords['longitude_nwp'].values) # noqa

        # Update active status in the grid metadata (important to deactivate grid points) # noqa
        is_active = mask_lat & mask_lon
        self.grid.loc[~is_active, 'is_active'] = 0
        self.grid.loc[is_active, 'is_active'] = 1

        self.insert_in_database(self.grid, self.grid_table)
        self.logger.info("Updating grid points status... Ok!")

    def insert_in_database(self, df, table):
        if df.empty:
            self.logger.info("DataFrame empty. Nothing to insert.")
            return None
        self.logger.info(f"Inserting data into table: {table}...")
        n_partitions = df.shape[0] // 10000
        if n_partitions > 0:
            self.logger.debug(f"Partitioning data for insertion: "
                              f"{n_partitions+1} partitions")
            for i in range(n_partitions+1):
                self.logger.debug(f"Inserting partition {i+1}")
                subdf = df.iloc[10000*i:10000*(i+1)]
                self.db_con.insert_query(subdf, table)
        else:
            self.db_con.insert_query(df, table)
        self.logger.info(f"Inserting data into table: {table}... Ok!")
