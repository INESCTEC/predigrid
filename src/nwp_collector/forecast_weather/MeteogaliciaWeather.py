import os
import pytz
import datetime
import functools
import numpy as np
import urllib.request
from operator import is_not
from datetime import timedelta
from urllib.error import HTTPError
from joblib import Parallel, delayed
from netCDF4 import num2date, Dataset
from multiprocessing import cpu_count
from http.client import IncompleteRead
from forecast_weather.helpers.log import create_log, just_console_log
from forecast_weather.helpers.date_helpers import set_date_parameter_url
from forecast_weather.helpers.variable_helpers import meteogalicia_variables

__author__ = "André Garcia"
__credits__ = ["Ricardo Andrade"]
__version__ = "1.0.2"
__maintainer__ = "André Garcia"
__email__ = "andre.f.garcia@inesctec.pt"
__status__ = "Development"

class MeteogaliciaWeather:

    METEOGALICIA_04KM = '_04km_history_'
    METEOGALICIA_12KM = '_12km_history_'
    METEOGALICIA_36KM = '_36km_history_'

    subset_degrees = 0.5

    sources = {
        '_04km_history_':
            'http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d03/{year}/{month}/'
            'wrf_arw_det_history_d03_{full_date}_0000.nc4?',
        '_12km_history_':
            'http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d02/{year}/{month}/'
            'wrf_arw_det_history_d02_{full_date}_0000.nc4?',
        '_36km_history_':
            'http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d01/{year}/{month}/'
            'wrf_arw_det_history_d01_{full_date}_0000.nc4?'
    }

    variables = meteogalicia_variables()

    def __init__(self, dataset_path):
        """
        :param dataset_path:
        """
        self.url = None
        self.source = None
        self.subset = False
        self.latitude = None
        self.longitude = None
        self.num_cores = cpu_count()
        self.dataset_path = dataset_path
        self.days = list([datetime.datetime.today()])
        self.variables = meteogalicia_variables()

        if not os.path.exists(dataset_path):
            os.makedirs(self.dataset_path)

    def set_source(self, source):

        if source in self.sources:
            self.source = source
            self.url = self.sources.get(source)
        else:
            raise KeyError("The source: {source} doesn\'t exist in the sources list".
                           format(source=source))

    def set_parallel_jobs(self, jobs):
        self.num_cores = jobs

    def set_dates(self, start_date, end_date):

        from datetime import datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        date_array = (start + timedelta(days=x) for x in range(0, (end - start).days))
        self.days = list(date_array)

    def set_subset_degrees(self, degrees):
        """
        Changes the degrees where grid will be cut around the coordinates
        By default is 0.5
        :param degrees:
        :return:
        """
        self.subset_degrees = degrees
        self.set_subset()

    def set_variables(self, variables):
        self.variables = variables

    def set_latitude(self, latitude):
        self.latitude = latitude

    def set_dataset_path(self, dataset_path):
        self.dataset_path = dataset_path

    def set_longitude(self, longitude):
        self.longitude = longitude

    def set_subset(self):
        self.__set_subset()

    def download(self, logger_path=None, force_download=False):

        self.__check_source()  # check if source is set
        if not force_download:
            days_to_download = self.__check_missing_files()
        else:
            days_to_download = self.days
        download = []
        if days_to_download:
            for day in days_to_download:
                download.append({'day': day, 'url': self.__parse_url(day)})

            if len(download) > 1:
                (Parallel(n_jobs=self.num_cores)(delayed(download_url)(data['url'], self.dataset_path, data['day'], logger_path)
                                                 for data in download))
            else:
                download_url(download[0]['url'], self.dataset_path, download[0]['day'], logger_path)

    def __check_missing_files(self):

        files = os.listdir(self.dataset_path)
        days_to_download = []
        # date_array = (self.dt_from + datetime.timedelta(days=x) for x in range(0, (self.dt_to - self.dt_from).days))

        for day in self.days:
            file = day.strftime('%Y%m%d') + '.nc4'
            if file not in files:
                days_to_download.append(day)

        return days_to_download

    def __check_source(self):

        if self.source is None:
            raise ValueError("You must set source where files must be downloaded")

    def __set_subset(self):
        """
        This method sets a subset for the netcdf4 download of a much lower grid
        Builds a boundary grid surrounding the central coordinate point.
        :return:
        """

        if not all([self.latitude, self.longitude]):
            raise TypeError("You must set latitude and longitude!")

        self.subset = {'north': self.latitude + self.subset_degrees,
                       'south': self.latitude - self.subset_degrees,
                       'west': self.longitude - self.subset_degrees,
                       'east': self.longitude + self.subset_degrees}

    def __parse_url(self, day):

        from urllib.parse import urlencode, quote_plus

        # build variables here (common to all downloads)
        year = day.strftime('%Y')
        month = day.strftime('%m')
        day_in_string = day.strftime('%Y%m%d')

        # Add url sources from other Mandeo grids if required and parse here
        if self.INESCTEC in self.source:
            url = self.url.format(year, day_in_string)
        else:
            url = self.url.format(year=year, month=month, full_date=day_in_string)

        url += urlencode({'var': self.variables}, quote_plus)

        if self.subset:
            url += '&' + urlencode(self.subset, quote_plus)

        # add lat and lon to collect
        url = url + '&' + urlencode({'var': 'lat'})
        url = url + '&' + urlencode({'var': 'lon'})

        url += '&' + urlencode(set_date_parameter_url(day))
        # at last add the netcdf4 type (common to all downloads)
        url += '&' + 'disableLLSubset=on&disableProjSubset=on&horizStride=1&'
        url += '&' + 'accept=netcdf'

        return url

    def process_weather_forecast(self):

        """
            Variables retrieved from the netcdf4 files are stored and available to be processed
            by a pandas dataframe in it's header.
            :return:
            """

        self.download()

        data = (Parallel(n_jobs=self.num_cores)(delayed(netcdf4_to_json)(os.path.join(
            self.dataset_path, "{}.nc4".format(day.strftime("%Y%m%d"))),
            self.latitude, self.longitude, self.variables) for day in self.days))

        data = list(filter(functools.partial(is_not, False), data))
        return [item for sublist in data for item in sublist]

    @staticmethod
    def check_if_corrupted(file):
        """
        This function verifies if the existent file is corrupted
        :param file:
        :return:
        """
        try:
            nc_file = Dataset(file)
            nc_file.close()
        except Exception:
            return file

        return False


def netcdf4_to_json(filename, latitude, longitude, variables):
    """
    This function main objective should be to check if weather file exist. If not it will try to request a new file
    If file exists it retrieves the weather variables values
    Historical files are tested before entering this function.
    Forecast files by the other hand are not. For that cause there is a method
    :param filename:
    :param latitude:
    :param longitude:
    :param variables:
    :param log:
    :return:
    """
    try:
        nc_file = Dataset(filename)
        return process_netcdf4(nc_file, latitude, longitude, variables)
    except (OSError, RuntimeError):
        return False


def download_url(url, dataset_path, day, logger_path=None):
    """
    :param url:
    :param dataset_path:
    :param day:
    :return:
    """
    if logger_path is not None:
        log_method = create_log(service="weather")
    else:
        log_method = just_console_log()
    tries = 3
    while tries > 0:
        try:
            if tries == 3:
                log_method.info("Downloading file {}".format(url))
            urllib.request.urlretrieve(url, os.path.join(dataset_path, day.strftime('%Y%m%d') + '.nc4'))
            break
        except HTTPError as e:
            log_method.exception("There's an HTTP error with the download file: {}".format(e))
            log_method.info(f"Retrying download: {url}")
        except IncompleteRead as e:
            log_method.exception("There's a read error with the download file: {}".format(e))
            log_method.info(f"Retrying download: {url}")
        except BaseException as e:
            log_method.exception("There's an error with the download file: {}".format(e))
            log_method.info(f"Retrying download: {url}")
        finally:
            tries -= 1
            if not tries:
                log_method.error(f"Unable to download file: {url}")


def process_netcdf4(nc_file, latitude, longitude, variables):
    """
    Variables retrieved from the netcdf4 files are stored and available to be processed
    by a pandas dataframe in it's header.
    :param nc_file:
    :param latitude:
    :param longitude:
    :param variables:
    :return:
    """

    ll_finder = np.sqrt(
        (nc_file['lat'][:][:] - latitude) ** 2 + (nc_file['lon'][:][:] - longitude) ** 2)
    y, x = np.unravel_index(ll_finder.argmin(), ll_finder.shape)
    i = 0

    request = datetime.datetime.strptime((' '.join((nc_file['time'].units.split(' ')[2:]))), "%Y-%m-%d %H:%M:%S")
    request = int(datetime.datetime.timestamp(pytz.utc.localize(request)))

    weather_data = []
    while i < len(nc_file['time']):
        ts = int(datetime.datetime.timestamp(pytz.utc.localize(num2date(nc_file['time'][i], nc_file['time'].units))))
        # The keys from this block of code may be changed
        aux_json = dict()
        for v in variables:
            aux_json[v] = float(nc_file[v][i, y, x])

        aux_json["datetime"] = ts
        aux_json["request"] = request
        aux_json["latitude"] = latitude
        aux_json["longitude"] = longitude
        weather_data.append(aux_json)
        i += 1

    nc_file.close()
    return weather_data
