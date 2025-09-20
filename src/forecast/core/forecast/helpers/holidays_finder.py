import numpy as np
import pandas as pd

from typing import Union
from dateutil import relativedelta

from core.forecast.helpers.holidays import Portugal as HolyPortugal
from core.forecast.helpers.holidays import France as HolyFrance
from core.forecast.helpers.holidays_constants import (
    RELEVANT_HOLIBRIDGES,
    RELEVANT_BRIDGEDAYS,
    RELEVANT_HOLI2BRIDGEDAYS,
    IMPORTANT_HOLIDAYS
)


class HolyFinder:
    def __init__(self,
                 launch_time: pd.Timestamp,
                 country: str = 'portugal',
                 lookback_year: int = 5):
        curr_year = launch_time.year
        year_range = [x for x in range(curr_year - lookback_year, curr_year + 2)] # noqa
        self.country = country.lower()

        if self.country == 'portugal':
            self.national_holidays = HolyPortugal(years=year_range)
        elif self.country == 'france':
            self.national_holidays = HolyFrance(years=year_range)

        # Load relevant metadata about holidays/bridgedays
        self.__load_holidays_bridgedays_metadata()

        # Find bridge days:
        self.__calc_bridgedays()

    def __load_holidays_bridgedays_metadata(self):
        # For christmas, all weekdays can possibly have a bridgeday before/after # noqa
        # for other days, only 2- Tuesday and 4- Thursday
        self.relevant_holi2bridgedays: dict = RELEVANT_HOLI2BRIDGEDAYS[self.country]  # noqa

        # For christmas, all the weekdays are valid bridgedays,
        # for other days, only 1- Monday and 5-Friday
        self.relevant_bridgedays: dict = RELEVANT_BRIDGEDAYS[self.country]

        # Important bridges (for other bridges there are no post-processing):
        self.relevant_holybridges: dict = RELEVANT_HOLIBRIDGES[self.country]

        # Important holidays (conditional filter in self.holidays_fix)
        self.important_holidays: dict = IMPORTANT_HOLIDAYS[self.country]

    def __calc_bridgedays(self):
        # Do not consider weekend holidays for bridgedays calculation
        # e.g. holiday at sunday should now cause a bridge at monday
        holidays_to_consider = {
            day: description
            for day, description in self.national_holidays.items()
            if day.strftime("%A") not in ["Saturday", "Sunday"]
        }

        # One day before bridges:
        bridge_before = {(day - pd.DateOffset(days=1)).date(): description
                         for day, description in holidays_to_consider.items()
                         if description in self.relevant_holybridges
                         and (day.strftime("%A")
                              in (self
                                  .relevant_holi2bridgedays
                                  .get(description,
                                       self.relevant_holi2bridgedays[
                                           "other"])))
                         }

        # One day after bridges:
        bridge_after = {(day + pd.DateOffset(days=1)).date(): description
                        for day, description in holidays_to_consider.items()
                        if description in self.relevant_holybridges
                        and (day.strftime("%A")
                             in (self
                                 .relevant_holi2bridgedays
                                 .get(description,
                                      self.relevant_holi2bridgedays["other"])))
                        }

        # Store all potential bridgedays:
        self.bridgedays = {}
        self.bridgedays.update(bridge_after)
        self.bridgedays.update(bridge_before)

        # Filter bridgedays according to self.relevant_bridgedays weekdays:
        self.bridgedays = {
            day: description
            for day, description in self.bridgedays.items()
            if day.strftime("%A") in self.relevant_bridgedays.get(description, self.relevant_bridgedays["other"])} # noqa

    def is_holiday(self, date: pd.Timestamp):
        ref_date = date.date()
        return ref_date in self.national_holidays.keys()

    def is_bridge(self, date: pd.Timestamp):
        ref_date = date.date()
        return ref_date in self.bridgedays.keys()

    def holidays_in_time_period(self,
                                start: pd.Timestamp,
                                end: pd.Timestamp):
        holidays_in_period = self.get_holidays_in_period(start, end)
        bridge_in_period = self.get_bridges_in_period(start, end)
        if len(holidays_in_period) + len(bridge_in_period) > 0:
            return True
        else:
            return False

    def get_holidays_in_period(self,
                               start: pd.Timestamp,
                               end: pd.Timestamp):
        holidays_in_period = [day for day in self.national_holidays.keys()
                              if start.date() <= day <= end.date()]
        return holidays_in_period

    def get_bridges_in_period(self, start, end):
        bridge_in_period = [day for day in self.bridgedays.keys()
                            if start.date() <= day <= end.date()]
        return bridge_in_period

    def find_holidays_analogous(self,
                                dates: Union[pd.DatetimeIndex, list],
                                full_historical: pd.DataFrame,
                                col_name: str = "forecast",
                                target: str = "real"):
        # -- Full Historical Data (without NaN)
        # and holiday/bridge classification (bool)
        data_historical = full_historical[[target, "is_holy", "is_bridge"]].copy().dropna() # noqa

        # -- Find max/min historical reference for past 5 months
        # this references will prevent retrieving holidays for points
        # where consumption increases compared to the
        # past year and no longer are valid references for current dates
        min_value = data_historical.loc[dates[0] - pd.DateOffset(months=5): dates[0], target].min() # noqa
        max_value = data_historical.loc[dates[0] - pd.DateOffset(months=5): dates[0], target].max() # noqa

        # -- Filter available holidays historical:
        holidays_historical = data_historical.loc[data_historical["is_holy"], :] # noqa

        # -- Initialize obj. for holidays forecast:
        values_df = pd.DataFrame(index=dates, columns=[col_name])

        # -- Iterate through every timestamp in the forecast horizon
        # and finds analogous values for each holiday
        for d in dates:
            # -- Holiday Description/Name (e.g. Christmas, New Year, etc.)
            holiday_description = self.national_holidays.get(d.date())

            if holiday_description not in self.important_holidays:
                important_holiday = True
                # All holidays 5 years range:
                similar_dates = list(self.national_holidays.keys())

                # Holidays in same weekday:
                similar_dates = [x for x in similar_dates if x.strftime("%A") == d.strftime("%A")] # noqa

                # Holidays in close range (max 2 month):
                similar_dates = [x for x in similar_dates if (0 < (d.date() - x).days <= 60)] # noqa
            else:
                important_holiday = False
                # -- Dates with holidays for the same festive date:
                similar_dates = [day for day, description
                                 in self.national_holidays.items()
                                 if description == holiday_description]

            # ---- First filter mask: Check which of the dates in "similar_dates" exist in the holidays historical data: # noqa
            mask1 = np.isin(holidays_historical.index.date, similar_dates)
            # ---- Second filter mask: Filter search values according to current datetime "time" component: # noqa
            mask2 = holidays_historical.index.time == d.time()

            # ---- Find similar holidays historical references:
            similar_values = holidays_historical.loc[mask1 & mask2, target].dropna() # noqa
            similar_values = [x for x in similar_values.values
                              if (x >= min_value) and (x < max_value)]

            if (len(similar_values) == 0) and important_holiday:
                # only applied for self.relevant_holidays
                # -- If no values found. 1st, tries to find Christmas days or other similar holidays in near timespan # noqa
                # --- Get the description of nearby holidays (e.g. for the New Years eve finds the christmas holiday) # noqa
                nearby_holidays = [self.national_holidays.get(day)
                                   for day in self.national_holidays.keys()
                                   if (0 < (d.date() - day).days <= 15)]

                # --- Get the dates of nearby holidays:
                similar_dates = [day for day, description
                                 in self.national_holidays.items()
                                 if description in nearby_holidays]

                mask1 = np.isin(holidays_historical.index.date, similar_dates)
                mask2 = holidays_historical.index.time == d.time()
                similar_values = (holidays_historical
                                  .loc[mask1 & mask2, target]
                                  .dropna())
                similar_values = [x for x in similar_values.values
                                  if (x >= min_value) and (x < max_value)]

            if len(similar_values) == 0:
                # -- As last resource, averages the closest sundays
                # from the date of interest:
                similar_values = []
                for i in range(1, 4):
                    previous_sunday_date = d + relativedelta.relativedelta(weekday=relativedelta.SU(-i)) # noqa
                    try:
                        similar_values.append(
                            data_historical.loc[previous_sunday_date, target])
                    except KeyError:
                        similar_values.append(np.NaN)

            # ---- Averages Similar Dates in order to create estimated values
            # for the holidays in horizon:
            values_df.loc[d, col_name] = np.nanmean(similar_values)

        return values_df.dropna()

    def find_bridge_analogous(self,
                              dates: Union[pd.DatetimeIndex, list],
                              full_historical: pd.DataFrame,
                              col_name: str = "forecast",
                              target: str = "real"):
        # -- Full Historical Data (without NaN)
        # and holiday/bridge classification (bool)
        data_historical = full_historical.copy().dropna()
        min_value = data_historical.loc[dates[0] - pd.DateOffset(months=5): dates[0], target].min() # noqa
        max_value = data_historical.loc[dates[0] - pd.DateOffset(months=5): dates[0], target].max() # noqa
        # -- Filter available bridges in the historical data:
        bridges_historical = data_historical.loc[data_historical["is_bridge"], :] # noqa
        values_df = pd.DataFrame(index=dates, columns=[col_name])

        for d in dates:
            # -- Bridge Description/Name (e.g. Christmas, New Year, etc.)
            bridge_description = self.bridgedays.get(d.date())

            # -- Find similar bridges (on same weekday)
            similar_dates = [day for day, description
                             in self.bridgedays.items()
                             if (description == bridge_description)
                             and (int(day.strftime("%w")) == int(d.strftime("%w")))] # noqa

            # Bridgedays in close range (max 1 month):
            similar_dates = [x for x in similar_dates
                             if (0 < (d.date() - x).days <= 30)]

            mask1 = np.isin(bridges_historical.index.date, similar_dates)
            mask2 = bridges_historical.index.time == d.time()
            similar_values = bridges_historical.loc[mask1 & mask2, target].dropna() # noqa
            similar_values = [x for x in similar_values.values
                              if (x >= min_value) and (x <= max_value)]

            if len(similar_values) == 0:
                # -- As last resource, averages the closest 4 saturdays
                similar_values = []
                for i in range(1, 4):
                    previous_saturday_date = d + relativedelta.relativedelta(weekday=relativedelta.SA(-i)) # noqa
                    try:
                        similar_values.append(data_historical.loc[previous_saturday_date, target]) # noqa
                    except KeyError:
                        similar_values.append(np.NaN)

            values_df.loc[d, col_name] = np.nanmean(similar_values)

        return values_df.dropna()
