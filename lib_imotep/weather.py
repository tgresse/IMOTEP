# Copyright CETHIL UMR 5008 - INSA Lyon (2025)
#
# teddy.gresse@gmail.com
# damien.david@insa-lyon.fr
#
# IMOTEP is a micro-meteorological model that simulates the thermal environments
# of urban scenes that may contain built shelters, building overhangs or trees,
# and evaluates outdoor thermal comfort.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import datetime
import numpy as np
import pandas as pd
from epw import epw
import pvlib.solarposition as spa
from pvlib.location import Location

SIGMA = 5.67e-8

class Weather:
    """
    A class representing the weather conditions.
    """

    state_variable_list = ['air_temperature', 'global_horizontal_radiation', 'direct_normal_radiation',  'diffuse_horizontal_radiation', 'relative_humidity', 'sky_temperature', 'wind_direction', 'wind_speed']

    def __init__(self, df, year, latitude, longitude, altitude, tz_info, system_orientation):
        """
        Initialize the Weather object.

        :param df: Pandas DataFrame containing weather data.
        :param year: Year of the weather data.
        :param latitude: Latitude of the location.
        :param longitude: Longitude of the location.
        :param altitude: Altitude of the location.
        :param tz_info: Timezone offset from UTC.
        :param system_orientation: Orientation of the system.
        """
        self.df = df
        self.year = year
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.tz_info = tz_info
        self.system_orientation = system_orientation
        self.sun = Sun(latitude, longitude, system_orientation)

    @classmethod
    def agile_constructor(cls, weather_def_dict, general_dict):
        """
        Generate a Weather instance based on the specified weather type.
        :param weather_def_dict: Dictionary containing weather data type and parameters.
        :param general_dict: Dictionary containing general parameters.
        :return: Weather instance.
        """
        w_type = weather_def_dict['type']
        valid_weather_type_list = ['weather_file', 'reconstructed']

        if w_type == 'weather_file':
            weather = cls.from_epw(weather_def_dict['filepath'], general_dict['system_orientation'])

        elif w_type == 'reconstructed':
            weather = cls.from_reconstruction(*[weather_def_dict[k] for k in ('location_args', 'weather_data_args')],
                                              *[general_dict[k] for k in ('target_date', 'warmup_days_num', 'dt', 'system_orientation')])

        elif w_type == 'marty':
            weather = cls.from_marty_spreadsheet(*[weather_def_dict[k] for k in ('filepath', 'location_args', 'dtas')],
                                                 *[general_dict[k] for k in ('target_date', 'warmup_days_num', 'dt', 'system_orientation')])

        else:
            raise ValueError(f"'{w_type}' value for parameter 'type' in weather_def_dict is not a valid. Expected value: {valid_weather_type_list}")

        return weather

    @classmethod
    def from_epw(cls, filepath, system_orientation):
        """
        Create a Weather object using EPW weather file.

        :param filepath: path to the EPW file
        :param system_orientation: Orientation of the system.
        :return: Weather instance initialized with EPW weather file.
        """
        f = epw()
        f.read(filepath)
        # extract necessary columns
        df = f.dataframe[['Year', 'Month', 'Day', 'Hour', 'Minute',
                          'Dry Bulb Temperature', 'Direct Normal Radiation',
                          'Diffuse Horizontal Radiation', 'Horizontal Infrared Radiation Intensity',
                          'Wind Speed', 'Wind Direction', 'Relative Humidity']].copy()
        # extract location metadata
        latitude, longitude, timezone, altitude = map(float, f.headers['LOCATION'][5:9])
        timezone = int(timezone)
        # determine year type (leap or non-leap)
        year = 1996 if len(df) == 8784 else 1995 if len(df) == 8760 else None
        if year is None:
            raise ValueError('Invalid data length: Cannot determine leap year status.')
        # construct datetime index
        df.insert(0, "datetime", pd.to_datetime(dict(year=year, month=df.Month, day=df.Day, hour=df.Hour)))
        df.set_index("datetime", inplace=True)
        # drop unnecessary columns
        df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
        # set proper timezone
        tz_info = f'Etc/GMT{-timezone:+d}'
        df.index = df.index.tz_localize(tz_info, ambiguous='NaT')
        # compute the sky temperature column
        df['Sky Temperature'] = (df['Horizontal Infrared Radiation Intensity'] / SIGMA) ** 0.25 - 273.15
        return cls(df, year, latitude, longitude, altitude, tz_info, system_orientation)


    @classmethod
    def from_marty_spreadsheet(cls, filepath, location_args, dtas, target_date_tuple, warmup_days_num, dt, system_orientation):
        latitude, longitude, altitude, tz_info = location_args

        if tz_info not in ('US/Arizona', 'MST') :
            raise ValueError("  To set weather from marty spreadsheet, tz_info must be 'US/Arizona' or 'MST'.")

        if dt % 300 != 0:
            raise ValueError(" To set weather from marty spreadsheet, dt must be a multiple of 300s (5min).")

        target_date = pd.to_datetime(datetime.datetime(*target_date_tuple)).tz_localize(tz=tz_info)
        date_start = target_date - datetime.timedelta(days=warmup_days_num)
        date_end = target_date + datetime.timedelta(days=1)

        #  Load MaRTy
        marty_df = pd.read_excel(filepath)
        marty_df.index = pd.to_datetime(marty_df['ID = KPHX'], errors='coerce')
        marty_df.index = marty_df.index.tz_localize(tz_info, ambiguous='NaT')
        marty_df.sort_index(inplace=True)

        # check if the target_date and the first date of the marty dataframe are the same
        if target_date != marty_df.index[0]:
            raise ValueError(f' Inconsistent dates between the marty spreadsheet and the simulation target date (target date:{target_date} / marty first date:{marty_df.index[0]}).')

        marty_df = marty_df[target_date:date_end]

        # 00:00..24:00 (exclusive) at 1-min for clearsky radiation
        date_range = pd.date_range(start=target_date, end=date_end, freq='60s', tz=tz_info, inclusive='left')

        # --- Clear-sky radiation for that day ---
        loc = Location(latitude, longitude, tz_info, altitude)
        weather_df = loc.get_clearsky(date_range)

        # Base df at user timestep
        df = weather_df[['ghi', 'dhi', 'dni']].resample(f'{dt}s').first()
        df = df.rename(columns={'ghi': 'Global Horizontal Radiation',
                                'dhi': 'Diffuse Horizontal Radiation',
                                'dni': 'Direct Normal Radiation'})

        met = marty_df[['TMP ° C', 'RELH %', 'SKNT m/s', 'DRCT °']].resample(f'{dt}s').first()
        met.columns = ['Dry Bulb Temperature', 'Relative Humidity', 'Wind Speed', 'Wind Direction']

        # Interpolate scalar columns
        scalar_cols = ['Dry Bulb Temperature', 'Relative Humidity', 'Wind Speed']
        met[scalar_cols] = met[scalar_cols].interpolate(method='time', limit_direction='both')

        def _interpolate_wind_direction(wd: pd.Series) -> pd.Series:
            """Circular interpolation of wind direction in degrees (0..360)."""
            if wd.isna().all():
                return wd  # nothing to do
            # Convert to unit vectors
            rad = np.deg2rad(wd)
            x = np.cos(rad)
            y = np.sin(rad)
            # Interpolate components along the index
            method = 'time' if isinstance(wd.index, pd.DatetimeIndex) else 'index'
            x = x.interpolate(method=method, limit_direction='both')
            y = y.interpolate(method=method, limit_direction='both')
            # Recompose angle
            ang = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
            # Preserve original NaNs if the whole neighborhood was NaN
            ang[wd.isna() & (x.isna() | y.isna())] = np.nan
            return ang

        # Interpolate circular column
        met['Wind Direction'] = _interpolate_wind_direction(met['Wind Direction'])

        # Align by nearest timestamp to the dt grid
        met_aligned = met.reindex(df.index, method='nearest')

        # Join and compute sky temperature
        df = df.join(met_aligned)
        df['Wind Speed'] = df['Wind Speed'].clip(lower=0.2)
        df['Sky Temperature'] = df['Dry Bulb Temperature'] - dtas

        # --- Repeat warmup days + target (robust to DST) ---
        num_days = warmup_days_num + 1
        dff = pd.concat([df] * num_days, axis=0)

        full_index = pd.date_range(start=date_start, end=date_end, freq=f'{dt}s', tz=tz_info)
        dff = pd.concat([dff, df.iloc[[-1]]], axis=0)  # add 00:00 at the end of the last day
        dff.index = full_index

        return cls(dff, target_date.year, latitude, longitude, altitude, tz_info, system_orientation)


    @classmethod
    def from_reconstruction(cls, location_args, weather_data_args, target_date_tuple, warmup_days_num, dt, system_orientation):
        """
        Create a Weather object reconstructed from clear sky conditions.

        :param location_args: Tuple (latitude, longitude, altitude, timezone).
        :param weather_data_args: Tuple (max air temp, delta max-min air temp, dew point temperature, wind speed, wind direction).
        :param target_date_tuple: Target date as a tuple (year, month, day).
        :param date_list: DatetimeIndex of simulation datetimes
        :param dt: Timestep interval in seconds.
        :param system_orientation: Orientation of the system.
        :return: Weather instance.
        """
        latitude, longitude, altitude, tz_info = location_args
        ta_max, dta_minmax, dtas, tdp, ws, wd = weather_data_args
        ta_min = ta_max - dta_minmax

        target_date = pd.to_datetime(datetime.datetime(*target_date_tuple)).tz_localize(tz=tz_info)
        date_start = target_date - datetime.timedelta(days=warmup_days_num)
        date_end = target_date + datetime.timedelta(days=1)

        # create a DatetimeIndex for one day (target date) with a one minute timestep -> 00:00..24:00 (next midnight EXCLUDED)
        # Note: local time including summer time shift
        date_range = pd.date_range(start=target_date, end=date_end, freq='60s', tz=tz_info, inclusive='left')

        # get clear sky solar heat fluxes -> create weather dataframe with 'ghi', 'dni', and 'dhi' columns
        loc = Location(latitude, longitude, tz_info, altitude)
        weather_df = loc.get_clearsky(date_range)

        # compute sunrise/sunset for the *target day* using local noon to avoid rollover; SPA expects UTC
        local_noon = target_date + pd.Timedelta(hours=12)
        noon_utc = pd.DatetimeIndex([local_noon.tz_convert('UTC')])
        sol_times_utc = spa.sun_rise_set_transit_spa(noon_utc, latitude, longitude)
        sol_times = sol_times_utc.tz_convert(tz_info)

        # round to minute and align to your minute index
        sunrise_time = sol_times['sunrise'].iloc[0].round('min')
        sunset_time  = sol_times['sunset' ].iloc[0].round('min')
        sunrise_time = weather_df.index.asof(sunrise_time)
        sunset_time  = weather_df.index.asof(sunset_time)

        # extract the clear-sky dataframe from sunrise to sunset (inclusive in pandas)
        weather_srs_df = weather_df.loc[sunrise_time:sunset_time].copy()

        # lengths (integers) of each daily segment (must sum to 1440)
        n_before_sunrise = int((sunrise_time - target_date).total_seconds() // 60)
        n_day            = int((sunset_time  - sunrise_time).total_seconds() // 60)
        n_after_sunset   = 1440 - n_before_sunrise - n_day

        # -- compute air temperature --
        # - daytime temperature calculation -
        flux_toa = 1366  # [W/m²] Flux on top of the atmosphere
        kx = weather_srs_df['ghi'].cumsum() / (flux_toa * np.arange(1, len(weather_srs_df) + 1))
        weather_srs_df['kx'] = kx
        kxmax = kx.max()
        kxmax_time = weather_srs_df['kx'].idxmax()
        slope_beforekxmax = (ta_max - ta_min) / kxmax
        slope_afterkxmax = 1.7 * slope_beforekxmax

        # build full daytime curve from sunrise..sunset (inclusive), then drop the sunset sample
        ta_day_full = np.where(
            weather_srs_df.index <= kxmax_time,
            ta_min + slope_beforekxmax * weather_srs_df['kx'],
            ta_max - slope_afterkxmax * (kxmax - weather_srs_df['kx'])
        )
        ta_sunset = float(ta_day_full[-1])
        ta_day_arr = ta_day_full[:-1]  # sunrise .. minute before sunset (length == n_day)

        # - nighttime temperature calculation - (sunset -> next sunrise)
        len_night = n_before_sunrise + n_after_sunset  # exact integer minutes

        # integer partition of len_night into 1/2, 1/3, 1/6 so they sum exactly
        L1 = len_night // 2
        L2 = len_night // 3
        L3 = len_night - L1 - L2  # ensures L1+L2+L3 == len_night

        slope_night = (ta_min - ta_sunset) / (13 / 18 * len_night)
        ta_night_1 = ta_sunset + slope_night * (len_night / 2)
        ta_night_2 = ta_night_1 + (slope_night / 2) * (len_night / 3)

        seg1 = np.linspace(ta_sunset,  ta_night_1, L1, endpoint=False)
        seg2 = np.linspace(ta_night_1, ta_night_2, L2, endpoint=False)
        seg3 = np.linspace(ta_night_2, ta_min,     L3, endpoint=True)
        ta_night_arr = np.concatenate([seg1, seg2, seg3])  # length == len_night

        # combine day and night into a 24h series aligned to the index (length 1440)
        pre_sunrise = ta_night_arr[n_after_sunset:]  # midnight -> sunrise (length n_before_sunrise)
        post_sunset = ta_night_arr[:n_after_sunset]  # sunset -> midnight (length n_after_sunset)

        weather_df['Dry Bulb Temperature'] = np.concatenate([pre_sunrise, ta_day_arr, post_sunset])

        # change timestep to the user-defined timestep and rename columns
        df = weather_df[['ghi', 'dhi', 'dni', 'Dry Bulb Temperature']].resample(f'{dt}s').first()
        df.rename(columns={'ghi': 'Global Horizontal Radiation',
                           'dhi': 'Diffuse Horizontal Radiation',
                           'dni': 'Direct Normal Radiation'}, inplace=True)

        # add missing variables
        df['Relative Humidity'] = 100 * np.exp((17.625 * tdp) / (243.04 + tdp)) / np.exp((17.625 * df[
                                  'Dry Bulb Temperature']) / (243.04 + df['Dry Bulb Temperature']))
        df['Sky Temperature'] = df['Dry Bulb Temperature'] - dtas
        df['Wind Speed'] = ws
        df['Wind Direction'] = wd

        # extend dataset to cover full simulation period by repeating data (keeping user-defined timestep)
        num_days = warmup_days_num + 1
        full_index = pd.date_range(start=date_start, end=date_end, freq=f'{dt}s', tz=tz_info)
        dff = pd.concat([df] * num_days, axis=0)  # duplicate num_days times the reference day from 00:00 to 23:XX
        dff = pd.concat([dff, df.iloc[[-1]]], axis=0)  # add 00:00 at the end of the last day
        dff.index = full_index

        return cls(dff, target_date.year, latitude, longitude, altitude, tz_info, system_orientation)


    def load_variables(self, datetime_i):
        """
        Load the weather variables at a given datetime and store the variable in object attributes.

        :param datetime_i: Current simulation datetime.
        """
        self.wind_speed = self._get_variable(datetime_i, 'Wind Speed')
        self.wind_direction = self._get_variable(datetime_i, 'Wind Direction')
        # rotate the wind direction counterclockwise by system_orientation degrees
        # the system_orientation defines the rotation of the system in clockwise direction, therefore it is equivalent
        # to a counterclockwise rotation of the wind direction
        self.wind_direction -= self.system_orientation
        self.sky_temperature = self._get_variable(datetime_i, 'Sky Temperature')
        self.diffuse_horizontal_radiation = self._get_variable(datetime_i, 'Diffuse Horizontal Radiation')
        self.air_temperature = self._get_variable(datetime_i, 'Dry Bulb Temperature')
        self.relative_humidity = self._get_variable(datetime_i, 'Relative Humidity')
        self.direct_normal_radiation = self._get_variable(datetime_i, 'Direct Normal Radiation')
        self.global_horizontal_radiation = self._get_variable(datetime_i, 'Global Horizontal Radiation')

    def compute_sun_direction_list(self, target_date_list):
        sunray_direction_list = []
        for date in target_date_list:
            self.load_sun_variables(date)
            sunray_direction_list.append(self.sun.ray_direction_arr)

        return sunray_direction_list

    def load_sun_variables(self, datetime_i):
        """
        Load the sun variables at a given datetime.

        :param date: Current simulation datetime.
        """
        self.sun.update_variables(datetime_i)

    def _get_variable(self, datetime_i, column):
        """
        Retrieve a weather variable at a given datetime with linear interpolation.

        :param datetime_i: Datetime for the weather data retrieval.
        :param column: Weather variable column name.
        :return: Interpolated weather variable value.
        """
        # Force the year (needed in case of epw fed weather data)
        datetime_i_modif = datetime_i.replace(year=self.year)
        if not (self.df.index[0] <= datetime_i_modif <= self.df.index[-1]):
            raise ValueError(f"Datetime '{datetime_i_modif}' is outside the range of the weather dataframe.")

        # extract the column that corresponds to the variable
        data_ser = self.df[column]

        # if the date belongs to the weather dataframe index, return the data
        if datetime_i_modif in data_ser.index:
            return data_ser.at[datetime_i_modif]

        # otherwise, interpolate (needed in case of epw fed weather data)
        new_index = data_ser.index.insert(np.searchsorted(data_ser.index, datetime_i_modif), datetime_i_modif)
        data_ser_interp = data_ser.reindex(new_index).interpolate(method='index')
        return data_ser_interp.at[datetime_i_modif]


class Sun:
    """
    A class representing the sun's position and direct normal radiation.
    """

    def __init__(self, latitude, longitude, system_orientation):
        """
        Initialize the Sun object with location data.

        :param latitude: Latitude of the location.
        :param longitude: Longitude of the location.
        :param system_orientation: Orientation of the system (for potential rotation adjustments).
        """
        self.latitude = latitude
        self.longitude = longitude
        self.system_orientation = system_orientation
        # initialize dynamic variables
        self.ray_direction_arr = np.zeros(3)

    def update_variables(self, date):
        """
        Update solar radiation and sun direction vector based on the datetime.

        :param date: Current simulation datetime.
        """
        # compute the sun position based on the spa library
        df = spa.spa_python(date, self.latitude, self.longitude)
        # rotate the sun vector counterclockwise by system_orientation degrees
        # the system_orientation defines the rotation of the system in clockwise direction, therefore it is equivalent
        # to a counterclockwise rotation of the sun
        # info: the default north direction = +Y
        azimuth = np.radians(df['azimuth'][date]-self.system_orientation)
        elevation = np.radians(df['elevation'][date])
        # compute base sunray direction vector
        self.ray_direction_arr = -np.array([ np.sin(azimuth) * np.cos(elevation),
                                             np.cos(azimuth) * np.cos(elevation),
                                             np.sin(elevation)])


if __name__ == '__main__':
    from _input_shelter_old import general_dict, weather_def_dict

    weather = Weather.agile_constructor(weather_def_dict, general_dict)

    datetime_test = pd.to_datetime(datetime.datetime(2025, 7, 23, 8, 0)).tz_localize(tz=weather.tz_info)

    weather.load_variables(datetime_test)
    weather.load_sun_variables(datetime_test)

    dt = 900
    date_start = pd.to_datetime(datetime.datetime(2025, 7, 23, 0, 0)).tz_localize(tz=weather.tz_info)
    date_end = date_start + datetime.timedelta(days=1)
    date_range_test = pd.date_range(date_start, date_end, freq=f"{dt}s")

    sunray_direction_list = weather.compute_sun_direction_list(date_range_test)

