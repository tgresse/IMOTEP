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

import numpy as np

def generate_airzone_list(airzone_def_dict_list):
    valid_airzone_type_list = ['fixed', 'weather', 'balanced']

    # initialize storage lists
    fixed_airzone_list = []
    weather_airzone_list = []
    balanced_airzone_list = []
    name_list = []

    for airzone_def_dict in airzone_def_dict_list:
        # store the name and check if it does not already exist
        name = airzone_def_dict['name']
        if name in name_list:
            raise ValueError(f"'{name}' value for parameter 'name' in airzone_dict is used more than once.")
        name_list.append(name)

        # create the airzone objects depending on their types
        airzone_type = airzone_def_dict['type']
        if airzone_type == 'fixed':
            airzone = FixedAirZone(*[airzone_def_dict[k] for k in ('name', 'hc_type', 'ta_value', 'hc_value')])
            fixed_airzone_list.append(airzone)
        elif airzone_type == 'weather':
            airzone = WeatherAirZone(*[airzone_def_dict[k] for k in ('name', 'hc_type', 'ta_value', 'hc_value', 'wind_factor_args')])
            weather_airzone_list.append(airzone)
        elif airzone_type == 'balanced':
            airzone = BalancedAirZone(*[airzone_def_dict[k] for k in ('name', 'hc_type', 'ta_value', 'hc_value', 'capacity', 'volume', 'internal_load', 'effective_area_args', 'wind_factor_args', 'inflow_airzone_name')])
            balanced_airzone_list.append(airzone)
        else:
            raise ValueError(f"'{airzone_type}' value for parameter 'type' in airzone_dict is not a valid. Expected value: {valid_airzone_type_list}")

    # store the airzone lists in a dictionary
    # weather: air temperature and wind speed directly extracted from weather
    # balanced: air temperature computed with a heat balance
    # dynamic: the air temperature changes at every timestep
    # all: all airzones
    airzone_list_dict = {'weather' : weather_airzone_list,
                         'balanced': balanced_airzone_list,
                         'dynamic' : weather_airzone_list + balanced_airzone_list,
                         'all'     : fixed_airzone_list + weather_airzone_list + balanced_airzone_list}

    return airzone_list_dict

def correlation(wind_speed):
    return 4. + 4. * wind_speed

class AirZone:

    state_variable_list = ['hc', 'temperature', 'wind_speed']

    def __init__(self, name, hc_type, ta_value, hc_value):
        valid_hc_type_list = ['fixed', 'correlation']

        # hc_type should be in the valid list
        if hc_type not in valid_hc_type_list:
            raise ValueError(f"'{hc_type}' value for parameter 'hc_type' in airzone_def_dict is not a valid. Expected value: {valid_hc_type_list}")
        self.hc_type = hc_type

        # if fixed hc, it should be a positive float numeric
        if self.hc_type == 'fixed':
            if hc_value is not None or hc_value > 0.:
                self.hc = hc_value
            else:
                raise ValueError(f"'{hc_value}' value for parameter hc_value' in airzone_def_dict is not a valid. Expected value: a positive float number")

        self.name = name
        # initialize surface list that will be filled up elsewhere
        self.surface_list = []
        # intialize variables
        self.temperature = ta_value
        self.wind_speed = .0
        self.relative_humidity = 60.

    def connect_surface(self, surface):
        self.surface_list.append(surface)


class FixedAirZone(AirZone):

    def __init__(self, name, hc_type, ta_value, hc_value):
        super().__init__(name, hc_type, ta_value, hc_value)
        if self.hc_type == 'correlation':
            raise ValueError(f"'{hc_type}' value for parameter 'hc_type' in fixed airzone_def_dict is not a valid. Expected value: 'fixed'")


class WeatherAirZone(AirZone):

    def __init__(self, name, hc_type, ta_value, hc_value, wind_factor_args):
        super().__init__(name, hc_type, ta_value, hc_value)
        self.wind_factor_wdl = WindDirectionList(*wind_factor_args)

    def connect_weather(self, weather):
        self.weather = weather

    def initialize_timestep(self):
        # set temperature, wind speed ad relative humidity from weather data
        # this function is called once weather temperature was updated
        self.temperature = self.weather.air_temperature
        wind_factor = self.wind_factor_wdl.get_value_for_closest_wind_direction(self.weather.wind_direction)
        self.wind_speed = wind_factor * self.weather.wind_speed
        self.relative_humidity = self.weather.relative_humidity
        # compute hc
        if self.hc_type == 'correlation':
            self.hc = correlation(self.wind_speed)


class BalancedAirZone(AirZone):

    def __init__(self, name, hc_type, ta_value, hc_value, capacity, volume, internal_load, effective_area_args, wind_factor_args, inflow_airzone_name):
        super().__init__(name, hc_type, ta_value, hc_value)
        self.capacity = capacity
        self.volume = volume
        self.internal_load = internal_load
        self.effective_area_wdl = WindDirectionList(*effective_area_args)
        self.wind_factor_wdl = WindDirectionList(*wind_factor_args)
        self.inflow_airzone_name = inflow_airzone_name

    def set_dt(self, dt):
        self.dt = dt

    def connect_weather(self, weather):
        self.weather = weather

    def connect_inflow_airzone(self, inflow_airzone):
        self.inflow_airzone = inflow_airzone

    def initialize_temperature(self, temperature):
        self.temperature = temperature

    def initialize_timestep(self):
        wind_factor = self.wind_factor_wdl.get_value_for_closest_wind_direction(self.weather.wind_direction)
        self.wind_speed = wind_factor * self.weather.wind_speed
        self.relative_humidity = self.weather.relative_humidity
        self.previous_temperature = self.temperature
        # compute hc
        if self.hc_type == 'correlation':
            self.hc = correlation(self.wind_speed)

    def compute_temperature(self):
        sum_hc_si = 0.
        sum_hc_si_tsi = 0.
        for surface in self.surface_list:
            temp_arr = np.array([facet.temperature for facet in surface.facet_list])
            hc_si_arr = self.hc * surface.mesh['cell_areas']
            sum_hc_si = np.sum(hc_si_arr)
            sum_hc_si_tsi = np.sum(hc_si_arr * temp_arr)

        equiv_area = self.effective_area_wdl.get_value_for_closest_wind_direction(self.weather.wind_direction)
        capa_flowrate = self.capacity * self.wind_speed * equiv_area
        capa_flowrate_tain = capa_flowrate * self.inflow_airzone.temperature
        capa_vol_over_dt = self.capacity * self.volume / self.dt

        self.temperature = (capa_vol_over_dt * self.previous_temperature + sum_hc_si_tsi + capa_flowrate_tain + self.internal_load) / \
                           (capa_vol_over_dt + sum_hc_si + capa_flowrate)


class WindDirectionList:

    def __init__(self, wd_list, value_list):
        if wd_list is None:
            self.wd_arr = np.linspace(0, 360, len(value_list) + 1)
        else:
            if wd_list[0] != 0 or wd_list != sorted(wd_list):
                raise ValueError("The wind directions must be sorted in ascending order starting at 0°.")
            if len(value_list) != len(wd_list):
                raise ValueError("Number of wind factors and wind directions are not equal.")
            self.wd_arr = np.array(wd_list + [360])
        # repeat last value for 360°
        value_list = list(value_list) + [value_list[0]]
        self.value_arr = np.array(value_list)

    def get_value_for_closest_wind_direction(self, wd):
        index = np.argmin(np.abs(self.wd_arr - wd))
        return self.value_arr[index]


if __name__ == '__main__':
    from _input_shelter_old import airzone_def_dict_list

    airzone_dict = generate_airzone_list(airzone_def_dict_list)

    # wd_list = [90, 0, 180, 270]
    # value_list = [1, 5, 3, 4]
    # wdl = WindDirectionList(wd_list, value_list)
    # val = wdl.get_value_for_closest_wind_direction(30)