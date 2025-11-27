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
import pickle as pkl
from scipy.interpolate import RegularGridInterpolator

from utils import global_timer
from weather import Weather
from airzone import generate_airzone_list, AirZone, FixedAirZone
from panel import generate_panel_list
from surface import Surface
from facet import Facet
from probe import generate_probe_set_list, Probe
from radiative_sys_geom import RadiativeSystemGeometry
from radiative_model import RadiativeModel

class IMOTEP:
    """
    The main class of the IMOTEP solver

    Attributes
    ----------
    general_dict: dict
        dictionary containing the general simulation parameters
    weather : Weather object
        object representing the weather
    radiative_model : RadiativeModel object
        object for radiative calculations
    airzone_list_dict : dict
        dictionary of lists containing airzone objects by type
    surface_list_dict : dict
        dictionary of lists containing surface objects by type
    facet_list_dict : dict
        dictionary of lists containing facet objects by type
    core_list_dict : list
    probe_set_list : list
        list containing probe objects
    output_dict : dict
        dictionary of lists containing the outputs
    """

    def __init__(self, general_dict, weather_def_dict, panel_def_dict_list, airzone_def_dict_list, probe_set_def_list, output_def_dict):
        """
        Initialize the IMOTEP solver
        :param dict general_dict: dictionary containing the general simulation parameters
        :param list panel_def_dict_list: list of dictionaries containing the panel definition parameters
        :param list airzone_def_dict_list: list of dictionaries containing the airzone definition parameters
        :param list probe_set_def_list: list containing the probes definition parameters
        :param dict output_def_dict: dictionary of output definition parameters
        """
        # store dictionaries as attributes
        self.general_dict = general_dict
        self.weather_def_dict = weather_def_dict
        self.panel_def_dict_list = panel_def_dict_list
        self.airzone_def_dict_list = airzone_def_dict_list
        self.probe_set_def_list = probe_set_def_list
        self.output_def_dict = output_def_dict


    def generate(self):
        print(" 1/ Generate the model\n")
        global_timer.start('generate')

        self._generate_model_objects()
        self._generate_radiative_viewfactors_and_prefactors()
        self._generate_radiative_model()
        self._initialize_variables()

        global_timer.stop('generate', print_type='first')

        del self.weather_def_dict
        del self.panel_def_dict_list
        del self.probe_set_def_list
        del self.airzone_def_dict_list
        del self.output_def_dict


    def solve(self):
        """
        Main function to solve the thermal problem
        """
        print("\n 2/ Solve\n")
        global_timer.start('solve')

        # start main loop
        self.solve_stat_list = []
        for datetime_i in self.time_management_dict['date_list']:
            print(f"    {datetime_i}")
            # update matrices at timestep switch
            if datetime_i == self.time_management_dict['switch_timestep_date']:
                self._solve_update_dt()

            # update weather data at current timestep
            self.weather.load_variables(datetime_i)

            # initialize timestep for core props (hc and ta in matrices), cores (prev timestep),
            # airzones (hc) and weather connected facets (temp)
            self._solve_initialize_timestep()

            # compute the net rad sw fluxes on facets
            self.radiative_model.compute_sw_flux(datetime_i)

            # initialize criteria for the pin-pong iterative process
            err = 1e9
            cnt = 0
            # begin the ping-pong process for cells with a balanced core
            while err > self.general_dict['tolerance']:
                err_old = err
                # compute the net rad lw fluxes on facets
                self.radiative_model.compute_lw_flux()

                # send fluxes from facets to cores
                self._solve_external_fluxes_facet_to_core()

                # compute air temperature
                self._solve_compute_balanced_airzone_temperature()

                # compute wall, surface, and air temperatures (using sub-relaxation for stability)
                err = self._solve_iterate_temperatures()

                # send temperatures from cores to facets
                self._solve_temperature_core_to_facet()

                # security
                if cnt > 1000:
                    raise ValueError(f"Too many ping-pong iterations for solving. "
                                     f"Info: datetime={datetime_i}, err={err}, err_old={err_old}, relax_coef={self.general_dict['relax_coef']}, tolerance={self.general_dict['tolerance']}")
                cnt += 1

            self.solve_stat_list.append(cnt)

            if datetime_i >= self.time_management_dict['switch_timestep_date_output']:
                self._solve_compute_metrics_at_probes(datetime_i)
                self._solve_write_outputs()

        print(f"    -> num iter tot: {np.sum(self.solve_stat_list)}")
        print(f"    -> num iter per timestep: {round(np.sum(self.solve_stat_list) / len(self.time_management_dict['date_list']),2)}\n")

        global_timer.stop('solve', print_type='first')

    def save_viewfactors_and_prefactors(self, save_manager):
        if self.general_dict['save_data']:
            global_timer.start('save_vf_and_pf')

            save_folderpath = save_manager.full_path
            print(save_manager.message)

            self.rad_domain_viewfactor_mat.dump(save_folderpath + 'rad_domain_viewfactor_mat.pkl')
            self.rad_domain_prefactor_arr.dump(save_folderpath + 'rad_domain_prefactor_arr.pkl')
            self.rad_domain_partial_obstruction_arr.dump(save_folderpath + 'rad_domain_partial_obstruction_arr.pkl')
            if len(self.probe_set_list) > 0:
                self.rad_probe_viewfactor_mat.dump(save_folderpath + 'rad_probe_viewfactor_mat.pkl')
                self.rad_probe_prefactor_arr.dump(save_folderpath + 'rad_probe_prefactor_arr.pkl')
                self.rad_probe_partial_obstruction_arr.dump(save_folderpath + 'rad_probe_partial_obstruction_arr.pkl')

            global_timer.stop('save_vf_and_pf', print_type='first')


    def save_case(self, save_manager):
        if self.general_dict['save_data']:
            print("\n 3/ Save case\n")
            global_timer.start('save_case')

            save_folderpath = save_manager.full_path
            print(save_manager.message)

            with open(save_folderpath + 'imotep.pkl', 'wb') as data:
                pkl.dump(self, data)

            global_timer.stop('save_case', print_type='first')


    def _generate_model_objects(self):
        # weather object (+load variables at date start)
        self.weather = Weather.agile_constructor(self.weather_def_dict, self.general_dict)

        # generate the time parameters stored in time_management_dict
        self._generate_time_parameters()

        # airzone objects
        self.airzone_list_dict = generate_airzone_list(self.airzone_def_dict_list)

        # panel objects: surfaces, cores and facets
        self.surface_list_dict, self.facet_list_dict, self.core_list_dict, self.mesh_list_dict = generate_panel_list(self.panel_def_dict_list)

        # probe objects
        self.probe_set_list = generate_probe_set_list(self.probe_set_def_list)

        # connect objects
        self._generate_connect_objects()


    def _generate_radiative_viewfactors_and_prefactors(self):
        # check facet and mesh dimensions
        if len(self.facet_list_dict['domain']) != self.mesh_list_dict['combined_domain'][0].n_cells + 1:
            raise ValueError('Incompatibility between number of facets and combined mesh triangles.'
                             'Normally the number of facets should be the number of triangles + 1 for the sky.')

        # see system_geometry arguments
        self._generate_rad_utility_lists()

        system_geometry = RadiativeSystemGeometry(self.rad_transmissivity_sw_list,
                                                  self.rad_opposite_transmissivity_sw_list,
                                                  self.rad_opposite_facet_index_list,
                                                  self.rad_probe_opposite_transmissivity_sw_list,
                                                  self.rad_tripoints_list,
                                                  self.rad_trinormal_list,
                                                  self.rad_triarea_list,
                                                  self.rad_procoord_list,
                                                  self.mesh_list_dict['combined_domain'][0])

        # compute sunray directions during target date
        target_date_list = self.time_management_dict['target_date_list']
        sunray_direction_list = self.weather.compute_sun_direction_list(target_date_list)

        # generate viewfactors and prefactors for direct solar radiation for the domain and the probes
        self._generate_for_domain(sunray_direction_list, system_geometry)
        self._generate_for_probes(sunray_direction_list, system_geometry)

        self.system_geometry = system_geometry


    def _generate_radiative_model(self):
        radiative_model = RadiativeModel()

        # connections
        radiative_model.connect_weather(self.weather)
        radiative_model.set_facet_list(self.facet_list_dict['domain'])
        radiative_model.set_probe_set_list(self.probe_set_list)
        radiative_model.load_viewfactor_mat(self.rad_domain_viewfactor_mat)
        radiative_model.load_utility_lists(self.rad_opposite_facet_index_list, self.rad_transmissivity_sw_list, self.rad_transmissivity_lw_list)

        # compute matrices and prefactors for sw radiation calculation with the radiosity method
        radiative_model.compute_radiosity_matrix()
        radiative_model.compute_sw_flux_prefactor(self.time_management_dict['target_date_list'],
                                                  self.rad_domain_prefactor_arr,
                                                  self.rad_domain_partial_obstruction_arr)

        self.radiative_model = radiative_model


    def _initialize_variables(self):
        # initialize conduction matrices and coefficients in cores
        for core_props in self.core_list_dict['props']:
            core_props.compute_base_model_matrix()

        # initialize objects temperatures
        self._initialize_temperatures()

        # initialize outputs
        self._initialize_outputs(self.output_def_dict)


    def _generate_time_parameters(self):
        """Initialize time parameters stored in time_management_dict."""
        # unpack general_dict and weather variables
        tz_info = self.weather.tz_info
        dt = self.general_dict['dt']
        warmup_days_num = self.general_dict['warmup_days_num']
        userdt_warmup_days_num = self.general_dict['userdt_warmup_days_num']

        # check values
        # warmup_days_num: integer > 1
        if type(warmup_days_num) is not int or warmup_days_num < 1:
            raise ValueError(f"'{warmup_days_num}' value for parameter 'warmup_days_num' "
                             f"in general_dict is not a valid. Expected value: an int larger than 0")

        # userdt_warmup_days_num: integer <= warmup_days_num
        if type(userdt_warmup_days_num) is not int or userdt_warmup_days_num > warmup_days_num:
            raise ValueError(f"'{userdt_warmup_days_num}' value for parameter 'userdt_warmup_days_num' "
                             f"in general_dict is not a valid. Expected value: an int smaller than '{warmup_days_num}'")

        # dt: multiple of 3600s
        if (3600 % dt) != 0:
            raise ValueError(f"'{dt}' value for parameter 'dt' "
                             f"in general_dict is not a valid. Expected value: an integer fraction of 3600s")

        # set date range with fixed dt of 1h over the first part of the warmup period
        # (from the beginning to userdt_warmup_days_num day before the target day)
        target_date = pd.to_datetime(datetime.datetime(*self.general_dict['target_date'])).tz_localize(tz=tz_info)
        date_start = target_date - datetime.timedelta(days=warmup_days_num)
        date_end = target_date - datetime.timedelta(days=userdt_warmup_days_num)
        date_range_1 = pd.date_range(date_start, date_end, freq='1h')[1:]  # remove first timestep

        # set date range with user-defined dt over the target day and userdt_warmup_days_num before
        date_start = target_date - datetime.timedelta(days=userdt_warmup_days_num)
        date_end = target_date + datetime.timedelta(days=1)
        date_range_2 = pd.date_range(date_start, date_end, freq=f"{dt}s")[1:]  # remove first timestep

        # set date range for target date only
        date_start = target_date
        date_end = target_date + datetime.timedelta(days=1)
        target_date_range = pd.date_range(date_start, date_end, freq=f"{dt}s")

        # store the date range in time_management_dict
        self.time_management_dict = {}
        self.time_management_dict['dt'] = dt
        self.time_management_dict['tz_info'] = tz_info
        self.time_management_dict['target_date'] = target_date
        self.time_management_dict['warmup_days_num'] = warmup_days_num
        self.time_management_dict['userdt_warmup_days_num'] = userdt_warmup_days_num
        self.time_management_dict['date_list'] = date_range_1.union(date_range_2)
        self.time_management_dict['target_date_list'] = target_date_range
        self.time_management_dict['switch_timestep_date'] = date_range_2[0]


    def _generate_rad_utility_lists(self):
        facet_list = self.facet_list_dict['domain']
        n_facets = len(facet_list)

        transmissivity_sw_list = [0.] * (n_facets - 1) # exclude sky facet
        transmissivity_lw_list = [0.] * (n_facets - 1)
        opposite_facet_index_list = [None] * (n_facets - 1)
        opposite_transmissivity_sw_list = [0.] * (n_facets - 1)
        probe_opposite_transmissivity_sw_list = [0.] * (n_facets - 1)
        # opposite_is_tree_list = [False] * (n_facets - 1)

        for i, facet_i in enumerate(facet_list[:-1]):
            transmissivity_sw_list[i] = facet_i.parent_props.transmissivity_sw
            transmissivity_lw_list[i] = facet_i.parent_props.transmissivity_lw
            if facet_i.opposite_facet is None:
                continue
            for j, facet_j in enumerate(facet_list):
                if facet_i.opposite_facet == facet_j:
                    opposite_facet_index_list[i] = j
                    opposite_transmissivity_sw_list[i] = facet_j.parent_props.transmissivity_sw
                    probe_opposite_transmissivity_sw_list[i] = facet_j.parent_props.probe_transmissivity_sw
                    # opposite_is_tree_list[i] = facet_j.is_tree
                    break

        self.rad_transmissivity_sw_list = transmissivity_sw_list
        self.rad_transmissivity_lw_list = transmissivity_lw_list
        self.rad_opposite_facet_index_list = opposite_facet_index_list
        self.rad_opposite_transmissivity_sw_list = opposite_transmissivity_sw_list
        self.rad_probe_opposite_transmissivity_sw_list = probe_opposite_transmissivity_sw_list
        # self.rad_opposite_is_tree_list = opposite_is_tree_list

        combined_domain = self.mesh_list_dict['combined_domain'][0]
        self.rad_tripoints_list = [np.array([combined_domain.points[idx] for idx in face]) for face in combined_domain.regular_faces]
        self.rad_trinormal_list = list(combined_domain.face_normals.astype(np.float64))
        self.rad_triarea_list = list(combined_domain['cell_areas'])

        # flat list of probe coordinates
        rad_probe_set_coord_list = [probe_set.coord_list for probe_set in self.probe_set_list]
        self.rad_procoord_list = [procoord for procoord_list in rad_probe_set_coord_list for procoord in procoord_list]


    def _generate_connect_objects(self):
        """Connect objects between themselves."""
        # connect surfaces and airzones
        for airzone in self.airzone_list_dict['all']:
            for surface in self.surface_list_dict['all']:
                if not surface.airzone_name == airzone.name:
                    continue
                airzone.connect_surface(surface)
                surface.connect_airzone(airzone)

        # connect facets to weather
        for facet in self.facet_list_dict['weather']:
                facet.connect_weather(self.weather)

        # connect weather to airzones, sky and tree surfaces, and probes
        for airzone in self.airzone_list_dict['weather']:
            airzone.connect_weather(self.weather)
        for airzone in self.airzone_list_dict['balanced']:
            airzone.connect_weather(self.weather)
        for probe_set in self.probe_set_list:
            probe_set.connect_weather(self.weather)

        # set dt for airzones and surfaces
        for airzone in self.airzone_list_dict['balanced']:
            # default timestep of 1h for simulation warmup
            # (will be updated to the user-defined timestep later)
            airzone.set_dt(3600)
        for core_props in self.core_list_dict['props']:
            # default timestep of 1h for simulation warmup
            # (will be updated to the user-defined timestep later)
            core_props.set_dt(3600)

        # connect neighbour airzones
        for airzone_i in self.airzone_list_dict['balanced']:
            for airzone_j in self.airzone_list_dict['all']:
                if airzone_i.inflow_airzone_name == airzone_j.name:
                    airzone_i.connect_inflow_airzone(airzone_j)
                    break

        # probe sets
        for probe_set in self.probe_set_list:
            # connect the airzone
            selected_airzone = None
            for airzone in self.airzone_list_dict['all']:
                if probe_set.airzone_name == airzone.name:
                    selected_airzone = airzone
                    if isinstance(airzone, FixedAirZone):
                        raise ValueError('The probe surface cannot be associated with a fixed airzone.')
            if selected_airzone is None:
                raise ValueError(f"Airzone not found for probe set {probe_set.name}.")
            else:
                probe_set.connect_airzone(selected_airzone)

        # compute and set pet interpolator to probe sets
        var_map, pet_map = np.load(self.general_dict['regpet_filepath'], allow_pickle=True)
        pet_interpolator = RegularGridInterpolator(var_map, pet_map)
        for probe_set in self.probe_set_list:
            probe_set.set_pet_interpolator(pet_interpolator)


    def _generate_for_domain(self, sunray_direction_list, system_geometry):
        if self.general_dict['from_save']:
            path = self.general_dict['from_save_folderpath']
            self.general_dict['from_save_folderpath'] = path if path.endswith('/') else path + '/'

            if self.general_dict['from_save_probes_type'] is not None and self.general_dict['from_save_domain_type'] not in ['viewfactors', 'solar_prefactors', 'all']:
                raise ValueError(" from_save_domain_type parameter in general_dict must be 'viewfactors', 'solar_prefactors', or 'all'.")

            if self.general_dict['from_save_domain_type'] in ['viewfactors', 'all']:
                # load view factors
                print(f"    load viewfactor matrix from save in folder '{self.general_dict['from_save_folderpath']}'.\n")
                rad_domain_viewfactor_mat = np.load(self.general_dict['from_save_folderpath'] + 'rad_domain_viewfactor_mat.pkl', allow_pickle=True)
                if rad_domain_viewfactor_mat.shape[0] != len(self.facet_list_dict['domain']):
                    raise ValueError('Inconsistency between rad_domain_viewfactor_mat from save file and the number of facets in the domain')
            else:
                # compute view factors
                system_geometry.compute_visibility_matrix()
                rad_domain_viewfactor_mat = system_geometry.compute_viewfactor_matrix()

            if self.general_dict['from_save_domain_type'] in ['solar_prefactors', 'all']:
                # load primary direct irradiance prefactors per sun direction
                print(f"    load solar prefactor matrix from save in folder '{self.general_dict['from_save_folderpath']}'.\n")
                rad_domain_prefactor_arr = np.load(self.general_dict['from_save_folderpath'] + 'rad_domain_prefactor_arr.pkl', allow_pickle=True)
                if rad_domain_prefactor_arr.shape[0] != len(sunray_direction_list):
                    raise ValueError('Inconsistency between rad_domain_prefactor_arr from save file and the number of sunray directions')
                if rad_domain_prefactor_arr.shape[1] != len(self.facet_list_dict['domain']):
                    raise ValueError('Inconsistency between rad_domain_prefactor_arr from save file and the number of facets in the domain')

                # load primary direct irradiance prefactors per sun direction
                rad_domain_partial_obstruction_arr = np.load(self.general_dict['from_save_folderpath'] + 'rad_domain_partial_obstruction_arr.pkl', allow_pickle=True)
                if rad_domain_partial_obstruction_arr.shape[0] != len(sunray_direction_list):
                    raise ValueError('Inconsistency between rad_domain_partial_obstruction_arr from save file and the number of sunray directions')
                if rad_domain_partial_obstruction_arr.shape[1] != len(self.facet_list_dict['domain']):
                    raise ValueError('Inconsistency between rad_domain_partial_obstruction_arr from save file and the number of facets in the domain')
            else:
                # compute primary direct irradiance prefactors per sun direction
                rad_domain_prefactor_arr, rad_domain_partial_obstruction_arr = system_geometry.compute_primary_direct_irradiance_prefactor(sunray_direction_list)

        else:
            # compute view factors
            system_geometry.compute_visibility_matrix()
            rad_domain_viewfactor_mat = system_geometry.compute_viewfactor_matrix()

            # compute primary direct irradiance prefactors per sun direction
            rad_domain_prefactor_arr, rad_domain_partial_obstruction_arr = system_geometry.compute_primary_direct_irradiance_prefactor(sunray_direction_list)

        self.rad_domain_viewfactor_mat = rad_domain_viewfactor_mat
        self.rad_domain_prefactor_arr = rad_domain_prefactor_arr
        self.rad_domain_partial_obstruction_arr = rad_domain_partial_obstruction_arr


    def _generate_for_probes(self, sunray_direction_list, system_geometry):
        if len(self.probe_set_list) > 0:
            if self.general_dict['from_save']:
                if self.general_dict['from_save_probes_type'] is not None and self.general_dict['from_save_probes_type'] not in ['viewfactors', 'solar_prefactors', 'all']:
                    raise ValueError(" from_save_probes_type parameter in general_dict must be 'viewfactors', 'solar_prefactors', or 'all'.")

                if self.general_dict['from_save_probes_type'] in ['viewfactors', 'all']:
                    # load view factors
                    rad_probe_viewfactor_mat = np.load(self.general_dict['from_save_folderpath'] + 'rad_probe_viewfactor_mat.pkl', allow_pickle=True)
                    if rad_probe_viewfactor_mat.shape[0] != len(self.rad_procoord_list):
                        raise ValueError('Inconsistency between rad_probe_viewfactor_mat from save file and the number of probes')
                    elif rad_probe_viewfactor_mat.shape[2] != len(self.facet_list_dict['domain']):
                        raise ValueError('Inconsistency between rad_probe_viewfactor_mat from save file and the number of facets in the domain')
                else:
                    # compute view factors
                    system_geometry.compute_visible_probes()
                    rad_probe_viewfactor_mat = system_geometry.compute_probe_viewfactor_matrix()

                if self.general_dict['from_save_probes_type'] in ['solar_prefactors', 'all']:
                    # load primary direct irradiance prefactors per sun direction
                    rad_probe_prefactor_arr = np.load(self.general_dict['from_save_folderpath'] + 'rad_probe_prefactor_arr.pkl', allow_pickle=True)
                    if rad_probe_prefactor_arr.shape[0] != len(sunray_direction_list):
                        raise ValueError('Inconsistency between rad_probe_prefactor_arr from save file and the number of sunray directions')
                    elif rad_probe_prefactor_arr.shape[1] != len(self.rad_procoord_list):
                        raise ValueError('Inconsistency between rad_probe_prefactor_arr from save file and the number of probes')

                    # load primary direct irradiance prefactors per sun direction
                    rad_probe_partial_obstruction_arr = np.load(self.general_dict['from_save_folderpath'] + 'rad_probe_partial_obstruction_arr.pkl', allow_pickle=True)
                    if rad_probe_partial_obstruction_arr.shape[0] != len(sunray_direction_list):
                        raise ValueError('Inconsistency between rad_probe_partial_obstruction_arr from save file and the number of sunray directions')
                    elif rad_probe_partial_obstruction_arr.shape[1] != len(self.rad_procoord_list):
                        raise ValueError('Inconsistency between rad_probe_partial_obstruction_arr from save file and the number of probes')
                else:
                    # compute primary direct irradiance prefactors per sun direction
                    rad_probe_prefactor_arr, rad_probe_partial_obstruction_arr = system_geometry.compute_probe_primary_direct_irradiance_prefactor(sunray_direction_list)

            else:
                # compute view factors
                system_geometry.compute_visible_probes()
                rad_probe_viewfactor_mat = system_geometry.compute_probe_viewfactor_matrix()

                # compute primary direct irradiance prefactors per sun direction
                rad_probe_prefactor_arr, rad_probe_partial_obstruction_arr = system_geometry.compute_probe_primary_direct_irradiance_prefactor(sunray_direction_list)

            # assign the view factor matrix to the corresponding probe
            probe_idx = 0
            for probe_set in self.probe_set_list:
                for probe in probe_set.probe_list:
                    probe.set_viewfactor_matrix(rad_probe_viewfactor_mat[probe_idx, :])
                    probe_idx += 1

            # assign the prefactors to the corresponding probe
            datetime_list = self.time_management_dict['target_date_list']
            for datetime_i, probe_prefactor_arr, probe_partial_obstruction_arr in zip(datetime_list, rad_probe_prefactor_arr, rad_probe_partial_obstruction_arr):
                probe_idx = 0
                for probe_set in self.probe_set_list:
                    for probe in probe_set.probe_list:
                        probe.set_primary_direct_irradiance_prefactor_dict((datetime_i.hour, datetime_i.minute),
                                                                           probe_prefactor_arr[probe_idx, :])
                        probe.set_primary_direct_irradiance_partial_obstruction_dict((datetime_i.hour, datetime_i.minute),
                                                                           probe_partial_obstruction_arr[probe_idx])
                        probe_idx += 1

            self.rad_probe_viewfactor_mat = rad_probe_viewfactor_mat
            self.rad_probe_prefactor_arr = rad_probe_prefactor_arr
            self.rad_probe_partial_obstruction_arr = rad_probe_partial_obstruction_arr


    def _initialize_temperatures(self):
        """Initialize temperature arrays in Airzone, Facet, and Core objects."""
        temp_init = self.general_dict['temperature_init']

        for airzone in self.airzone_list_dict['balanced']:
            airzone.initialize_temperature(temp_init)

        for facet in self.facet_list_dict['domain']:
            facet.initialize_temperature(temp_init)

        for core in self.core_list_dict['all']:
            core.initialize_temperature_arr(temp_init)


    def _initialize_outputs(self, output_def_dict):
        """
        Initializes the output dictionary based on a definition dictionary.
        :param dict output_def_dict: Dictionary defining output configurations.
        """
        # initialize output dictionary with empty dictionaries for each output type
        self.output_dict = {key: {} for key in output_def_dict.keys()}

        # define mappings for object lists and variable names
        obj_list_dict = {
            'airzone'         : self.airzone_list_dict['all'],
            'surface_averaged': self.surface_list_dict['no_sky'],
            'surface_field'   : self.surface_list_dict['no_sky'],
            'probe'           : [probe for probe_set in self.probe_set_list for probe in probe_set.probe_list],
            'weather'         : self.weather
        }
        var_name_list_dict = {
            'airzone'         : AirZone.state_variable_list,
            'surface_averaged': Surface.state_variable_list,
            'surface_field'   : Facet.state_variable_list,
            'probe'           : Probe.state_variable_list,
            'weather'         : Weather.state_variable_list
        }

        # iterate through each output type in the definition dictionary
        for o_type, o_def_list in output_def_dict.items():
            if o_type == 'weather':
                if o_def_list == ['*']:
                    o_def_list = var_name_list_dict['weather']
                for var_name in o_def_list:
                    self.output_dict[o_type][(None, var_name)] = (self.weather, [])
                continue

            obj_list = obj_list_dict[o_type]
            var_list_ref = var_name_list_dict[o_type]
            valid_obj_name_list = {obj.name for obj in obj_list} | {'*'}
            valid_var_name_list = set(var_list_ref) | {'*'}

            for output_def in o_def_list:
                obj_name, var_name = output_def

                if obj_name not in valid_obj_name_list:
                    raise ValueError(f"Invalid object name '{obj_name}' for {o_type}")
                if var_name not in valid_var_name_list:
                    raise ValueError(f"Invalid variable name '{var_name}' for {o_type}. Must be in {var_list_ref}")

                selected_obj_list = obj_list if obj_name == '*' else [obj for obj in obj_list if obj.name == obj_name]
                selected_var_list = var_list_ref if var_name == '*' else [var_name]

                for obj in selected_obj_list:
                    for var in selected_var_list:
                        self.output_dict[o_type][(obj.name, var)] = (obj, [])

        output_type_list = ['target_date', 'full_period']
        if self.general_dict['output_period'] == 'target_date':
            self.time_management_dict['switch_timestep_date_output'] = self.time_management_dict['target_date']
        elif self.general_dict['output_period'] == 'full_period':
            self.time_management_dict['switch_timestep_date_output'] = self.time_management_dict['date_list'][0]
        else:
            raise ValueError(f"'{self.general_dict['output_period']}' value for parameter 'output_period' "
                             f"in general_dict is not valid. Expected value: {output_type_list}.")


    def _solve_update_dt(self):
        # reset dt to user-defined timestep for balanced airzones
        for airzone in self.airzone_list_dict['balanced']:
            airzone.set_dt(self.time_management_dict['dt'])

        # reset dt to user-defined timestep for core props and recompute matrices
        for core_props in self.core_list_dict['props']:
            core_props.set_dt(self.time_management_dict['dt'])
            core_props.compute_base_model_matrix()


    def _solve_initialize_timestep(self):
        """
            Update the temperature of facets from weather data for each surface without cores
            Update the temperature array of the previous timestep for each surface with cores
            """
        # update hc and prev_ta on dynamic airzones and compute the modified wind speed on balanced airzones (under shelter)
        for airzone in self.airzone_list_dict['dynamic']:
            airzone.initialize_timestep()

        # update facet temperature connected to the weather (sky and tree)
        for facet in self.facet_list_dict['weather']:
            facet.get_temperature_from_weather()

        # update matrices in core_props
        for core in self.core_list_dict['single_per_panel']:
            core.update_matrix()

        # update previous timestep temperatures in cores
        for core in self.core_list_dict['all']:
            core.update_prev_timestep()


    def _solve_external_fluxes_facet_to_core(self):
        for core in self.core_list_dict['all']:
            core.get_external_fluxes_from_facet()


    def _solve_temperature_core_to_facet(self):
        """
        Get facet temperature from balanced core cells
        """
        for facet in self.facet_list_dict['core']:
            if 'front' in facet.parent_surface.name:
                facet.get_temperature_from_core('front')
            else: # back
                facet.get_temperature_from_core('back')


    def _solve_compute_balanced_airzone_temperature(self):
        for airzone in self.airzone_list_dict['balanced']:
            airzone.compute_temperature()


    def _solve_iterate_temperatures(self):
        """
        Iterate the temperatures calculations for each cell with a massic core object
        :return: the maximum temperature error over all Wall objects
        """
        err_list = [core.iterate_temperature(self.general_dict['relax_coef']) for core in self.core_list_dict['all']]
        return max(err_list)

    def _solve_compute_metrics_at_probes(self, datetime_i):
        self.radiative_model.compute_probe_sw_and_lw_fluxes(datetime_i)
        for probe_set in self.probe_set_list:
            probe_set.compute_tmrt()
            probe_set.compute_comfort_index()


    def _solve_write_outputs(self):
        # append outputs to dict
        for o_type in self.output_dict.keys():
            if o_type == 'surface_field':
                for (obj_name, var_name), (obj, val_list) in self.output_dict[o_type].items():
                    val_list.append([getattr(facet, var_name) for facet in obj.facet_list])
            else:
                for (obj_name, var_name), (obj, val_list) in self.output_dict[o_type].items():
                    if o_type == 'surface_averaged':
                        obj.compute_average_variable(var_name)
                    val_list.append(getattr(obj, var_name))


if __name__ == '__main__':
    from _input_shelter_old import general_dict, weather_def_dict, panel_def_dict_list, airzone_def_dict_list, probe_set_def_list, output_def_dict

    # create the state object
    imotep = IMOTEP(general_dict,
                    weather_def_dict,
                    panel_def_dict_list,
                    airzone_def_dict_list,
                    probe_set_def_list,
                    output_def_dict)

    imotep.weather = Weather.agile_constructor(imotep.weather_def_dict, imotep.general_dict)

    imotep._generate_time_parameters()
