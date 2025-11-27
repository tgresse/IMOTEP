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

SIGMA = 5.67e-8

def generate_probe_set_list(probe_set_def_list):
    # initialize storage lists
    probe_set_list = []
    name_list = []

    for probe_set_def_dict in probe_set_def_list:
        # store the name and check if it does not already exist
        name = probe_set_def_dict['name']
        if probe_set_def_dict in name_list:
            raise ValueError(f"'{name}' value for parameter 'name' in comfort_probe_dict is used more than once.")
        name_list.append(name)

        # create the probe set objects depending on the format of the inputs
        if type(probe_set_def_dict['mesh']) is list:
            comfort_probe_set = ProbeSet(*[probe_set_def_dict[k] for k in ('name', 'mesh', 'airzone_name')])
        else:
            comfort_probe_set = ProbeSet.from_polydata(*[probe_set_def_dict[k] for k in ('name', 'mesh', 'airzone_name')])
        probe_set_list.append(comfort_probe_set)

    return probe_set_list


class ProbeSet:

    def __init__(self, name, coord_list, airzone_name):
        self.name = name
        self.coord_list = np.array(coord_list)
        self.airzone_name = airzone_name
        self.probe_list = [Probe(name+'_'+str(idx), coord) for idx, coord in enumerate(coord_list)]

    @classmethod
    def from_polydata(cls, name, mesh, airzone_name):
        coord_list = mesh.points
        return cls(name, coord_list, airzone_name)

    def connect_airzone(self, airzone):
        self.airzone = airzone

    def connect_weather(self, weather):
        self.weather = weather

    def set_pet_interpolator(self, pet_interpolator):
        self.pet_interpolator = pet_interpolator

    def compute_tmrt(self):
        """
        Compute the mean radiant temperature (tmrt) for each probe of a probe set
        """
        for probe in self.probe_list:
            probe.compute_tmrt()

    def compute_comfort_index(self):
        """
        Compute the SET comfort index for each probe of a probe set using PET interpolation.
        """
        n_probes = len(self.probe_list)

        # Retrieve or compute required inputs
        ta = self.airzone.temperature
        ws = self.airzone.wind_speed
        rh = self.airzone.relative_humidity
        met = 1.2
        clo = 0.5

        # Generate arrays for interpolation
        ta_arr = np.full(n_probes, ta)
        tr_arr = np.array([probe.tmrt for probe in self.probe_list])
        ws_arr = np.full(n_probes, ws)
        rh_arr = np.full(n_probes, rh)
        met_arr = np.full(n_probes, met)
        clo_arr = np.full(n_probes, clo)

        # Stack inputs for interpolation (shape: n_probes x 6)
        inputs = np.column_stack([ta_arr, tr_arr, ws_arr, rh_arr, met_arr, clo_arr])

        # Interpolate PET from the precomputed PET mapping
        try:
            pet_values = self.pet_interpolator(inputs, method='linear')
        except:
            raise ValueError(f'Error in pet interpolator inputs: {inputs}.\n'
                             f'Check correspondence with ranges used to generate regpet.pkl file.')

        # Store results in each probe
        for probe, pet in zip(self.probe_list, pet_values):
            probe.comfort_index = pet


class Probe:

    state_variable_list = ['comfort_index', 'tmrt', 'sw_flux_arr', 'lw_flux_arr', 'sun_exposure']

    def __init__(self, name, coord):
        self.name = name
        self.coord = coord
        self.primary_direct_irradiance_prefactor_dict = {}
        self.primary_direct_irradiance_partial_obstruction_dict = {}

    def set_viewfactor_matrix(self, viewfactor_mat):
        self.viewfactor_mat = viewfactor_mat

    def set_primary_direct_irradiance_prefactor_dict(self, key, value):
        self.primary_direct_irradiance_prefactor_dict[key] = value

    def set_primary_direct_irradiance_partial_obstruction_dict(self, key, value):
        self.primary_direct_irradiance_partial_obstruction_dict[key] = value

    def set_sw_flux_arr(self, sw_flux_arr):
        self.sw_flux_arr = sw_flux_arr

    def set_lw_flux_arr(self, lw_flux_arr):
        self.lw_flux_arr = lw_flux_arr

    def set_sun_exposure(self, sun_exposure):
        self.sun_exposure = sun_exposure

    def compute_tmrt(self):
        """
        Compute the mean radiant temperature for comfort index calculation using the methodology presented in
        Lindberg F. et al (2008), SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant
        temperature in complex urban settings, Int J Biometeorol (DOI: 10.1007/s00484-008-0162-7)
        (originally in Höppe (1992), A new procedure to determine the mean radiant temperature outdoors. Wetter Leben)
        """
        # compute the mean radiant flux density s_str
        angular_factor_arr = np.array([0.06, 0.06, 0.22, 0.22, 0.22, 0.22])  # top, bottom, east, west, north, south
        ksi_k = .7  # absorption coefficient for sw radiation
        eps_p = 0.97  # emissivity of a human body
        s_str = ksi_k * angular_factor_arr * self.sw_flux_arr + eps_p * angular_factor_arr * self.lw_flux_arr

        # compute the mean radiant temperature using the Stephan-Boltzmann law
        self.tmrt = np.power(np.sum(s_str) / (eps_p * SIGMA), 0.25) - 273.15