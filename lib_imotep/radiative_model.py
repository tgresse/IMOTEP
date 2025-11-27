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

class RadiativeModel:
    """
    Radiative model class where the shortwave and longwave radiosity and net fluxes are computed.

    Attributes
    ----------
    viewfactor_mat: np.ndarray
        matrix containing the view factors of all the facets in the scene
    viewfactor_complementarity_check_arr: np.ndarray
        array of the sum of viewfactor_mat columns to check complementarity property of view factors
    viewfactor_trans_mat: np.ndarray
        matrix containing the view factors of the opposite for facets that transmit flux in the scene
    sw_radiosity_mat: np.ndarray
        matrix for shortwave radiosity calculation
    lw_radiosity_mat: np.ndarray
        matrix for longwave radiosity calculation
    """

    def __init__(self):
        """
        Initialize the RadiativeModel object
        """
        pass

    def connect_weather(self, weather):
        self.weather = weather

    def set_facet_list(self, facet_list):
        self.facet_list = facet_list

    def set_probe_set_list(self, probe_set_list):
        self.probe_set_list = probe_set_list

    def load_viewfactor_mat(self, viewfactor_mat):
        self.viewfactor_mat = viewfactor_mat

    def load_utility_lists(self, opposite_facet_index_list, transmissivity_sw_list, transmissivity_lw_list):
        self.opposite_facet_index_list = opposite_facet_index_list
        self.transmissivity_sw_list = transmissivity_sw_list
        self.transmissivity_lw_list = transmissivity_lw_list

    def compute_radiosity_matrix(self):
        # compute the viewfactor matrices for transmissivity
        viewfactor_trans_sw_rad_mat = np.zeros(self.viewfactor_mat.shape)
        viewfactor_trans_lw_rad_mat = np.zeros(self.viewfactor_mat.shape)
        for i, (j, transmissivity_sw, transmissivity_lw) in enumerate(zip(self.opposite_facet_index_list, self.transmissivity_sw_list, self.transmissivity_lw_list)):
            if transmissivity_sw > 0.0:
                if j is not None:
                    viewfactor_trans_sw_rad_mat[i, :] = self.viewfactor_mat[j, :]
            if transmissivity_lw > 0.0:
                if j is not None:
                    viewfactor_trans_lw_rad_mat[i, :] = self.viewfactor_mat[j, :]
        self.viewfactor_trans_sw_rad_mat = viewfactor_trans_sw_rad_mat
        self.viewfactor_trans_lw_rad_mat = viewfactor_trans_lw_rad_mat

        # compute the radiosity matrices [[Id-rho*FF]] used to compute the facets shortwave and longwave radiosity
        # that account for the transmissivity of the fluxes across semi-transparent surfaces
        reflectivity_sw_mat = np.diag(np.array([facet.parent_props.albedo for facet in self.facet_list]))
        reflectivity_lw_mat = np.diag(np.array([facet.parent_props.reflectivity_lw for facet in self.facet_list]))
        transmissivity_sw_mat = np.diag(np.array([facet.parent_props.transmissivity_sw for facet in self.facet_list]))
        transmissivity_lw_mat = np.diag(np.array([facet.parent_props.transmissivity_lw for facet in self.facet_list]))
        self.sw_radiosity_mat = np.eye(len(self.facet_list)) - reflectivity_sw_mat @ self.viewfactor_mat - transmissivity_sw_mat @ self.viewfactor_trans_sw_rad_mat
        self.lw_radiosity_mat = np.eye(len(self.facet_list)) - reflectivity_lw_mat @ self.viewfactor_mat - transmissivity_lw_mat @ self.viewfactor_trans_lw_rad_mat
        # modified matrix for net flux calculation
        self.sw_nf_radiosity_mat = (np.eye(len(self.facet_list)) - (reflectivity_sw_mat + transmissivity_sw_mat)) @ self.viewfactor_mat
        self.lw_nf_radiosity_mat = (np.eye(len(self.facet_list)) - (reflectivity_lw_mat + transmissivity_lw_mat)) @ self.viewfactor_mat

    def compute_lw_flux(self):
        """
        Compute the net longwave radiative flux at facets
        """
        # STEP 1: primary fluxes arrays
        # primary emittance: Stephan-Boltzmann law
        primary_emittance_arr = np.array([facet.parent_props.emissivity * SIGMA * (facet.temperature + 273.15) ** 4 for facet in self.facet_list])
        # STEP 2: multi-reflections
        # compute radiosity array by solving the system of equations (lw_radiosity_mat * radiosity_arr = primary_emittance_arr)
        radiosity_arr = np.linalg.solve(self.lw_radiosity_mat, primary_emittance_arr)
        # STEP 3: net fluxes
        # compute the net flux array (difference between the global irradiance array and the radiosity array)
        # convention: positive if the surface absorbs heat
        # strategy 1: flux balance with the transmitted flux accounted on the opposite side of the wall where the primary flux is incident
        # net_lw_flux_arr = self.viewfactor_mat.dot(radiosity_arr) - radiosity_arr
        # strategy 2: keep the transmitted flux component to the flux balance on the side of the wall where the primary flux is incident
        # Consistent with the strategy used for the primary direct irradiance flux
        net_lw_flux_arr = self.lw_nf_radiosity_mat.dot(radiosity_arr) - primary_emittance_arr
        # STEP 4: store in facet attributes
        for idx, facet in enumerate(self.facet_list):
            facet.set_lw_radiosity(radiosity_arr[idx])
            facet.set_lw_rad_flux(net_lw_flux_arr[idx])
        self.lw_radiosity_arr = radiosity_arr

    def compute_sw_flux_prefactor(self, datetime_list, primary_direct_irradiance_prefactor_per_sundir_arr, primary_direct_irradiance_partial_obstruction_per_sundir_arr):
        """
        Compute the net shortwave radiative flux prefactors at facets
        """
        self.prefactor_date = datetime_list[0] # should be at midnight
        albedo_arr = np.array([facet.parent_props.albedo for facet in self.facet_list])
        transmissivity_sw_arr = np.array(self.transmissivity_sw_list + [1.]) # +1 for the sky

        # -- direct radiation --
        sun_exposure_dict = {}
        radiosity_direct_prefactor_dict = {}
        net_direct_prefactor_dict = {}
        for i, (datetime_i, primary_direct_irradiance_prefactor_arr, primary_direct_irradiance_partial_obstruction_arr) in enumerate(zip(datetime_list, primary_direct_irradiance_prefactor_per_sundir_arr, primary_direct_irradiance_partial_obstruction_per_sundir_arr)):
            # compute an  array that states which facets are exposed to the sun (value 1) or not (value 0)
            sun_exposure_arr = [1 if (prefactor > 0. and not partial_obstruction) else 0 for prefactor, partial_obstruction in zip(primary_direct_irradiance_prefactor_arr, primary_direct_irradiance_partial_obstruction_arr)]

            # STEP 1 : first reflexion to compute the emittance
            # no need to perform first transmission: already done as direct to direct transmission in primary prefactor
            primary_direct_emittance_prefactor_arr = albedo_arr * primary_direct_irradiance_prefactor_arr

            # STEP 2: multi-reflections
            # compute radiosity prefactor array by solving the system of equations
            radiosity_direct_prefactor_arr = np.linalg.solve(self.sw_radiosity_mat, primary_direct_emittance_prefactor_arr)

            # STEP 3: net fluxes
            # compute the net flux prefactor array (difference between the global irradiance prefactor array and the radiosity prefactor array)
            # convention: positive if the surface absorbs heat
            # old version
            # net_direct_prefactor_arr = ((1. - transmissivity_sw_arr) * primary_direct_irradiance_prefactor_arr
            #                             + self.viewfactor_mat.dot(radiosity_direct_prefactor_arr)
            #                             - radiosity_direct_prefactor_arr)
            # new version
            net_direct_prefactor_arr = ((1. - transmissivity_sw_arr) * primary_direct_irradiance_prefactor_arr
                                        + self.sw_nf_radiosity_mat.dot(radiosity_direct_prefactor_arr)
                                        - primary_direct_emittance_prefactor_arr)

            sun_exposure_dict[(datetime_i.hour, datetime_i.minute)] = sun_exposure_arr
            radiosity_direct_prefactor_dict[(datetime_i.hour, datetime_i.minute)] = radiosity_direct_prefactor_arr
            net_direct_prefactor_dict[(datetime_i.hour, datetime_i.minute)] = net_direct_prefactor_arr

        self.sw_sun_exposure_dict = sun_exposure_dict
        self.sw_primary_direct_irradiance_prefactor_per_sundir_arr = primary_direct_irradiance_prefactor_per_sundir_arr
        self.sw_radiosity_direct_prefactor_dict = radiosity_direct_prefactor_dict
        self.sw_net_direct_prefactor_dict = net_direct_prefactor_dict

        # -- diffuse radiation --
        # initialize diffuse emittance at 0 W/m2 for all surfaces
        primary_diffuse_emittance_prefactor_arr = np.zeros(len(self.facet_list))
        # assign 1 W/m2 for the sky
        primary_diffuse_emittance_prefactor_arr[-1] = 1
        # compute radiosity prefactor array by solving the system of equations
        self.sw_radiosity_diffuse_prefactor_arr = np.linalg.solve(self.sw_radiosity_mat, primary_diffuse_emittance_prefactor_arr)
        # compute the net flux prefactor array
        # strategy 1: flux balance with the transmitted flux accounted on the opposite side of the wall where the primary flux is incident
        # self.sw_net_diffuse_prefactor_arr = self.viewfactor_mat.dot(self.sw_radiosity_diffuse_prefactor_arr) - self.sw_radiosity_diffuse_prefactor_arr
        # strategy 2: keep the transmitted flux component to the flux balance on the side of the wall where the primary flux is incident
        # Consistent with the strategy used for the primary direct irradiance flux
        self.sw_net_diffuse_prefactor_arr = self.sw_nf_radiosity_mat.dot(self.sw_radiosity_diffuse_prefactor_arr) - primary_diffuse_emittance_prefactor_arr

    def compute_sw_flux(self, datetime_i):
        """
        Compute the net shortwave radiative flux at facets
        """
        sw_radiosity_arr = np.zeros(len(self.facet_list))
        for i, (facet, sun_exposure, radiosity_direct_prefactor, radiosity_diffuse_prefactor, net_sw_flux_direct_prefactor, net_sw_flux_diffuse_prefactor) in enumerate(zip(
                self.facet_list,
                self.sw_sun_exposure_dict[(datetime_i.hour, datetime_i.minute)],
                self.sw_radiosity_direct_prefactor_dict[(datetime_i.hour, datetime_i.minute)],
                self.sw_radiosity_diffuse_prefactor_arr,
                self.sw_net_direct_prefactor_dict[(datetime_i.hour, datetime_i.minute)],
                self.sw_net_diffuse_prefactor_arr)):
            sw_radiosity_arr[i] = radiosity_direct_prefactor * self.weather.direct_normal_radiation + \
                                   radiosity_diffuse_prefactor * self.weather.diffuse_horizontal_radiation
            facet.set_sun_exposure(sun_exposure)
            facet.set_sw_radiosity(sw_radiosity_arr[i])
            facet.set_sw_rad_flux(net_sw_flux_direct_prefactor * self.weather.direct_normal_radiation + \
                                  net_sw_flux_diffuse_prefactor * self.weather.diffuse_horizontal_radiation)
        self.sw_radiosity_arr = sw_radiosity_arr

    def compute_probe_sw_and_lw_fluxes(self, datetime_i):
        for probe_set in self.probe_set_list:
            for probe in probe_set.probe_list:
                # compute diffuse sw and lw flux on probe
                lw_flux_arr = probe.viewfactor_mat.dot(self.lw_radiosity_arr)
                sw_flux_arr = probe.viewfactor_mat.dot(self.sw_radiosity_arr)

                # compute direct sw flux on probe and add it to the sw heat flux array
                primary_direct_irradiance_partial_obstruction = probe.primary_direct_irradiance_partial_obstruction_dict[(
                                    datetime_i.hour, datetime_i.minute)]
                primary_direct_irradiance_prefactor_arr = probe.primary_direct_irradiance_prefactor_dict[(
                                    datetime_i.hour, datetime_i.minute)]
                sw_flux_arr += primary_direct_irradiance_prefactor_arr * self.weather.direct_normal_radiation

                # store as probe argument
                probe.set_lw_flux_arr(lw_flux_arr)
                probe.set_sw_flux_arr(sw_flux_arr)

                if np.any(primary_direct_irradiance_prefactor_arr) > 0. and not primary_direct_irradiance_partial_obstruction:
                    probe.set_sun_exposure(1)
                else:
                    probe.set_sun_exposure(0)