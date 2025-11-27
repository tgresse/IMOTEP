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

class FacetProperties:
    """
    A class containing the facet properties of all the facets of a surface.

    Attributes
    ----------
    albedo: float
        albedo of the facet
    emissivity: float
        emissivity of the facet
    transmissivity_sw: float
        transmissivity for short-wave radiative fluxes
    transmissivity_lw: float
        transmissivity for long-wave radiative fluxes
    """

    def __init__(self, emissivity, albedo, transmissivity_lw, transmissivity_sw, probe_transmissivity_sw=None):
        """
        Initialize a FacetProperties object
        :param float emissivity: emissivity
        :param float albedo: albedo
        :param float transmissivity_lw: transmissivity of the long-wave radiative flux through the surface
        :param float transmissivity_sw: transmissivity of the short-wave radiative flux through the surface
        """
        # shortwave properties
        if albedo < 0.0 or albedo > 1.0:
            raise ValueError(f'albedo must be between 0 and 1. Here the value is {albedo}')
        if transmissivity_sw < 0.0 or transmissivity_sw > 1.0:
            raise ValueError(f'transmissivity_sw must be between 0 and 1. Here the value is {transmissivity_sw}')
        self.albedo = albedo
        self.transmissivity_sw = transmissivity_sw
        if albedo + transmissivity_sw > 1:
            raise ValueError(f'albedo + transmissivity_sw must be between 0 and 1. Here the value is {albedo + transmissivity_sw}')
        self.absorptivity_sw = 1. - albedo - transmissivity_sw

        # long wave properties
        if emissivity < 0.0 or emissivity > 1.0:
            raise ValueError(f'emissivity must be between 0 and 1. Here the value is {emissivity}')
        if transmissivity_lw < 0.0 or transmissivity_lw > 1.0:
            raise ValueError(f'transmissivity_lw must be between 0 and 1. Here the value is {transmissivity_lw}')
        self.emissivity = emissivity
        self.transmissivity_lw = transmissivity_lw
        if emissivity + transmissivity_lw > 1:
            raise ValueError(f'emissivity + transmissivity_lw must be between 0 and 1. Here the value is {emissivity + transmissivity_lw}')
        self.reflectivity_lw = 1. - emissivity - transmissivity_lw

        # flag transparency
        if transmissivity_sw > .0 or transmissivity_lw > .0:
            self.is_transparent = True
        else:
            self.is_transparent = False

        if probe_transmissivity_sw is None:
            self.probe_transmissivity_sw = transmissivity_sw
        else:
            self.probe_transmissivity_sw = probe_transmissivity_sw


class Facet:
    """
    A class representing a facet.

    Attributes
    ----------
    temperature: float
        temperature of the facet
    opposite_facet: Facet object
        pointer to the opposite facet that belong to the same cell
    core: Core object
        pointer to the core that belong to the same cell
    sw_rad_flux: float
        net shortwave radiative flux at the facet
    lw_rad_flux: float
        net longwave radiative flux at the facet
    sw_radiosity: float
        shortwave radiosity of the facet
    lw_radiosity: float
        longwave radiosity of the facet
    conv_flux: float
        convective flux at the facet
    """

    state_variable_list = ['temperature', 'cond_flux', 'conv_flux', 'lw_rad_flux', 'sw_rad_flux', 'lw_radiosity', 'sw_radiosity', 'sun_exposure']

    def __init__(self, area, parent_props, parent_surface):
        """
        Initialize a Facet object
        """
        self.area = area
        # initialize parent objects : FacetProperties and Surface
        self.parent_props = parent_props
        self.parent_surface = parent_surface
        self.opposite_facet = None
        self.cover_back_facet = None
        # self.is_tree = False
        # create state variables
        self.sw_rad_flux = .0
        self.lw_rad_flux = .0
        self.sw_radiosity = .0
        self.lw_radiosity = .0
        self.sun_exposure = 0
        self.conv_flux = .0
        self.cond_flux = .0

    # def set_is_tree(self):
    #     self.is_tree = True

    def set_sw_rad_flux(self, sw_rad_flux):
        self.sw_rad_flux = sw_rad_flux

    def set_lw_rad_flux(self, lw_rad_flux):
        self.lw_rad_flux = lw_rad_flux

    def set_sw_radiosity(self, sw_radiosity):
        self.sw_radiosity = sw_radiosity

    def set_lw_radiosity(self, lw_radiosity):
        self.lw_radiosity = lw_radiosity

    def set_sun_exposure(self, sun_exposure):
        self.sun_exposure = sun_exposure

    def initialize_temperature(self, temperature_init):
        self.temperature = temperature_init

    def connect_core(self, core):
        self.core = core

    def connect_weather(self, weather):
        self.weather = weather
        # if parent surface is a building set, the facet should belong to the cover back surface
        if self.parent_surface.type == 'tree' or self.parent_surface.type == 'building_set_cover_back':
            self.weather_variable = 'air_temperature'
        elif self.parent_surface.type == 'sky':
            self.weather_variable = 'sky_temperature'
        else:
            raise ValueError(f"Cannot connect belonging to a {self.parent_surface.type} type surface to the weather")

    def connect_opposite_facet(self, opposite_facet):
        self.opposite_facet = opposite_facet

    def connect_cover_back_facet(self, cover_back_facet):
        self.cover_back_facet = cover_back_facet

    def get_temperature_from_weather(self):
        self.temperature = getattr(self.weather, self.weather_variable)

    def get_temperature_from_core(self, facet_type):
        """
        Get the facet temperature from the core
        :param string facet_type: type of facet
        """
        if facet_type == 'front':
            self.temperature = self.core.facet_temperature_arr[0]
            # compute the conductive flux for outputs
            self.cond_flux = self.core.parent_props.front_cell_cond_over_dx * (self.core.temperature_guess_arr[0] - self.temperature)
        elif facet_type == 'back':
            self.temperature = self.core.facet_temperature_arr[1]
            # compute the conductive flux for outputs
            self.cond_flux = self.core.parent_props.back_cell_cond_over_dx * (self.core.temperature_guess_arr[-1] - self.temperature)
        # compute the convective flux for outputs
        self.conv_flux = self.parent_surface.airzone.hc * (self.parent_surface.airzone.temperature - self.temperature)