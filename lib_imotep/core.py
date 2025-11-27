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

class CoreProperties:
    """
    A class containing the core properties of all the cores of a surface.

    Attributes
    ----------
    a_mat: np.ndarray
        state-transition matrix for heat equation resolution
    b_coef_arr: np.ndarray
        array of coefficients of the input-control matrix at the upper and lower boundaries for heat equation resolution
    conductance_arr: np.ndarray
        array of conductances between the ext facet and the first wall node and the int facet and the last wall node
    nnodes: int
        number of nodes in the wall discretisation
    """

    def __init__(self, layer_def_dict_list):
        """
        Initialize a CoreProperties object
        :param list layer_def_dict_list: list of dictionary containing the wall layers properties
        """
        self.compute_model_matrix_coef(layer_def_dict_list)

    @classmethod
    def from_list(cls, layer_def_list_list):
        layer_def_dict_list = []
        for layer_def_list in layer_def_list_list:
            layer_def_dict_list.append({
                'thickness'      : layer_def_list[0],
                'conductivity'   : layer_def_list[1],
                'capacity'       : layer_def_list[2],
                'exp_coef'       : layer_def_list[3],
                'nnodes'         : layer_def_list[4]
            })
        return cls(layer_def_dict_list)

    def set_dt(self, dt):
        self.dt = dt

    def compute_model_matrix_coef(self, layer_def_dict_list):
        """
        Compute matrices coefficients used to resolve the differential system of equations
        :param list layer_def_dict_list: list of dictionary containing the wall layers properties
        """
        # compute cells thickness and properties arrays (a cell is an element of the discretisation of the wall layers)
        # initialize mesh cells arrays
        cell_dx_arr = np.empty(0)
        cell_cond_arr = np.empty(0)
        cell_capa_arr = np.empty(0)
        # loop through material layers
        for layer_def_dict in layer_def_dict_list:
            # unpack dictionary
            # do a specific treatment if nnodes and exp_coefficient of the layer are None to compute cell_dx_arr with
            # adapted values (not yet available)
            layer_nnodes = layer_def_dict['nnodes']
            layer_exp_coefficient = layer_def_dict['exp_coef']
            layer_thickness = layer_def_dict['thickness']
            layer_conductivity = layer_def_dict['conductivity']
            layer_capacity = layer_def_dict['capacity']
            # layer characteristics
            layer_cell_index_arr = np.arange(0, layer_nnodes)
            layer_first_cell_dxi = layer_thickness / (
                np.sum(np.power(layer_exp_coefficient, layer_cell_index_arr)))  # thickness of the first cell
            layer_cell_dxi_arr = layer_first_cell_dxi * np.power(layer_exp_coefficient,
                                                                 layer_cell_index_arr)  # cell thickness array
            # append to global arrays
            cell_dx_arr = np.append(cell_dx_arr, layer_cell_dxi_arr)
            cell_cond_arr = np.append(cell_cond_arr, np.array(layer_conductivity * np.ones(layer_nnodes)))
            cell_capa_arr = np.append(cell_capa_arr, np.array(layer_capacity * np.ones(layer_nnodes)))
        # compute state-transition matrix
        self.cell_capa_dx_arr = cell_dx_arr * cell_capa_arr
        self.front_cell_capa_dx = self.cell_capa_dx_arr[0]
        self.back_cell_capa_dx = self.cell_capa_dx_arr[-1]
        cell_dx_over_cond_arr = cell_dx_arr / (2. * cell_cond_arr)  # half thermal resistance of each cell (size(N))
        self.front_cell_cond_over_dx = 1. / cell_dx_over_cond_arr[0]
        self.back_cell_cond_over_dx = 1. / cell_dx_over_cond_arr[-1]
        self.conductance_arr = 1 / (cell_dx_over_cond_arr[:-1] + cell_dx_over_cond_arr[1:])  # thermal conductance between the cells (size(N-1))
        # compute total number of cells
        self.nnodes = len(cell_dx_arr)

    def compute_base_model_matrix(self):
        """
        We solve: A * T^n+1 = T^n + B * U^n
        A = a_mat with a_mat[0,0] and a_mat[N,N] updated depending on hc
        B = [b_0, 0, ..., 0, b_N] with coefficients updated depending on hc
        U = [U_0, 0, ..., 0, U_N] = sum of radiative fluxes + hc * Ta, U_N remains zero if adiabatic surface
        Compute the base matrix of the conduction system
        """
        self.a_mat = np.eye(self.nnodes) - self.dt * (
                     np.diag(-1. / self.cell_capa_dx_arr * (np.append(self.conductance_arr, 0) + np.append(0, self.conductance_arr))) +
                     np.diag(1. / self.cell_capa_dx_arr[1:] * self.conductance_arr, -1) +
                     np.diag(1. / self.cell_capa_dx_arr[:-1] * self.conductance_arr, 1))
        self.base_a00 = self.a_mat[0, 0]
        self.base_aNN = self.a_mat[-1, -1]
        # compute control-input matrix elements
        self.b_coef_arr = [self.dt / self.front_cell_capa_dx, self.dt / self.back_cell_capa_dx]
        self.base_b0 = self.b_coef_arr[0]
        self.base_bN = self.b_coef_arr[-1]

    def update_model_matrix_front(self, hc):
        """
        Compute matrices and coefficients used to resolve the differential system of equations
        :param float hc: 
        """
        # compute the conductance between last wall node and air node (coef)
        coef = hc * self.front_cell_cond_over_dx / (hc + self.front_cell_cond_over_dx)
        self.a_mat[0,0] = self.base_a00 + self.dt / self.front_cell_capa_dx * coef
        self.b_coef_arr[0] = self.base_b0 * coef / hc

    def update_model_matrix_back(self, hc):
        """
        Compute matrices and coefficients used to resolve the differential system of equations
        :param float hc:
        """
        # compute the conductance between last wall node and air node (coef)
        coef = hc * self.back_cell_cond_over_dx / (hc + self.back_cell_cond_over_dx)
        self.a_mat[-1,-1] = self.base_aNN + self.dt / self.back_cell_capa_dx * coef
        self.b_coef_arr[-1] = self.base_bN * coef / hc


class Core:
    """
    A class representing a core.

    Attributes
    ----------
    facet_temperature_arr: np.ndarray
        array of facet temperatures at the end of a timestep
    front_facet: Facet object
        a pointer to the front facet that belong to the same cell
    back_facet: Facet object
        a pointer to the back facet that belong to the same cell
    temperature_guess_arr: np.ndarray
        array of temperatures computed by resolving the heat equation with a differential system of equations
    temperature_relax_new_iter_arr: np.ndarray
        array of relaxed temperatures at the end of the new iteration of the iterative resolution procedure
    temperature_relax_prev_iter_arr: np.ndarray
        array of relaxed temperatures at the end of the previous iteration of the iterative resolution procedure
    temperature_prev_timestep_arr: np.ndarray
        array of temperatures at the end of the previous calculation timestep
    facet_temperature_guess_arr: np.ndarray
        array of ext and int facets temperatures computed by resolving a heat balance
    facet_temperature_relax_new_iter_arr: np.ndarray
        array of relaxed ext and int facets temperatures at the end of the new iteration of the iterative resolution procedure
    facet_temperature_relax_prev_iter_arr: np.ndarray
        array of relaxed ext and int facets temperatures at the end of the previous iteration of the iterative resolution procedure
    external_flux_arr : np.ndarray()
        array of the external fluxes from front_facet and back_facet
    """

    def __init__(self, parent_props):
        """
        Initialize a Core object
        """
        # initialize parent object : CoreProperties
        self.parent_props = parent_props

    def connect_front_facet(self, front_facet):
        self.front_facet = front_facet

    def connect_back_facet(self, back_facet):
        self.back_facet = back_facet

    def initialize_temperature_arr(self, temperature):
        nnodes = self.parent_props.nnodes
        self.facet_temperature_arr = temperature * np.ones(2)
        self.temperature_guess_arr = temperature * np.ones(nnodes)
        self.temperature_relax_new_iter_arr = temperature * np.ones(nnodes)
        self.temperature_relax_prev_iter_arr = temperature * np.ones(nnodes)
        self.temperature_prev_timestep_arr = temperature * np.ones(nnodes)
        self.facet_temperature_guess_arr = temperature * np.ones(2)
        self.facet_temperature_relax_new_iter_arr = temperature * np.ones(2)
        self.facet_temperature_relax_prev_iter_arr = temperature * np.ones(2)
        self.external_flux_arr = np.zeros(2)

    def _iter_guess_temperatures(self):
        """
        Compute the guessed temperatures arrays
        """
        # compute temperature_guess_arr by resolving the differential system of equations
        bu_extremity_arr = self.parent_props.b_coef_arr * self.external_flux_arr
        bu_arr = np.zeros(self.parent_props.nnodes)
        bu_arr[0] = bu_extremity_arr[0]
        bu_arr[-1] = bu_extremity_arr[1]
        self.temperature_guess_arr = np.linalg.solve(self.parent_props.a_mat, self.temperature_prev_timestep_arr + bu_arr)

    def _iter_relax_temperatures(self, relax_coef):
        """
        Compute the relaxed temperatures arrays
        :param relax_coef: relaxation coefficient for stability of the temperature calculation
        """
        # reset temperature_relax_prev_iter_arr
        self.temperature_relax_prev_iter_arr = self.temperature_relax_new_iter_arr
        self.facet_temperature_relax_prev_iter_arr = self.facet_temperature_relax_new_iter_arr
        # compute temperature_relax_new_iter_arr by relaxing temperature_guess_arr with temperature_relax_prev_iter_arr
        self.temperature_relax_new_iter_arr = self.temperature_relax_prev_iter_arr + relax_coef * (
                self.temperature_guess_arr - self.temperature_relax_prev_iter_arr)
        self.facet_temperature_relax_new_iter_arr = self.facet_temperature_relax_prev_iter_arr + relax_coef * (
                self.facet_temperature_guess_arr - self.facet_temperature_relax_prev_iter_arr)
        self.facet_temperature_arr = self.facet_temperature_relax_new_iter_arr

    def _iter_compute_error_temperatures(self):
        """
        Compute the temperature errors to check the convergence of the ping-pong iterative process
        :return: the maximum temperature error
        """
        err_core = np.max(np.abs(self.temperature_guess_arr - self.temperature_relax_prev_iter_arr))
        err_facet = np.max(np.abs(self.facet_temperature_guess_arr - self.facet_temperature_relax_prev_iter_arr))
        return max(err_core, err_facet)

    def update_prev_timestep(self):
        """
        Update the temperature array of the previous timestep
        """
        self.temperature_prev_timestep_arr = self.temperature_relax_new_iter_arr

    def _iter_front_facet_guess_temperature(self):
        self.facet_temperature_guess_arr[0] = (self.parent_props.front_cell_cond_over_dx * self.temperature_guess_arr[0] + self.external_flux_arr[0]) / \
                                               (self.front_facet.parent_surface.airzone.hc + self.parent_props.front_cell_cond_over_dx)

    def _iter_back_facet_guess_temperature(self):
        self.facet_temperature_guess_arr[-1] = (self.parent_props.back_cell_cond_over_dx * self.temperature_guess_arr[-1] + self.external_flux_arr[-1]) / \
                                                (self.back_facet.parent_surface.airzone.hc + self.parent_props.back_cell_cond_over_dx)

    def _get_external_fluxes_from_front_facet(self):
        """
        Get the sum of the external fluxes at front facet (net shortwave radiativeflux, net longwave radiative flux, convective flux). Convention: flux positif entrant
        """
        self.external_flux_arr[0] = self.front_facet.sw_rad_flux + self.front_facet.lw_rad_flux + self.front_facet.parent_surface.airzone.hc * self.front_facet.parent_surface.airzone.temperature

    def _get_external_fluxes_from_back_facet(self):
        """
        Get the sum of the external fluxes at back facet (net shortwave radiativeflux, net longwave radiative flux, convective flux). Convention: flux positif entrant
        """
        self.external_flux_arr[-1] = self.back_facet.sw_rad_flux + self.back_facet.lw_rad_flux + self.back_facet.parent_surface.airzone.hc * self.back_facet.parent_surface.airzone.temperature

class CoreSingleFacet(Core):
    
    def update_matrix(self):
        self.parent_props.update_model_matrix_front(self.front_facet.parent_surface.airzone.hc)

    def iterate_temperature(self, relax_coef):
        # --- step 1 : compute guess temperature ---
        # compute iternal temperatures
        self._iter_guess_temperatures()
        # compute front facet temperature
        self._iter_front_facet_guess_temperature()
        # --- step 2 : relax temperature ---
        self._iter_relax_temperatures(relax_coef)
        # --- step 3 : compute temperature errors ---
        return self._iter_compute_error_temperatures()

    def get_external_fluxes_from_facet(self):
        self._get_external_fluxes_from_front_facet()


class CoreDoubleFacet(Core):
    
    def update_matrix(self):
        self.parent_props.update_model_matrix_front(self.front_facet.parent_surface.airzone.hc)
        self.parent_props.update_model_matrix_back(self.back_facet.parent_surface.airzone.hc)

    def iterate_temperature(self, relax_coef):
        # --- step 1 : compute guess temperature ---
        # compute iternal temperatures
        self._iter_guess_temperatures()
        # compute front and back facets temperatures
        self._iter_front_facet_guess_temperature()
        self._iter_back_facet_guess_temperature()
        # --- step 2 : relax temperature ---
        self._iter_relax_temperatures(relax_coef)
        # --- step 3 : compute temperature errors ---
        return self._iter_compute_error_temperatures()

    def get_external_fluxes_from_facet(self):
        self._get_external_fluxes_from_front_facet()
        self._get_external_fluxes_from_back_facet()