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
import pickle as pkl
from pythermalcomfort.models import pet_steady

# define input ranges
ta_arr = np.linspace(10, 50, 17) # step = 2.5
tr_arr = np.linspace(0, 90, 37) # step = 2.5
va_arr = np.geomspace(1, 11, 25) - 1
rh_arr = np.linspace(0, 80, 9)
met_arr = np.array([1.2, 2.4])
clo_arr = np.array([0.25, 0.5])
var_map = [ta_arr, tr_arr, va_arr, rh_arr, met_arr, clo_arr]

# compute all combinations
mesh = np.meshgrid(*var_map, indexing='ij') # use indexing='ij' to preserve axis order
val_grid = np.stack(mesh, axis=-1).reshape(-1, 6)

# extract individual lists of combinations
ta_list, tr_list, va_list, rh_list, met_list, clo_list = val_grid.T.tolist()

# compute the PET values
pet_flat = pet_steady(ta_list, tr_list, va_list, rh_list, met_list, clo_list).pet

# reshape into N-dimensional array matching var_map shape
grid_shape = tuple(len(arr) for arr in var_map)
pet_map = np.array(pet_flat).reshape(grid_shape)

with open('regpet.pkl', 'wb') as data:
    pkl.dump((var_map, pet_map), data)