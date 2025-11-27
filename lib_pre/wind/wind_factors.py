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

def compute_effective_area_rectangle(width, length, height, num_dir):
    """ compute equivalent area for volume flow rate calculation. """
    wd_arr = np.linspace(0,360,num_dir+1) # 0° = north, corresponding to +X direction
    wd_arr = wd_arr[:-1]
    bound_area_arr = np.array([width*height, length*height, width*height, length*height])  # [+X, -Y, -X, +Y] clockwise rotation
    bound_normal_arr = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])  # [+X, -Y, -X, +Y] clockwise rotation
    equiv_area_list = []
    for wd in wd_arr:
        wd_rad = np.deg2rad(wd)
        # normalize the view direction
        view_direction_arr = np.array([np.cos(wd_rad), np.sin(wd_rad)])
        # compute dot product to determine face visibility
        visibility_arr = bound_normal_arr @ view_direction_arr
        # get indices of the two most visible faces (largest dot products)
        visible_face_arr = np.argsort(visibility_arr)[-2:]  # take top 2 visible faces
        # sum the areas of the two visible faces
        equiv_area_list.append(sum(bound_area_arr[i]*visibility_arr[i] for i in visible_face_arr))
    return (list(wd_arr), equiv_area_list)

def compute_wind_factor_flat_ground(met_height, target_height, roughness, num_dir):
    wd_arr = np.linspace(0, 360, num_dir + 1)  # 0° = north, corresponding to +X direction
    wd_arr = wd_arr[:-1]
    wf = np.log10(target_height / roughness) / np.log10(met_height / roughness)
    return (list(wd_arr), [wf] * len(wd_arr))


if __name__ == "__main__":
    wf0 = compute_wind_factor_flat_ground(10, 1, 0.9, 8) # -> wf = 0.044
    wf1 = compute_wind_factor_flat_ground(10, 1, 0.774, 8) # -> wf = 0.1
    wf2 = compute_wind_factor_flat_ground(10, 1, 0.7, 8) # -> wf = 0.13 (under continuous tree cover)
    wf3 = compute_wind_factor_flat_ground(10, 1, 0.56234, 8) # -> wf = 0.2
    wf4 = compute_wind_factor_flat_ground(10, 1, 0.5, 8) # -> wf = 0.23
    wf5 = compute_wind_factor_flat_ground(10, 1, 0.4, 8) # -> wf = 0.285
    wf6 = compute_wind_factor_flat_ground(10, 1, 0.2, 8)  # -> wf = 0.41
    wf7 = compute_wind_factor_flat_ground(10, 1, 0.1, 8)  # -> wf = 0.5
    wf8 = compute_wind_factor_flat_ground(10, 1, 0.01, 8) # -> wf = 0.67
    wf9 = compute_wind_factor_flat_ground(10, 1, 0.001, 8)  # -> wf = 0.75 (under shelter)
    wf10 = compute_wind_factor_flat_ground(10, 1, 0.0001, 8)  # -> wf = 0.8