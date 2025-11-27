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
from facet import FacetProperties, Facet
import pyvista as pv

class Surface:

    state_variable_list = ['avg_cond_flux', 'avg_conv_flux', 'avg_lw_rad_flux', 'avg_sw_rad_flux', 'avg_temperature']

    def __init__(self, type, base_name, name, mesh=None, airzone_name=None):
        self.type = type
        self.base_name = base_name
        self.name = name
        self.mesh = self._check_mesh(mesh)
        self.airzone_name = airzone_name
        self.opposite_surface = None

    def _check_mesh(self, mesh):
        if mesh is not None:
            if not 'cell_areas' in mesh.cell_data.keys():
                mesh.cell_data['cell_areas'] = mesh.compute_cell_sizes(length=False, area=True, volume=False)['Area']
            if not 'cell_centers' in mesh.cell_data.keys():
                mesh.cell_data['cell_centers'] = mesh.cell_centers().points
        return mesh

    @classmethod
    def front_surface(cls, surf_type, surf_name, mesh, airzone_name):
        base_name = surf_name
        name = surf_name + '_front'
        airzone_name = airzone_name
        # create the surface object
        return cls(surf_type, base_name, name, mesh, airzone_name)

    @classmethod
    def back_surface(cls, front_surface, airzone_name):
        surf_type = front_surface.type
        base_name = front_surface.base_name
        name = base_name + '_back'
        mesh_back = front_surface.mesh.copy()
        mesh_back = mesh_back.compute_normals(flip_normals=True)
        airzone_name = airzone_name
        # create the surface object
        back_surface = cls(surf_type, base_name, name, mesh_back, airzone_name)
        # connect the two surfaces
        back_surface.opposite_surface = front_surface
        front_surface.opposite_surface = back_surface
        return back_surface

    @classmethod
    def cover_back_surface(cls, front_surface):
        surf_type = front_surface.type + '_cover_back'
        base_name = front_surface.base_name
        name = base_name + '_cover_back'
        airzone_name = None

        # check if the front surface is a vertical plane
        fs_normals = front_surface.mesh.cell_normals
        fs_normal_ref = fs_normals[0]
        for fs_normal in fs_normals:
            if fs_normal[2] > 0.0:
                raise ValueError("The front surface of a cover back should be vertical")
            if np.linalg.norm(fs_normal - fs_normal_ref) > 1e-6:
                raise ValueError("The front surface of a cover back should be a plane")

        # get the bounds of the mesh: (xmin, xmax, ymin, ymax, zmin, zmax)
        xmin, xmax, ymin, ymax, zmin, zmax = front_surface.mesh.bounds
        min_corner_index = front_surface.mesh.find_closest_point((xmin, ymin, zmin))
        min_corner_point = front_surface.mesh.points[min_corner_index]
        if np.linalg.norm(min_corner_point - np.array([xmin, ymin, zmin])) < 1e-6:
            y_when_xmin = ymin
            y_when_xmax = ymax
        else:
            y_when_xmin = ymax
            y_when_xmax = ymin
        # construct the 4 corner points
        points = [
            [xmin, y_when_xmin, zmin],  # bottom-left
            [xmax, y_when_xmax, zmin],  # bottom-right
            [xmax, y_when_xmax, zmax],  # top-right
            [xmin, y_when_xmin, zmax],  # top-left
        ]
        faces = [
            3, 0, 1, 2,  # Triangle 1 (nb of points, then indices)
            3, 0, 2, 3,  # Triangle 2 (nb of points, then indices)
        ]
        mesh_cover_back = pv.PolyData(points, faces)

        # check if mesh_cover surface if facing toward the opposite direction of the front surface
        # otherwise, flip the surface
        mesh_cover_normal = mesh_cover_back.cell_normals[0]
        front_surf_normal = front_surface.mesh.cell_normals[0]
        if np.dot(mesh_cover_normal, front_surf_normal) > 0:
            mesh_cover_back = mesh_cover_back.compute_normals(flip_normals=True)

        # set a small offset between the front and the cover back surface
        # Later, the surfaces are not connected, thus adding an offset prevent the
        # opposite facets intersect when computing obstructions
        mesh_cover_back.points += 1e-3 * mesh_cover_back.cell_normals[0]

        # create the surface object
        cover_back_surface = cls(surf_type, base_name, name, mesh_cover_back, airzone_name)

        # no connection between front and cover_back surface
        return cover_back_surface

    @classmethod
    def sky_surface(cls):
        return cls('sky', 'sky', 'sky_front')

    def create_facets(self, facet_args):
        facet_props = FacetProperties(*facet_args)
        if self.mesh is not None:
            mesh = self.mesh.compute_cell_sizes(length=False, area=True, volume=False)
            facet_area_list = mesh['cell_areas']
        else: # for sky surface which has no mesh
            facet_area_list = [1]
        self.facet_list = [Facet(facet_area, facet_props, self) for facet_area in facet_area_list]

    def connect_airzone(self, airzone):
        self.airzone = airzone

    def compute_average_variable(self, variable_name):
        if variable_name == 'avg_temperature':
            temp_pow_4_arr = np.array([np.power(facet.temperature + 273.15, 4) for facet in self.facet_list])
            num = np.sum(self.mesh['cell_areas'] * temp_pow_4_arr)
            denom = np.sum(self.mesh['cell_areas'])
            avg_variable = np.power(num / denom, .25) - 273.15
        elif variable_name in [var_name for var_name in Surface.state_variable_list if var_name != 'avg_temperature']:
            flux_arr = np.array([getattr(facet, variable_name.replace('avg_', '')) for facet in self.facet_list])
            avg_variable = np.sum(self.mesh['cell_areas'] * flux_arr) / np.sum(self.mesh['cell_areas'])
        else:
            raise ValueError(f"'{variable_name}' value for parameter 'variable_name' in compute_average_variable function "
                             f"not valid. Expected value: {Surface.state_variable_list}")
        setattr(self, variable_name, avg_variable)


if __name__ == '__main__':
    from _input_shelter_old import panel_def_dict_list

    pdd = panel_def_dict_list[4]
    front_surf = Surface.front_surface(pdd['name'], pdd['type'], pdd['mesh'], pdd['front_airzone_name'])
    cover_surf = Surface.cover_back_surface(front_surf)

    pt = pv.Plotter()
    pt.add_mesh(front_surf.mesh, color='b')
    pt.add_mesh(cover_surf.mesh, color='r')
    pt.add_arrows(front_surf.mesh['cell_centers'], front_surf.mesh.cell_normals, mag=0.35, color='k')
    pt.add_arrows(cover_surf.mesh['cell_centers'], cover_surf.mesh.cell_normals, mag=0.35, color='g')
    pt.show()