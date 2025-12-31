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

import lib_pre.geometry.geogen as gg
import numpy as np
import pyvista as pv

#-----------------------------------------------------------------------------------------------------------------------
#shelter geometries
def geomorph_flat_shelter(x_min, x_max, y_min, y_max, h, dx=None, dy=None):
    shelter = gg.geogen_hrect(x_min=x_min, x_max=x_max,
                           y_min=y_min, y_max=y_max,
                           h=h,
                           dx=dx, dy=dy)

    return shelter

def geomorph_flat_shelter_with_sides(x_min, x_max, y_min, y_max, h_min, h_max, sides_orientation, dx=None, dy=None, dz=None):

    if h_min == h_max:
        raise ValueError('A flat shelter with sides cannot have zero length sides (h_min == h_max).')

    if sides_orientation not in ('east', 'west', 'north', 'south', 'east-west', 'north-south', 'east-west-north-south'):
        raise ValueError("  The orientation of the sides of a flat shelter must be 'east', 'west', 'north', 'south', "
                         "'east-west', 'north-south' or 'east-west-north-south'.")

    mesh_list = [geomorph_flat_shelter(x_min=x_min, x_max=x_max,
                                      y_min=y_min, y_max=y_max,
                                      h=h_max,
                                      dx=dx, dy=dy)]

    if sides_orientation in ('east', 'east-west', 'east-west-north-south'):
        mesh_list.append(gg.geogen_vrect(x_0=x_min, x_1=x_min,
                                      y_0=y_min, y_1=y_max,
                                      h_min=h_min, h_max=h_max,
                                      front_left=True,
                                      ds=dy, dh=dz))

    if sides_orientation in ('west', 'east-west', 'east-west-north-south'):
        mesh_list.append(gg.geogen_vrect(x_0=x_max, x_1=x_max,
                                      y_0=y_min, y_1=y_max,
                                      h_min=h_min, h_max=h_max,
                                      front_left=False,
                                      ds=dy, dh=dz))

    if sides_orientation in ('north', 'north-south', 'east-west-north-south'):
        mesh_list.append(gg.geogen_vrect(x_0=x_min, x_1=x_max,
                                      y_0=y_min, y_1=y_min,
                                      h_min=h_min, h_max=h_max,
                                      front_left=False,
                                      ds=dx, dh=dz))

    if sides_orientation in ('south', 'north-south', 'east-west-north-south'):
        mesh_list.append(gg.geogen_vrect(x_0=x_min, x_1=x_max,
                                      y_0=y_max, y_1=y_max,
                                      h_min=h_min, h_max=h_max,
                                      front_left=True,
                                      ds=dx, dh=dz))

    shelter = pv.merge(mesh_list)

    return shelter


def geomorph_gable_shelter(x_min, x_max, y_min, y_max, h_min, angle, orientation, dx=None, dy=None):

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    len_x = x_max - x_min
    len_y = y_max - y_min

    angle_rad = np.deg2rad(float(angle))
    sinus, cosinus = np.sin(angle_rad), np.cos(angle_rad)

    if orientation not in ('east-west', 'north-south'):
        raise ValueError("  Orientation of a gable shelter must be 'east-west' or 'north-south'.")

    if orientation == 'north-south':
        # Ridge along X, slope along ±Y
        half_span = len_y / 2  # horizontal distance ridge->eave
        rise = half_span * np.tan(angle_rad)  # vertical rise ridge above eave
        center_z = h_min + 0.5 * rise  # mid-height of each half-plane

        # Plane sizes in their local axes
        i_size = len_x  # along ridge (x)
        j_size = half_span / max(cosinus, 1e-9)  # sloped length ridge->eave

        # Normals: rotate (0,0,1) around +X by ±angle_rad → (0, ±sinθ, cosθ)
        n_plus = (0.0, +sinus, cosinus)  # slopes down toward +Y eave
        n_minus = (0.0, -sinus, cosinus)  # slopes down toward -Y eave

        # Centers: halfway between ridge (y=0) and eave (y=±half_span), at mid height
        center_plus = (center_x, center_y + half_span / 2, center_z)
        center_minus = (center_x, center_y - half_span / 2, center_z)

    else:  # 'east-west'
        # Ridge along Y, slope along ±X
        half_span = len_x / 2
        rise = half_span * np.tan(angle_rad)
        center_z = h_min + rise / 2

        # Sizes
        i_size = half_span / max(cosinus, 1e-9)  # sloped length ridge->eave
        j_size = len_y  # along ridge (y)

        # Normals: rotate (0,0,1) around +Y by ±angle_rad → (±sinθ, 0, cosθ)
        n_plus = (+sinus, 0.0, cosinus)  # slopes down toward +X eave
        n_minus = (-sinus, 0.0, cosinus)  # slopes down toward -X eave

        # Centers
        center_plus = (center_x + half_span / 2, center_y, center_z)
        center_minus = (center_x - half_span / 2, center_y, center_z)

    # Resolutions
    if dx is None:
        i_res = 1
    else:
        i_res = int(i_size // dx)
        if i_size % dx > 0:
            print(f"    [Warning] The length of the tilted rectangle is not multiple of the step (i_size={i_size}, "
                  f"dx={dx} -> i_res={i_res}).")
    if dy is  None:
        j_res = 1
    else:
        j_res = int(j_size // dy)
        if j_size % dy > 0:
            print(f"    [Warning] The length of the tilted rectangle is not multiple of the step (j_size={j_size}, "
                  f"dy={dy} -> j_res={j_res}).")

    mesh_list = [gg.build_plane(center_plus, n_plus, i_size, j_size, i_res, j_res),
                 gg.build_plane(center_minus, n_minus, i_size, j_size, i_res, j_res)]

    shelter = pv.merge(mesh_list)

    return shelter

def geomorph_tree(origin, radius, dr, ds):
    tree_base = gg.geogen_hdisc(origin=origin, radius=radius, dr=dr, ds=ds)
    tree_crown = gg.geogen_hhemisphere(origin, radius, dr, ds)

    return tree_base, tree_crown

#-----------------------------------------------------------------------------------------------------------------------
# floor geometries
def geomorph_flat_floor(x_min, x_max, y_min, y_max, dx=None, dy=None):
    floor = gg.geogen_hrect(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, h=0, dx=dx, dy=dy)

    return floor

#-----------------------------------------------------------------------------------------------------------------------
# building geometries
def geomorph_building(x_0, x_1,
                      y_0, y_1,
                      h_min, h_max,
                      front_left,
                      ds=None, dh=None):
    if h_min == h_max:
        raise ValueError(' A single building surface cannot have zero height (h_min == h_max).')

    build = gg.geogen_vrect(x_0, x_1, y_0, y_1, h_min, h_max, front_left, ds, dh)

    return build

def geomorph_roof(x_min, x_max,
                  y_min, y_max,
                  h,
                  dx=None, dy=None):

    ndir = [0., 0., -1]
    build = gg.geogen_hrect(x_min, x_max, y_min, y_max, h, dx, dy, ndir)

    return build

def geomorph_building_with_overhang(x_0, x_1,
                                    y_0, y_1,
                                    h_min, h_max,
                                    front_left,
                                    s_side, h_side,
                                    ds=None, dh=None,
                                    ds_side = None,
                                    split=None):

    if split:
        build_list = [geomorph_building(x_0=x_0, x_1=x_1,
                                      y_0=y_0, y_1=y_1,
                                      h_min=h_min, h_max=h_min+h_side,
                                      front_left=front_left,
                                      ds=ds, dh=dh),
                      geomorph_building(x_0=x_0, x_1=x_1,
                                      y_0=y_0, y_1=y_1,
                                      h_min=h_min+h_side, h_max=h_max,
                                      front_left=front_left)]
        build = gg.geogen_merge(build_list)
    else:
        build = geomorph_building(x_0=x_0, x_1=x_1,
                                y_0=y_0, y_1=y_1,
                                h_min=h_min, h_max=h_max,
                                front_left=front_left,
                                ds=ds, dh=dh)

    if x_0 < x_1 and front_left or x_0 > x_1 and not front_left:
        overhang = geomorph_flat_shelter(x_min=x_0, x_max=x_1,
                                       y_min=y_0, y_max=y_1+s_side,
                                       h=h_side,
                                       dx=ds, dy=ds_side)

    if x_0 < x_1 and not front_left or x_0 > x_1 and front_left:
        overhang = geomorph_flat_shelter(x_min=x_0, x_max=x_1,
                                       y_min=y_0-s_side, y_max=y_1,
                                       h=h_side,
                                       dx=ds, dy=ds_side)

    if y_0 < y_1 and front_left or y_0 > y_1 and not front_left:
        overhang = geomorph_flat_shelter(x_min=x_0-s_side, x_max=x_1,
                                       y_min=y_0, y_max=y_1,
                                       h=h_side,
                                       dx=ds_side, dy=ds)

    if y_0 < y_1 and not front_left or y_0 > y_1 and front_left:
        overhang = geomorph_flat_shelter(x_min=x_0, x_max=x_1+s_side,
                                       y_min=y_0, y_max=y_1,
                                       h=h_side,
                                       dx=ds_side, dy=ds)

    return build, overhang

def geomorph_urban_canyon(x_min, x_max,
                          y_min, y_max,
                          h,
                          orientation,
                          ds=None, dh=None):
    if orientation == 'east-west':
        # build the north building
        build1 = geomorph_building(x_0=x_min, x_1=x_max,
                                      y_0=y_min, y_1=y_min,
                                      h_min=0, h_max=h,
                                      front_left = True,
                                      ds=ds, dh=dh)

        # build the south building
        build2 = geomorph_building(x_0=x_min, x_1=x_max,
                                      y_0=y_max, y_1=y_max,
                                      h_min=0, h_max=h,
                                      front_left=False,
                                      ds=ds, dh=dh)

    elif orientation == 'north-south':
        # build the east building
        build1 = geomorph_building(x_0=x_min, x_1=x_min,
                                      y_0=y_min, y_1=y_max,
                                      h_min=0, h_max=h,
                                      front_left=False,
                                      ds=ds, dh=dh)

        # build the west building
        build2 = geomorph_building(x_0=x_max, x_1=x_max,
                                      y_0=y_min, y_1=y_max,
                                      h_min=0, h_max=h,
                                      front_left=True,
                                      ds=ds, dh=dh)

    else:
        raise ValueError("  The urban canyon orientation must be 'east-west' or 'north-south'.")

    return build1, build2


def geomorph_urban_canyon_with_overhang(x_min, x_max,
                                        y_min, y_max,
                                        h_b,
                                        s_side, h_side,
                                        orientation,
                                        ds=None, dh=None, ds_side=None,
                                        split=None):

    if orientation == 'east-west':
        # build the north building
        build1, shelter1 = geomorph_building_with_overhang(x_0=x_min, x_1=x_max,
                                                             y_0=y_min, y_1=y_min,
                                                             h_min=0, h_max=h_b,
                                                             front_left = True,
                                                             s_side=s_side, h_side=h_side,
                                                             ds=ds, dh=dh, ds_side=ds_side,
                                                             split=split)

        # build the south building
        build2, shelter2 = geomorph_building_with_overhang(x_0=x_min, x_1=x_max,
                                                             y_0=y_max, y_1=y_max,
                                                             h_min=0, h_max=h_b,
                                                             front_left=False,
                                                             s_side=s_side, h_side=h_side,
                                                             ds=ds, dh=dh, ds_side=ds_side,
                                                             split=split)

    elif orientation == 'north-south':
        # build the east building
        build1, shelter1 = geomorph_building_with_overhang(x_0=x_min, x_1=x_min,
                                                             y_0=y_min, y_1=y_max,
                                                             h_min=0, h_max=h_b,
                                                             front_left=False,
                                                             s_side=s_side, h_side=h_side,
                                                             ds=ds, dh=dh, ds_side=ds_side,
                                                             split=split)

        # build the west building
        build2, shelter2 = geomorph_building_with_overhang(x_0=x_max, x_1=x_max,
                                                             y_0=y_min, y_1=y_max,
                                                             h_min=0, h_max=h_b,
                                                             front_left=True,
                                                             s_side=s_side, h_side=h_side,
                                                             ds=ds, dh=dh, ds_side=ds_side,
                                                             split=split)

    else:
        raise ValueError("  The single building surface orientation must be 'east-west' or 'north-south'.")

    return build1, build2, shelter1, shelter2

#-----------------------------------------------------------------------------------------------------------------------
# buffer geometries
def geomorph_buffer(in_x_min, in_x_max, in_y_min, in_y_max, x_margin, y_margin=None, dx=None, dy=None):

    if y_margin is None:
        y_margin = x_margin

    out_x_min = in_x_min - x_margin
    out_x_max = in_x_max + x_margin
    out_y_min = in_y_min - y_margin
    out_y_max = in_y_max + y_margin

    buffer = gg.geogen_hrect_ring(in_x_min, in_x_max, in_y_min, in_y_max, out_x_min, out_x_max, out_y_min, out_y_max, dx, dy)

    return buffer


if __name__ == "__main__":

    x_min = 0
    x_max = 24
    y_min = 0
    y_max = 18
    h_min = 0
    hb = 10
    hs = 4
    sw = 6
    res = 1
    orientation = 'east-west'


    # surfaces
    # build, sideway = geomorph_building_with_overhang(x_0=x_max, x_1=x_max,
    #                                                  y_0=y_min, y_1=y_max,
    #                                                  h_min=h_min, h_max=hb,
    #                                                  front_left=False,
    #                                                  s_side=sw, h_side=hs,
    #                                                  ds=1, dh=1,
    #                                                  ds_side=1)
    #
    # mesh_surf_list  = [build, sideway]

    # build1, build2 = geomorph_building_canyon(x_min=x_min, x_max=x_max,
    #                                      y_min=y_min, y_max=y_max,
    #                                      h=hb,
    #                                      orientation=orientation,
    #                                      ds=res, dh=res)
    #
    # mesh_surf_list = [build1, build2]

    build1, build2, shelter1, shelter2 = geomorph_urban_canyon_with_overhang(x_min=x_min, x_max=x_max,
                                                                             y_min=y_min, y_max=y_max,
                                                                             h_b=hb,
                                                                             s_side=sw, h_side=hs,
                                                                             orientation=orientation,
                                                                             ds = res, dh = res, ds_side = res,
                                                                             split=True)

    mesh_surf_list = [build1, build2, shelter1, shelter2]


    import matplotlib as mpl

    plotter = pv.Plotter()
    cmap_panel = mpl.colormaps['Set3']
    for i, mesh in enumerate(mesh_surf_list):
        i=i%12
        panel_color = cmap_panel.colors[i]
        plotter.add_mesh(mesh, style="wireframe", color="blue")
        plotter.add_mesh(mesh, color=panel_color)
        plotter.add_arrows(mesh.cell_centers().points, mesh.cell_normals, mag=0.35, color='k')
    plotter.add_axes()
    plotter.show()