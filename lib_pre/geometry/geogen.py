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

import pyvista as pv
import numpy as np

def geogen_hrect(x_min, x_max, y_min, y_max, h, dx=None, dy=None, ndir = [0., 0., 1.]):

    if x_min == x_max or y_min == y_max:
        raise ValueError('  An horizontal rectangle surface cannot have x_min==x_max or y_min==y_max.')

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center = [center_x, center_y, h]

    i_size = x_max - x_min
    j_size = y_max - y_min

    if dx is None:
        i_res = 1
    else:
        i_res = int(i_size // dx)
        if i_size % dx > 0:
            print(f"    [Warning] The length of the horizontal rectangle is not multiple of the step (i_size={i_size}, dx={dx} -> i_res={i_res}).")
    if dy is None:
        j_res = 1
    else:
        j_res = int(j_size // dy)
        if j_size % dy > 0:
            print(f"    [Warning] The length of the horizontal rectangle is not multiple of the step (j_size={j_size}, dy={dy} -> j_res={j_res}).")

    hrect = pv.Plane(center, ndir, i_size, j_size, i_res, j_res).triangulate().clean()

    return hrect


def geogen_vrect(x_0, x_1, y_0, y_1, h_min, h_max, front_left, ds=None, dh=None):

    if x_0 == x_1 and y_0 == y_1:
        raise ValueError('  A vertical rectangle surface cannot have x_0==x_1 and y_0==y_1.')

    if x_0 != x_1 and y_0 != y_1:
        raise ValueError('  A vertical rectangle surface cannot have x_0!=x_1 and y_0!=y_1.')

    if h_min >= h_max:
        raise ValueError('  A vertical rectangle surface cannot have hmin >= h_max.')

    center_x = (x_0 + x_1) / 2
    center_y = (y_0 + y_1) / 2
    center_z = (h_min + h_max) / 2
    center = [center_x, center_y, center_z]

    v_x = x_1 - x_0
    v_y = y_1 - y_0
    v_arg = np.sqrt(v_x ** 2 + v_y ** 2)
    e_x = v_x / v_arg
    e_y = v_y / v_arg

    if front_left: # normal on the left when going from 0 to 1
        ndir = [-e_y, e_x, 0.]
    else:
        ndir = [e_y, -e_x, 0.]

    i_size = h_max - h_min
    j_size = v_arg

    if dh is None:
        i_res = 1
    else:
        i_res = int(i_size // dh)
        if i_size % dh > 0:
            print(f"    [Warning] The length of the vertical rectangle is not multiple of the step (i_size={i_size}, dx={dh} -> i_res={i_res}).")
    if ds is None:
        j_res = 1
    else:
        j_res = int(j_size // ds)
        if j_size % ds > 0:
            print(f"    [Warning] The length of the vertical rectangle is not multiple of the step (j_size={j_size}, dx={ds} -> j_res={j_res}).")

    vrect = pv.Plane(center, ndir, i_size, j_size, i_res, j_res).triangulate().clean()

    return vrect

def geogen_hdisc(origin, radius, dr, ds):
    r_res = int(radius // dr)
    s_res = int(2*np.pi*radius // ds)
    disc = pv.Disc(origin, 0., radius, (0.0, 0.0, 1.0), r_res, s_res).triangulate().clean()

    return disc

def geogen_hhemisphere(origin, radius, dr, ds):
    r_res = int(radius // dr)
    s_res = int(2*np.pi*radius // ds)
    hemisphere = pv.Sphere(radius, origin, (0.0, 0.0, 1.0), s_res, r_res, end_phi=90)

    return hemisphere

def geogen_hrect_ring(in_x_min, in_x_max, in_y_min, in_y_max, out_x_min, out_x_max, out_y_min, out_y_max, dx=None, dy=None):

    hrect_list = [
        # sides
        geogen_hrect(in_x_min, in_x_max, in_y_max, out_y_max, 0., dx, dy), # top
        geogen_hrect(in_x_min, in_x_max, out_y_min, in_y_min, 0., dx, dy), # bottom
        geogen_hrect(out_x_min, in_x_min, in_y_min, in_y_max, 0., dx, dy), # left
        geogen_hrect(in_x_max, out_x_max, in_y_min, in_y_max, 0., dx, dy), # right
        # corners
        geogen_hrect(out_x_min, in_x_min, in_y_max, out_y_max, 0., dx, dy), # top left
        geogen_hrect(in_x_max, out_x_max, in_y_max, out_y_max, 0., dx, dy), # top right
        geogen_hrect(out_x_min, in_x_min, out_y_min, in_y_min, 0., dx, dy), # bottom left
        geogen_hrect(in_x_max, out_x_max, out_y_min, in_y_min, 0., dx, dy), # bottom right
    ]

    return pv.merge(hrect_list)


def geogen_inflate(mesh, factor):
    # grow the mesh by a small factor uniformly
    center = np.asarray(mesh.center)
    pts = mesh.points.astype(float).copy()
    pts[:] = (pts - center) * factor + center
    mesh.points = pts

    return mesh


def geogen_rect_to_prism(rect, thickness):

    normal = rect.cell_normals[0]

    prism = rect.extrude(thickness * normal, capping=True)
    prism.translate(-thickness/2 * normal, inplace=True)

    return prism


def geogen_footprint(mesh, factor):
    # flatten the mesh in z=0
    flat = mesh.copy(deep=True)
    pts = flat.points.astype(float).copy()
    pts[:, 2] = 0.0
    flat.points = pts

    # triangulate to ensure valid polygon connectivity
    tri = flat.triangulate()

    # extract boundary edges = the projected footprint
    # returns a contour: a collection of lines
    footprint = tri.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    ).clean()

    footprint = geogen_inflate(footprint, factor)

    return footprint


def geogen_footprint_to_prism(footprint_line):
    # turn the footprint line into a surface
    from pyvista import _vtk
    tri = _vtk.vtkContourTriangulator()
    tri.SetInputData(footprint_line)
    tri.Update()
    footprint = pv.wrap(tri.GetOutput()).clean(inplace=False)

    # generate a thick closed volume (prism) from the polygon for point-in-region (pir) testing
    thickness = 100
    prism = footprint.extrude((0.0, 0.0, thickness), capping=True)
    prism.translate((0.0, 0.0, -thickness / 2.0), inplace=True)

    return prism


def geogen_split_poi(poi, prism):
    # select points located in the prism
    sel_pir = poi.select_enclosed_points(prism, check_surface=False)
    # extract boolean array (1: in the prism, 0: outside the prism)
    sel_pir_arr = np.asarray(sel_pir.point_data["SelectedPoints"].astype(bool))

    # function that extract the cells for which mask = 1
    def extract(mask):
        # security if no cells to extract
        if not np.any(mask):
            return pv.PolyData()
        # extract_points: return an unstructured grid object
        # extract_surface: create surface mesh from unstructured grid (PolyData)
        # clean: remove duplicated points etc..
        out = sel_pir.extract_points(np.flatnonzero(mask)).extract_surface()
        return out

    poi_inside = pv.PolyData(extract(sel_pir_arr).points)
    poi_outside = pv.PolyData(extract(~sel_pir_arr).points)

    return poi_inside, poi_outside



def geogen_split_roi(roi, prism):
    # select points located in the prism
    sel_pir = roi.select_enclosed_points(prism)
    # extract boolean array (1: in the prism, 0: outside the prism)
    sel_pir_arr = sel_pir.point_data["SelectedPoints"].astype(bool).astype(np.uint8)

    flag = "__inside__"
    roi.point_data[flag] = sel_pir_arr
    # pass the point data to cell data: for each cell compute the average of the boolean values among the 3 vertices
    # 0: no vertex in the prism
    # 0.33: one vertex in the prism
    # 0.66: two vertices in the prism
    # 1: all the vertices in the prism
    sel_cir = roi.point_data_to_cell_data(pass_point_data=True, categorical=False)

    # boolean array for selecting cells with all vertices inside the prism
    tol = 1e-12
    sel_cir_arr = (sel_cir.cell_data[flag] >= 1.0 - tol)

    # function that extract the cells for which mask = 1
    def extract(mask: np.ndarray) -> pv.PolyData:
        # security if no cells to extract
        if not np.any(mask):
            return pv.PolyData()
        # extract_cells: return an unstructured grid object
        # extract_surface: create surface mesh from unstructured grid (PolyData)
        # clean: remove duplicated points etc..
        out = sel_cir.extract_cells(np.flatnonzero(mask)).extract_surface().clean()
        # remove the boolean array selection data from the PolyData variables
        if flag in out.point_data: del out.point_data[flag]
        if flag in out.cell_data:  del out.cell_data[flag]
        return out

    roi_inside = extract(sel_cir_arr)
    roi_outside = extract(~sel_cir_arr)

    return roi_inside, roi_outside


def geogen_extract_points(mesh):
    return pv.PolyData(mesh.points.copy())


def geogen_duplicate_and_zshift(mesh, offset):
    # duplicate the mesh
    m = mesh.copy(deep=True)
    pts = m.points.astype(float)
    # offset along z axis
    pts[:, 2] += offset
    m.points =pts

    return m

def geogen_split_mesh(mesh_to_split, mesh_splitter):
    mesh_splitter_footprint = geogen_footprint(mesh_splitter, 1.001)
    prism = geogen_footprint_to_prism(mesh_splitter_footprint)
    mesh_to_split_inside, mesh_to_split_outside = geogen_split_roi(mesh_to_split, prism)

    return mesh_to_split_inside, mesh_to_split_outside

def geogen_split_points_internal(points_to_split, mesh_splitter):
    mesh_splitter_footprint = geogen_footprint(mesh_splitter, 1.001)
    prism = geogen_footprint_to_prism(mesh_splitter_footprint)
    points_to_split_inside, points_to_split_outside = geogen_split_poi(points_to_split, prism)

    return points_to_split_inside, points_to_split_outside

def geogen_split_points_external_vrect(points_to_split, mesh_splitter): # geogen_split_points_external_rect
    prism = geogen_rect_to_prism(mesh_splitter, 0.01)
    points_to_split_inside, points_to_split_outside = geogen_split_poi(points_to_split, prism)

    return points_to_split_inside, points_to_split_outside

def geogen_split_points_external_footprint(points_to_split, mesh_splitter):
    mesh_splitter_footprint = geogen_footprint(mesh_splitter, 0.99)
    prism = geogen_footprint_to_prism(mesh_splitter_footprint)
    points_to_split_inside, points_to_split_outside = geogen_split_poi(points_to_split, prism)

    return points_to_split_inside, points_to_split_outside

#-----------------------------------------------------------------------------------------------------------------------
# old

def geogen_peel_boundary_vertices(mesh):
    # stored faces as a flat array: [n, id0, id1, ..., id(n-1),  n, ...]
    faces = mesh.faces.reshape(-1)

    # 1) Analyze cell connectivity and count the number of uses of undirected edges (
    from collections import Counter
    # edge_use_count[(i, j)] = number of cells that contain the undirected edge {i, j}
    edge_use_count = Counter()

    # keep each cell's point ids so we can later decide which cells to drop
    cells_point_ids = []

    idx = 0
    while idx < faces.size:
        npts = int(faces[idx]); idx += 1
        cell_ids = faces[idx:idx + npts].astype(np.int64); idx += npts
        cells_point_ids.append(cell_ids)

        # build all consecutive edges (wrap-around for the closing edge)
        a = cell_ids
        b = np.roll(cell_ids, -1)
        edges = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)

        # remove duplicates inside this cell before counting
        if edges.shape[0] > 1:
            edges = np.unique(edges, axis=0)

        # update global counts
        for u, v in edges:
            edge_use_count[(int(u), int(v))] += 1

    # 2) Identify boundary vertices: endpoints of edges seen exactly once
    # if there are no boundary edges, we are done
    if not any(cnt == 1 for cnt in edge_use_count.values()):
        return mesh.copy(deep=True)

    boundary_vertex_mask = np.zeros(mesh.n_points, dtype=bool)
    for (i, j), cnt in edge_use_count.items():
        if cnt == 1:  # free boundary edge
            boundary_vertex_mask[i] = True
            boundary_vertex_mask[j] = True

    # 3) Mark cells to remove: any cell touching at least one boundary vertex
    remove_cell_ids = [
        cid for cid, ids in enumerate(cells_point_ids)
        if np.any(boundary_vertex_mask[ids])
    ]

    if not remove_cell_ids:
        return mesh.copy(deep=True)

    # 4) Extract the remaining cells and clean dangling points/topology
    keep_cell_ids = np.setdiff1d(
            np.arange(mesh.n_cells, dtype=np.int64),
            np.asarray(remove_cell_ids, dtype=np.int64)
    )

    peeled = mesh.extract_cells(keep_cell_ids).extract_surface().clean()
    return peeled


def geogen_merge(mesh_list):
    if mesh_list:
        return pv.merge(mesh_list)
    else:
        return None

def build_plane(center, ndir, i_size, j_size, i_res, j_res):
    return pv.Plane(center, ndir, i_size, j_size, i_res, j_res).triangulate().clean()


def build_polyplane(center_list, ndir_list, i_size_list, j_size_list, i_res_list, j_res_list):
    plane_list = []
    for center, ndir, i_size, j_size, i_res, j_res in zip(
            center_list, ndir_list,
            i_size_list, j_size_list,
            i_res_list, j_res_list):
        plane_list.append(pv.Plane(center, ndir, i_size, j_size, i_res, j_res).triangulate().clean())
    return pv.merge(plane_list)


def build_from_stl(filename):
    reader = pv.get_reader(filename)
    return reader.read()


if __name__ == "__main__":

    # h_rect_1 = geogen_hrect(x_min=0, x_max=3, y_min=0, y_max=3, h=0)
    # h_rect_2 = geogen_hrect(x_min=0, x_max=3, y_min=0, y_max=3, h=3, dx=1, dy=1)
    # v_rect = geogen_vrect(x_0=1, x_1=1, y_0=-1, y_1=1, h_min=0, h_max=2, front_left=False, ds=1, dh=1)
    # mesh_list = [h_rect_1, h_rect_2, v_rect]

    roig = geogen_hrect(x_min=0, x_max=20, y_min=0, y_max=20, h=0, dx=1, dy=1)

    roip = geogen_duplicate_and_zshift(roig, 1)

    poi = geogen_extract_points(roip)

    shelter = geogen_hrect(x_min=5, x_max=15, y_min=5, y_max=10, h=3, dx=1, dy=1)
    footprint = geogen_footprint(shelter, 1.001)


    prism = geogen_footprint_to_prism(footprint)


    poi_inside, poi_outside = geogen_split_poi(poi, prism)

    build = geogen_vrect(x_0=0, x_1=0, y_0=0, y_1=20, h_min=0, h_max=5, front_left=False, ds=1, dh=1)

    # remove probes on the contour of roig
    # method 1
    # prism_b = geogen_vrect_to_prism(build, 0.1)
    # trash, poi_outside = geogen_split_poi(poi_outside, prism_b)

    # method 2
    footprint_roig = geogen_footprint(roig, 0.999)
    prism_roig = geogen_footprint_to_prism(footprint_roig)
    poi_outside, trash = geogen_split_poi(poi_outside, prism_roig)

    plotter = pv.Plotter()

    plotter.add_mesh(roig, color='g')
    plotter.add_mesh(build)
    plotter.add_points(poi_inside, color='r')
    plotter.add_points(poi_outside, color='b')

    plotter.show()

    # # shelter, footprint = geogen_shelter_flat(x_min=5, x_max=15, y_min=5, y_max=10, h_min=3, h_max=6, dx=1, dy=1, dz=1, sides='North-South')
    # # shelter, footprint = geogen_shelter_gable(x_min=5, x_max=15, y_min=5, y_max=10, h_min=3, angle=30, orientation='North-South',dx=1, dy=1)
    # roig_inside, roig_outside = geogen_split_roi(roig, footprint)
    # roip_inside = geogen_duplicate_and_zshift(roig_inside, 1)
    # roip_outside = geogen_duplicate_and_zshift(roig_outside, 1)
    # roip_outside_peeled = geogen_peel_boundary_vertices(roip_outside)
    # buffer = geogen_buffer(in_x_min=0, in_x_max=20, in_y_min=0, in_y_max=20, x_margin=10)
    # mesh_list = [shelter, roig_outside, roig_inside, buffer, roip_inside, roip_outside_peeled]
    #
    # import matplotlib as mpl
    #
    # plotter = pv.Plotter()
    # cmap_panel = mpl.colormaps['Set3']
    # for i, mesh in enumerate(mesh_list):
    #     panel_color = cmap_panel.colors[i]
    #     # plot mesh contours
    #     plotter.add_mesh(mesh, style="wireframe", color="blue")
    #     # plot mesh surface with different colors for each panel
    #     plotter.add_mesh(mesh, color=panel_color)
    #     # plot normals
    #     plotter.add_arrows(mesh.cell_centers().points, mesh.cell_normals, mag=0.35, color='k')
    # plotter.add_axes()
    # plotter.show()