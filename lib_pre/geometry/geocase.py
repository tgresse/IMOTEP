from geomorph import *

def geocase_urban_scene(sl, sw, sh, tr, th, bl, bh):
    case_dict = {}

    # shelter surface
    shelter = geomorph_flat_shelter(x_min=-sl/2+7, x_max=sl/2+7,
                                    y_min=-sw/2, y_max=sw/2,
                                    h=sh,
                                    dx=2, dy=2)
    case_dict['shelter'] = shelter

    # tree surface
    tree_base, tree_top = geomorph_tree([-10., 0., th], tr, 2, 3)
    case_dict['tree'] = tree_top

    # building surface
    building = geomorph_building(x_0=20., x_1=20.,
                                 y_0=-bl/2, y_1=bl/2,
                                 h_min=0., h_max=bh,
                                 front_left=True,
                                 ds=2, dh=2)
    case_dict['building'] = building

    # floor surface
    floor = geomorph_flat_floor(x_min=-20, x_max=20,
                                 y_min=-20, y_max=20,
                                 dx=2, dy=2)


    floor_inside_shelter, floor_outside = geogen_split_mesh(floor, tree_base)
    floor_inside_tree, floor_outside = geogen_split_mesh(floor_outside, shelter)

    case_dict['floor_inside_tree'] = floor_inside_tree
    case_dict['floor_inside_shelter'] = floor_inside_shelter
    case_dict['floor_outside'] = floor_outside

    # buffer surface
    buffer = geomorph_buffer(in_x_min=-20, in_x_max=20,
                             in_y_min=-20, in_y_max=20,
                             x_margin=20, y_margin=20)
    case_dict['buffer'] = buffer

    # probe sets
    probeset = geogen_duplicate_and_zshift(floor, 1)
    probeset = geogen_extract_points(probeset)

    probeset, trash = geogen_split_points_external_footprint(probeset, floor)

    probeset_inside_shelter, probeset_outside = geogen_split_points_internal(probeset, shelter)
    probeset_inside_tree, probeset_outside = geogen_split_points_internal(probeset_outside, tree_base)

    case_dict['probeset_inside_shelter'] = probeset_inside_shelter
    case_dict['probeset_inside_tree'] = probeset_inside_tree
    case_dict['probeset_outside'] = probeset_outside

    return case_dict

if __name__ == '__main__':

    import pyvista as pv
    import matplotlib as mpl

    mpl.use("Qt5Agg")

    # put here the sample of geocase you want to visualize
    sl = 12; sw = 12; sh = 4 # shelter length, width, height
    tr = 6.01; th = 4. # tree radius, height
    bl=40; bh = 20. # building length, height
    geocase_dict = geocase_urban_scene(sl, sw, sh, tr, th, bl, bh)

    plotter = pv.Plotter()

    cmap_panel = mpl.colormaps['Set3']
    cmap_probes = mpl.colormaps['Set1']
    i_mesh = 0
    i_probe = 0
    for mesh_name, mesh in geocase_dict.items():
        if mesh_name.startswith('probeset'):
            probe_color = cmap_probes.colors[i_probe]
            plotter.add_points(mesh, render_points_as_spheres=True, point_size=10.0, color=probe_color)
            i_probe += 1
        else:
            i_mesh = i_mesh % 12
            panel_color = cmap_panel.colors[i_mesh]
            # plot mesh surface with different colors for each panel
            plotter.add_mesh(mesh, show_edges=True, color=panel_color)
            # plot normals
            plotter.add_arrows(mesh.cell_centers().points, mesh.cell_normals, mag=0.85, color='k')
            i_mesh += 1



    plotter.show_bounds(grid='back', location='outer', all_edges=True)
    plotter.add_axes()
    plotter.show()