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

from lib_imotep.core import CoreProperties, CoreSingleFacet, CoreDoubleFacet
from lib_imotep.surface import Surface

def generate_panel_list(panel_def_dict_list):
    valid_panel_type_list = ['ground', 'building', 'building_set', 'shelter', 'tree']
    
    # initialize dictionaries of lists
    # all: all the surfaces
    # no_sky: all except sky surface
    surface_list_dict = {key : [] for key in ['all', 'no_sky']}
    # core: facets connected to a core
    # domain: facets in the radiative domain
    # out_domain: facets not in the radiative domain
    # weather: facets connected directly to the weather
    facet_list_dict = {key : [] for key in ['core', 'domain', 'out_domain', 'weather']}
    # all: all the cores
    # props: all the core props
    # single_per_panel: one sample of core for each core props (used for updating core props matrices)
    core_list_dict = {key : [] for key in ['all', 'props', 'single_per_panel']}
    # domain: mesh of surface connected to the radiative domain
    # out_domain: mesh of surface not connected to the radiative domain
    # combined_domain: the merge of all the meshes of surfaces connected to the radiative domain
    mesh_list_dict = {key : [] for key in ['domain', 'out_domain', 'combined_domain']}

    name_list= []
    for panel_def_dict in panel_def_dict_list:
        name = panel_def_dict['name']
        # check if panel name does not already exist
        if name in name_list:
            raise ValueError(f"'{name}' value for parameter 'name' in panel_dict is used more than once.")
        name_list.append(name)

        panel_type = panel_def_dict['type']
        if panel_type == 'sky':
            raise ValueError("The panel type cannot be 'sky'.")

        if len(panel_def_dict['front_facet_args']) == 5 and panel_type != 'tree':
            raise ValueError(" The optional front_facet_args 'probe_transmissivity_sw' can be defined only for trees.")
        if panel_def_dict['back_facet_args'] is not None:
            if len(panel_def_dict['back_facet_args']) == 5 and panel_type != 'tree':
                raise ValueError(" The optional back_facet_args 'probe_transmissivity_sw' can be defined only for trees.")

        #---------------------------------------------------------------------------------------------------------------
        # front surface
        # create the front surface
        front_surf = Surface.front_surface(panel_type,
                                           name,
                                           panel_def_dict['mesh'],
                                           panel_def_dict['front_airzone_name'])
        # create the facets that belong to the front surface
        front_surf.create_facets(panel_def_dict['front_facet_args'])

        # whatever the panel type, the front surface is always in the radiative domain
        facet_list_dict['domain'] += front_surf.facet_list
        mesh_list_dict['domain'].append(front_surf.mesh)

        #---------------------------------------------------------------------------------------------------------------
        # back surface and cores
        # initialize the back surface
        back_surf = None
        cover_back_surf = None

        if panel_type == 'ground':
            # create the core properties object
            core_props = CoreProperties.from_list(panel_def_dict['core_args'])
            for facet in front_surf.facet_list:
                # create core element
                core = CoreSingleFacet(core_props)
                core_list_dict['all'].append(core)
                # connect facet and core
                facet.connect_core(core)
                core.connect_front_facet(facet)
            # store last created core as sample
            core_list_dict['single_per_panel'].append(core)
            core_list_dict['props'].append(core_props)
            facet_list_dict['core'] += front_surf.facet_list

        elif panel_type in ['building', 'building_set', 'shelter']:
            # create the back surface
            back_surf = Surface.back_surface(front_surf, panel_def_dict['back_airzone_name'])
            # if back_facet args is None:
            # - for shelters: raise error
            # - for buildings: create a default back_facet_args
            if panel_def_dict['back_facet_args'] is None:
                if panel_type == 'shelter':
                    raise ValueError("No back facet definition for a shelter type panel.")
                else:
                    panel_def_dict['back_facet_args'] = (1., 0., 0., 0.) # emissivity, albedo, transmissivity_lw, transmissivity_sw
            if panel_type == 'shelter':
                pass
            # create the facets that belong to the back surface
            back_surf.create_facets(panel_def_dict['back_facet_args'])
            # create the core properties object
            core_props = CoreProperties.from_list(panel_def_dict['core_args'])
            for ff, bf in zip(front_surf.facet_list, back_surf.facet_list):
                # create the core elements
                core = CoreDoubleFacet(core_props)
                core_list_dict['all'].append(core)
                # connect facets and core
                ff.connect_core(core)
                bf.connect_core(core)
                ff.connect_opposite_facet(bf)
                bf.connect_opposite_facet(ff)
                core.connect_front_facet(ff)
                core.connect_back_facet(bf)
            # store last created core as sample
            core_list_dict['single_per_panel'].append(core)
            core_list_dict['props'].append(core_props)
            facet_list_dict['core'] += front_surf.facet_list
            facet_list_dict['core'] += back_surf.facet_list
            # if shelter -> back surface in the domain
            # otherwise (buildings) -> back surface not in the domain
            if panel_type == 'shelter':
                facet_list_dict['domain'] += back_surf.facet_list
                mesh_list_dict['domain'].append(back_surf.mesh)
            else:
                facet_list_dict['out_domain'] += back_surf.facet_list
                mesh_list_dict['out_domain'].append(back_surf.mesh)

            if panel_type == 'building_set':
                # create the cover back surface
                cover_back_surf = Surface.cover_back_surface(front_surf)
                # create the facets that belong to the cover back surface with default surface properties
                cover_back_surf.create_facets((1., 0., 0., 0.)) # emissivity, albedo, transmissivity_lw, transmissivity_sw
                # no connection between front and cover_back facets
                facet_list_dict['domain'] += cover_back_surf.facet_list
                mesh_list_dict['domain'].append(cover_back_surf.mesh)
                facet_list_dict['weather'] += cover_back_surf.facet_list

        elif panel_type == 'tree':
            # create the back surface
            back_surf = Surface.back_surface(front_surf, panel_def_dict['back_airzone_name'])
            # create the facets that belong to the back surface
            back_surf.create_facets(panel_def_dict['back_facet_args'])
            for ff, bf in zip(front_surf.facet_list, back_surf.facet_list):
                # ff.set_is_tree()
                # bf.set_is_tree()
                # connect facets
                ff.connect_opposite_facet(bf)
                bf.connect_opposite_facet(ff)
            facet_list_dict['domain'] += back_surf.facet_list
            mesh_list_dict['domain'].append(back_surf.mesh)
            facet_list_dict['weather'] += front_surf.facet_list
            facet_list_dict['weather'] += back_surf.facet_list

        else:
            raise ValueError(f"'{panel_type}' value for parameter 'type' in weather_dict is not a valid. Expected value: {valid_panel_type_list}")

        surface_list_dict['all'].append(front_surf)
        surface_list_dict['no_sky'].append(front_surf)
        if back_surf is not None:
            surface_list_dict['all'].append(back_surf)
            surface_list_dict['no_sky'].append(back_surf)
        if cover_back_surf is not None:
            surface_list_dict['all'].append(cover_back_surf)
            surface_list_dict['no_sky'].append(cover_back_surf)

    # combine the meshes and store it
    combined_domain_mesh = None
    # when one mesh is merged into another mesh, its cells are added at the beginning of the host mesh. Therefore,
    # mesh_list_dict['domain'] is spanned from the end to the beginning to keep consistent order between facets and mesh elements
    for mesh in reversed(mesh_list_dict['domain']):
        if combined_domain_mesh is None:
            combined_domain_mesh = mesh.copy()
        else:
            combined_domain_mesh = combined_domain_mesh.merge(mesh.copy(), merge_points=False)
    # add area and cell centers to combined mesh polydata
    combined_domain_mesh.cell_data['cell_areas'] = combined_domain_mesh.compute_cell_sizes(length=False, area=True, volume=False)['Area']
    combined_domain_mesh.cell_data['cell_centers'] = combined_domain_mesh.cell_centers().points

    # import pyvista as pv
    # plotter = pv.Plotter()
    # plotter.add_mesh(combined_domain_mesh, style='wireframe', color='b')
    # plotter.add_mesh(combined_domain_mesh, color='w')
    # plotter.add_arrows(combined_domain_mesh.cell_centers().points, combined_domain_mesh.cell_normals, mag=0.65, color='k')
    # plotter.show()

    # store the combined mesh into to dictionary
    # note: the combined_domain list has only one element
    mesh_list_dict['combined_domain'].append(combined_domain_mesh)

    # create the default sky surface and facet at the end
    sky_surf = Surface.sky_surface()
    # create the facets
    sky_surf.create_facets((1., 0., 0., 0.)) # (emissivity, albedo, transmissivity_lw, transmissivity_sw)
    facet_list_dict['weather'] += sky_surf.facet_list
    facet_list_dict['domain'] += sky_surf.facet_list
    surface_list_dict['all'].append(sky_surf)

    return surface_list_dict, facet_list_dict, core_list_dict, mesh_list_dict