import sys
sys.path.insert(1, r'lib_pre/geometry')
import math

import lib_pre.wind.wind_factors as wind_factors
import input.geocase as geocase

# load the geocase dictionary
tr = 6.01; th = 4. # tree radius, height
geocase_dict = geocase.geocase_tree(tr, th)

pre_visualize = True

global_input_list = []

## dictionaries to fill in
general_dict = {
    'case_name'             : 'demo_tree',
    'dt'                    : 900,                  # [s] delta time between two timesteps
    'target_date'           : (2025, 7, 14),        # (year, month, day) # WARNING: DST not applied from 2038 onward !! -> use 2025 instead
    'warmup_days_num'       : 6,                    # number of days to simulate before target_date for initialization
    'userdt_warmup_days_num': 1,                    # number of warmup days to simulate with the user-defined dt (fixed dt of 1h for the rest of the period)
    'tolerance'             : 1.e-2,                # [°C]
    'relax_coef'            : .8,
    'system_orientation'    : 0,                   # [° from north, rotation of the system: clockwise] info: North direction = +Y. At 0° the system orientation is West(-X)/East(+X)
    'temperature_init'      : 25.,                  # [°C]
    'output_period'         : 'target_date',        # target_date or full_period
    'regpet_filepath'       : r'lib_pet/regpet.pkl',
    'from_save'             : False,
    'from_save_domain_type' : None,  # 'viewfactors', 'solar_prefactors', 'all', None
    'from_save_probes_type' : None,  # 'viewfactors', 'solar_prefactors', 'all', None
    'from_save_folderpath'  : r'',
    'save_data'             : True,
    'save_folderpath'       : r'output/'
}

# weather in Phoenix, Arizona, US
# weather_def_dict = {
#     'type'              : 'reconstructed', # weather_file or reconstructed
#     'filepath'          : None,
#     #                      latitude,  longitude,  altitude, tz_info
#     'location_args'     : (33.42,     -111.94,    340,      'US/Arizona'),
#     #                      ta_max,  dta_maxmin, dtas,   tdp,    ws,     wd
#     'weather_data_args' : (42.,     10.,        17.5,   5.,     3.,     0.)
# }

# weather in Paris, France
weather_def_dict = {
        'type'              : 'reconstructed', # weather_file or reconstructed
        'filepath'          : None,
        #                      latitude,  longitude,  altitude, tz_info
        'location_args'     : (45.73,     5.08,       240,      'Europe/Paris'),
        #                      ta_max,  dta_maxmin, dtas, tdp,    ws,     wd
        'weather_data_args' : (35.7,    18.9,       17.5,  13,     2.7,     0.)
    }

airzone_def_dict_list = [
    {
        'name'                   : 'outdoor',
        'type'                   : 'weather', # 'fixed', 'weather', or 'balanced'
        'ta_value'               : None,
        'hc_type'                : 'correlation', # 'fixed', or 'correlation'
        'hc_value'               : None,
        'capacity'               : None, # [J/m^3/K]
        'volume'                 : None, # [m^3]
        'internal_load'          : None, # [W]
        'effective_area_args'    : None,
#                                                                               met_height,   target_height,  roughness,  num_dir
        'wind_factor_args'       : wind_factors.compute_wind_factor_flat_ground(10,           1,              0.5,      8),
        'inflow_airzone_name'    : None
    },
    {
        'name'                  : 'under_tree',
        'type'                  : 'balanced',
        'ta_value'              : None,
        'hc_type'               : 'correlation',
        'hc_value'              : None,
        'capacity'              : 1200., # [J/m^3/K]
        'volume'                : math.pi*math.pow(tr,2)*th, # [m^3]
        'internal_load'         : 0, # [W]
#                                                                               width,                              length,                             height, num_dir
        'effective_area_args'   : wind_factors.compute_effective_area_rectangle(math.sqrt(math.pi*math.pow(tr,2)),  math.sqrt(math.pi*math.pow(tr,2)),  th,     8), # equivalent width and length (pi*r^2=L^2)
#                                                                              met_height,   target_height,  roughness,  num_dir
        'wind_factor_args'      : wind_factors.compute_wind_factor_flat_ground(10,           1,              0.5,      8),
        'inflow_airzone_name'   : 'outdoor'
    }
]

core_args_lib_dict = {
#                                   thickness,  conductivity,   capacity,   exp_coefficient,    nnodes
#                                   [m],        [W/m/K],        [J/m^3/K],  [],                 []
    'mineral_soil'              : [[2.,         2.,            2.5e6,      1.2,                20]],
    'natural_soil'              : [[2.,         1.,             2.e6,      1.2,                20]],
}

trans_tree = 0.15
facet_args_lib_dict = {
    #                      emissivity,          albedo,                 transmissivity_lw,  transmissivity_sw,  probe_direct_transmissivity_sw (for trees only, optional)
    'common_material'   : (0.95,                0.25,                   0.0,                0.0),
    'tree_crown'        : (0.98*(1-trans_tree), 0.2*(1-trans_tree),     trans_tree,         trans_tree,         trans_tree/2),
}

# warning: all the panel names must be different !
panel_def_dict_list = [
    {
        'name'              : 'floor_inside',
        'type'              : 'ground',
        'mesh'              : geocase_dict['floor_inside'],
        'core_args'         : core_args_lib_dict['natural_soil'],
        'front_facet_args'  : facet_args_lib_dict['common_material'],
        'back_facet_args'   : None,
        'front_airzone_name': 'under_tree',
        'back_airzone_name' : None
    },
    {
        'name'              : 'floor_outside',
        'type'              : 'ground',
        'mesh'              : geocase_dict['floor_outside'],
        'core_args'         : core_args_lib_dict['mineral_soil'],
        'front_facet_args'  : facet_args_lib_dict['common_material'],
        'back_facet_args'   : None,
        'front_airzone_name': 'outdoor',
        'back_airzone_name' : None
    },
    {
        'name'              : 'buffer',
        'type'              : 'ground',
        'mesh'              : geocase_dict['buffer'],
        'core_args'         : core_args_lib_dict['mineral_soil'],
        'front_facet_args'  : facet_args_lib_dict['common_material'],
        'back_facet_args'   : None,
        'front_airzone_name': 'outdoor',
        'back_airzone_name' : None
    },
    {
        'name'              : 'tree',
        'type'              : 'tree',
        'mesh'              : geocase_dict['tree'],
        'core_args'         : None,
        'front_facet_args'  : facet_args_lib_dict['tree_crown'],
        'back_facet_args'   : facet_args_lib_dict['tree_crown'],
        'front_airzone_name': 'outdoor',
        'back_airzone_name' : 'under_tree'
    }
]

# probe_set_def_list = [
#     {
#         'name'        : 'probe',
#         'mesh'        : [[0., 0., 1.]],
#         'airzone_name': 'under_tree',
#     }
# ]

probe_set_def_list = [
    {
        'name'        : 'plane_ut',
        'mesh'        : geocase_dict['probeset_inside'],
        'airzone_name': 'under_tree',
    },
    {
        'name'        : 'plane_out',
        'mesh'        : geocase_dict['probeset_outside'],
        'airzone_name': 'outdoor',
    }
]


output_def_dict = {
    'airzone' : [
    #   air-related variables
    #   name        = * or airzone names
    #   variable    = '*' for all or 'hc', 'temperature', 'wind_speed'
    #   name,       variable
        ('*',       '*'),
    ],
    'surface_averaged' : [
    #   average surface-related variables
    #   name        = * or surfaces names (warning: surface name must contain the side at the end ! Ex: 'floor_front')
    #   variable    = '*' for all or 'avg_temperature', 'avg_lw_rad_flux', 'avg_sw_rad_flux', 'avg_conv_flux', 'avg_cond_flux', 'avg_lw_radiosity', 'avg_sw_radiosity'
    #   name,       variable
        ('*',       '*'),
    ],
    'surface_field' : [
    #   surface-related variables fields
    #   name        = * or surfaces names
    #   variable    = '*' for all or 'temperature', 'lw_rad_flux', 'sw_rad_flux', 'conv_flux', 'cond_flux', 'lw_radiosity', 'sw_radiosity', 'sun_exposure'
    #    name,      variable
        ('*',       '*')
    ],
     'probe' : [
    #   metrics on probes
    #   name        = * or comfort probe names
    #   variable    = '*' for all or 'tmrt', 'comfort_index', 'sw_flux_arr', 'lw_flux_arr', 'sun_exposure'
    #   name,       variable
        ('*',       '*')
    ],
    'weather' : [
    #   weather data
    #   variable = *, air_temperature, direct_normal_radiation, diffuse_horizontal_radiation, relative_humidity, sky_temperature, wind_direction, wind_speed
        '*',
    ]
}

#-----------------------------------------------------------------------------------------------------------------------
# pre-visualization
if pre_visualize:
    import numpy as np
    import pyvista as pv
    import matplotlib as mpl
    mpl.use("Qt5Agg")

    plotter = pv.Plotter()

    cmap_panel = mpl.colormaps['Set3']
    for i, panel in enumerate(panel_def_dict_list):
        i = i%12
        panel_color = cmap_panel.colors[i]
        # plot mesh surface with different colors for each panel
        plotter.add_mesh(panel['mesh'], show_edges=True, color=panel_color)
        # plot normals
        plotter.add_arrows(panel['mesh'].cell_centers().points, panel['mesh'].cell_normals, mag=0.85, color='k')

    cmap_probes = mpl.colormaps['Set1']
    for i, probe_set in enumerate(probe_set_def_list):
        probe_color = cmap_probes.colors[i]
        if type(probe_set['mesh']) is list:
            points = np.array(probe_set['mesh'])
        else: # is a PolyData
            points = probe_set['mesh'].points
        plotter.add_points(points, render_points_as_spheres=True, point_size=10.0, color=probe_color)

    plotter.show_bounds(grid='back', location='outer', all_edges=True)
    plotter.add_axes()
    plotter.show()

#-----------------------------------------------------------------------------------------------------------------------
# storing for sensitivity analysis
global_input_list.append((general_dict,
                           weather_def_dict,
                           panel_def_dict_list,
                           airzone_def_dict_list,
                           probe_set_def_list,
                           output_def_dict))

