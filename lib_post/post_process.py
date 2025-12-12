import sys

sys.path.insert(1, r'../lib_imotep')
import pickle as pkl

from lib_post.plot_2D import *
from lib_post.visu_3D import visualize_fields

# ----------------------------------------------------------------------------------------------------------------------
# load results
scene_folderpath_root = '../output/'

scene_foldername_list = ['demo_shelter_building_tree'] # demo_tree, demo_shelter_building, demo_shelter_building_tree
scene_list = []
for scene_foldername in scene_foldername_list:
    with open(scene_folderpath_root + scene_foldername + '/imotep.pkl', 'rb') as data:
        scene_list.append(pkl.load(data))

# nighttime
# 14/07/2025 (Lyon)
sunrise_time_tuple = (6, 15)
sunset_time_tuple = (21, 30)
# 14/07/2025 (Phoenix)
# sunrise_time_tuple = (5, 45)
# sunset_time_tuple = (19, 45)

# save figures
save = False
save_foldername = scene_foldername_list[0]+'/graphs/'

# ----------------------------------------------------------------------------------------------------------------------
# comparative plot of airzone variables
# x-axis: time
# y-axis: variable ('hc', 'temperature', 'wind_speed')
plot_comparative(scene_list=scene_list,
                 out_type='airzone',
                 obj_name='outdoor',
                 var_name='temperature',
                 legend_label_list=scene_foldername_list,
                 figsize=(9, 6),
                 fontsize=18,
                 sunrise_time_tuple=sunrise_time_tuple,
                 sunset_time_tuple=sunset_time_tuple,
                 save=save,
                 save_folderpath=scene_folderpath_root+save_foldername)

# ----------------------------------------------------------------------------------------------------------------------
# comparative plot of surface-averaged variables
# x-axis: time
# y-axis: variable ('avg_temperature', 'avg_lw_rad_flux', 'avg_sw_rad_flux', 'avg_conv_flux', 'avg_cond_flux',
# 'avg_lw_radiosity', 'avg_sw_radiosity')
plot_comparative(scene_list=scene_list,
                 out_type='surface_averaged',
                 obj_name='floor_inside_shelter_front', # floor_inside_front ,floor_inside_shelter_front
                 var_name='avg_sw_rad_flux',
                 legend_label_list=scene_foldername_list,
                 figsize=(9, 6),
                 fontsize=18,
                 sunrise_time_tuple=sunrise_time_tuple,
                 sunset_time_tuple=sunset_time_tuple,
                 save=save,
                 save_folderpath=scene_folderpath_root+save_foldername)

# ----------------------------------------------------------------------------------------------------------------------
# comparative plot of probe variables
# x-axis: time
# y-axis: variable ('tmrt', 'comfort_index', 'sw_flux_arr', 'lw_flux_arr', 'sun_exposure')
plot_comparative(scene_list=scene_list,
                 out_type='probe',
                 obj_name='plane_us_0', # plane_ut_0 ,plane_us_0, probe_0
                 var_name='tmrt',
                 legend_label_list=scene_foldername_list,
                 figsize=(9, 6),
                 fontsize=18,
                 sunrise_time_tuple=sunrise_time_tuple,
                 sunset_time_tuple=sunset_time_tuple,
                 save=save,
                 save_folderpath=scene_folderpath_root+save_foldername)

# ----------------------------------------------------------------------------------------------------------------------
# plot weather variables
# x-axis: time
# y-axis: variable ('air_temperature', 'global_horizontal_radiation',  'direct_normal_radiation',
# 'diffuse_horizontal_radiation', 'relative_humidity', 'sky_temperature', 'wind_direction', 'wind_speed')
plot_weather(scene=scene_list[0],
             sunrise_time_tuple=sunrise_time_tuple,
             sunset_time_tuple=sunset_time_tuple,
             save=save,
             save_folderpath=scene_folderpath_root+save_foldername)

# ----------------------------------------------------------------------------------------------------------------------
# plot calculation statistics
# x-axis: time
# y-axis: number of iteration per timestep
plot_statistics(scene_list=scene_list,
                figsize=(9, 6),
                fontsize=18,
                save=save,
                save_folderpath=scene_folderpath_root+save_foldername)

# # ----------------------------------------------------------------------------------------------------------------------
# 3D visualization of the field results on surfaces and/or at probes
visualize_fields(scene=scene_list[0],
                 tree=None)

# Render all figures
import matplotlib.pyplot as plt
plt.show()