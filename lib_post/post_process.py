import sys

sys.path.insert(1, r'../lib_imotep')
import pickle as pkl

from lib_post.lib_2D_plots import *
from lib_post.lib_3D_visualization import visualize_fields

# ----------------------------------------------------------------------------------------------------------------------
# load results
scene_folderpath_root = '../output/'

# scene_foldername_list = ["exposed", "exposed_old"]
scene_foldername_list = ["shelter", "shelter_old"]
# scene_foldername_list = ["shelter_trans", "shelter_trans_old"]
scene_list = []
for scene_foldername in scene_foldername_list:
    with open(scene_folderpath_root + scene_foldername + '/imotep.pkl', 'rb') as data:
        scene_list.append(pkl.load(data))

# nightimes
# 14/07/2025
sunrise_time_tuple = (5, 45)
sunset_time_tuple = (19, 45)

# save figures
save = False

# # ----------------------------------------------------------------------------------------------------------------------
# # plot weather variables
# # x-axis: time
# # y-axis: variable ('air_temperature', 'global_horizontal_radiation',  'direct_normal_radiation', 'diffuse_horizontal_radiation', 'relative_humidity', 'sky_temperature', 'wind_direction', 'wind_speed')
# plot_weather(scene=scene_list[0],
#              save=save,
#              sunrise_time_tuple=sunrise_time_tuple,
#              sunset_time_tuple=sunset_time_tuple,
#              save_folderpath=scene_folderpath_root+scene_foldername_list[0]+'/graphs/')

# ----------------------------------------------------------------------------------------------------------------------
# comparative plot of surface averaged variables
# x-axis: time
# y-axis: variable ('tmrt', 'comfort_index', 'sw_flux_arr', 'lw_flux_arr')
plot_comparative(scene_list=scene_list,
                 out_type='surface_averaged',
                 obj_name='floor_inside_front',
                 var_name='avg_sw_rad_flux',
                 legend_label_list=scene_foldername_list,
                 # ylim=[33, 72],
                 figsize=(9, 6),
                 fontsize=18,
                 sunrise_time_tuple=sunrise_time_tuple,
                 sunset_time_tuple=sunset_time_tuple,
                 save=save,
                 save_folderpath=scene_folderpath_root+scene_foldername_list[0]+'/graphs/')

# ----------------------------------------------------------------------------------------------------------------------
# comparative plot of probe variables
# x-axis: time
# y-axis: variable ('tmrt', 'comfort_index', 'sw_flux_arr', 'lw_flux_arr')
plot_comparative(scene_list=scene_list,
                 out_type='probe',
                 obj_name='probe_0',
                 var_name='tmrt',
                 legend_label_list=scene_foldername_list,
                 # ylim=[33, 72],
                 figsize=(9, 6),
                 fontsize=18,
                 sunrise_time_tuple=sunrise_time_tuple,
                 sunset_time_tuple=sunset_time_tuple,
                 save=save,
                 save_folderpath=scene_folderpath_root+scene_foldername_list[0]+'/graphs/')


plot_comparative(scene_list=scene_list,
                 out_type='probe',
                 obj_name='probe_0',
                 var_name='sw_flux_arr',
                 legend_label_list=scene_foldername_list,
                 # ylim=[-10, 1050],
                 figsize=(9, 6),
                 fontsize=12,
                 sunrise_time_tuple=sunrise_time_tuple,
                 sunset_time_tuple=sunset_time_tuple,
                 save=save,
                 save_folderpath=scene_folderpath_root+scene_foldername_list[0]+'/graphs/')

plot_comparative(scene_list=scene_list,
                 out_type='probe',
                 obj_name='probe_0',
                 var_name='lw_flux_arr',
                 legend_label_list=scene_foldername_list,
                 # ylim=[380, 720],
                 figsize=(9, 6),
                 fontsize=12,
                 sunrise_time_tuple=sunrise_time_tuple,
                 sunset_time_tuple=sunset_time_tuple,
                 save=save,
                 save_folderpath=scene_folderpath_root+scene_foldername_list[0]+'/graphs/')

# # ----------------------------------------------------------------------------------------------------------------------
# # 3D visualization of the field results on surfaces and at probes
# visualize_fields(scene_list[0], None)

# Render all figures
import matplotlib.pyplot as plt
plt.show()