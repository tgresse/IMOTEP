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

import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyvista as pv
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['grid.linewidth'] = 1.3

def plot_geometry(scene):
        """
        Plots the geometry of the simulation domain (excluding the sky).
        """
        plotter = pv.Plotter()
        cmap_surf = mpl.colormaps['Set3']
        for i, surface in enumerate(scene.surface_list_dict['no_sky']):
            i=i%12
            surf_color = cmap_surf.colors[i]
            plotter.add_mesh(surface.mesh, style="wireframe", color="b")
            plotter.add_mesh(surface.mesh, color=surf_color)
            plotter.add_arrows(surface.mesh.cell_centers().points, surface.mesh.cell_normals, mag=0.4, color="k")
        plotter.show_bounds(grid='back', location='outer', all_edges=True)
        plotter.add_axes()
        plotter.show()


def plot_statistics(scene_list,
                    xlim='auto',
                    ylim='auto',
                    remove_legend=False,
                    figsize=(6, 5),
                    fontsize=12,
                    sunrise_time_tuple=(6,30),
                    sunset_time_tuple=(20,30),
                    save=False,
                    save_folderpath=''):
        # ensure all scenes share the same time index
        out_time_list = [
            scene.time_management_dict['date_list'].tz_localize(None)
            for scene in scene_list
        ]
        reference_time = out_time_list[0]
        if not all(t.equals(reference_time) for t in out_time_list):
            raise ValueError("Not all DateTimeIndex entries are identical across scenes.")
        out_time = reference_time

        # gather data
        out_data_list = [scene.solve_stat_list for scene in scene_list]
        col_name_list = [scene.general_dict['case_name'] for scene in scene_list]
        df_glob = pd.DataFrame(index=out_time, data=np.array(out_data_list).T, columns=col_name_list)

        # plot
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        sns.lineplot(df_glob, palette='Greys', dashes=False, markers=True)

        # add night background shading
        sunrise_m = sunrise_time_tuple[0] * 60 + sunrise_time_tuple[1]
        sunset_m = sunset_time_tuple[0] * 60 + sunset_time_tuple[1]
        minutes_since_midnight = out_time.hour * 60 + out_time.minute
        night_mask = (minutes_since_midnight < sunrise_m) | (minutes_since_midnight >= sunset_m)

        ax.fill_between(
                out_time, 0, 1,
                where=night_mask,
                color='gainsboro',
                alpha=0.8,
                transform=ax.get_xaxis_transform()
        )

        # configure axis
        ax.grid(True)
        if remove_legend:
            leg = ax.get_legend()
            if leg:
                leg.remove()

        ax.set_xlabel('Local time [d]', fontsize=fontsize)
        ax.set_ylabel('$n_{iter}$', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)

        if xlim != 'auto':
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([out_time[0], out_time[-1]])

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=60))

        if ylim != 'auto':
            ax.set_ylim(ylim)

        # save if needed
        if save:
            save_path = Path(save_folderpath)
            save_path.mkdir(parents=True, exist_ok=True)
            for ext in ['png', 'pdf']:
                plt.savefig(save_path / f'solve_stat.{ext}', bbox_inches='tight')

        fig.canvas.manager.set_window_title('solve_stat')


def plot_weather(scene,
                 xlim='auto',
                 ylim='auto',
                 figsize=(5, 9),
                 fontsize=12,
                 linewidth=2,
                 sunrise_time_tuple=(6,30),
                 sunset_time_tuple=(20,30),
                 save=False,
                 save_folderpath=''):
        out_dict = scene.output_dict['weather']

        # get variables
        available_vars = [var for _, var in out_dict.keys()]

        # get time data
        out_time = scene.time_management_dict['target_date_list'].tz_localize(None)

        # set up figure and axes
        fig, axs = plt.subplots(4, 1, figsize=figsize, layout='constrained')

        # define plotting configuration
        plot_config = {
            'air_temperature'             : (0, 'r', 'solid', 'air temperature'),
            'sky_temperature'             : (0, 'dimgrey', 'dashed', 'sky temperature'),
            'global_horizontal_radiation' : (1, 'gold', 'solid', 'global'),
            'diffuse_horizontal_radiation': (1, 'gold', 'dashed', 'diffuse'),
            'relative_humidity'           : (2, 'b', 'solid', 'relative humidity'),
            'wind_speed'                  : (3, 'g', 'solid', 'wind speed'),
            'wind_direction'              : (3, 'g', '', 'wind direction'),
        }

        # plot each variable from output dict (except wind speed)
        for var_name in available_vars:
            if var_name == 'direct_normal_radiation':
                continue
            ax_idx, color, style, label = plot_config[var_name]
            data = out_dict[(None, var_name)][-1]
            if var_name == 'wind_direction':
                ax32 = axs[ax_idx].twinx()
                ax32.plot(out_time, data, color=color, linestyle=style, marker='o', markersize=3, linewidth=linewidth, label=label)
            else:
                axs[ax_idx].plot(out_time, data, color=color, linestyle=style, linewidth=linewidth, label=label)

        # add night background shading
        sunrise_time = sunrise_time_tuple[0] * 60 + sunrise_time_tuple[1]
        sunset_time = sunset_time_tuple[0] * 60 + sunset_time_tuple[1]
        time = out_time.hour * 60 + out_time.minute
        for ax in axs:
            night_mask = (time < sunrise_time) | (time >= sunset_time)
            ax.fill_between(
                out_time, 0, 1,
                where=night_mask,
                color='gainsboro',
                alpha=0.8,
                transform=ax.get_xaxis_transform()
            )

        # configure axes
        ylabel_list = ['Temperatures [°C]', 'Solar fluxes [W/m²]', 'Relative Humidity [%]', 'Wind speed (━) [m/s]', 'Wind direction (•) [°]']

        for i, ax in enumerate(axs):
            ax.grid(True)
            ax.set_ylabel(ylabel_list[i], fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize)
            if xlim != 'auto':
                ax.set_xlim(xlim)
            else:
                ax.set_xlim([out_time[0], out_time[-1]])
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            if ylim != 'auto':
                ax.set_ylim(ylim)
            if i < 2:
                ax.legend(fontsize=fontsize-2)
            if i < 3:
                ax.set_xticklabels([])

        ax32.set_ylabel(ylabel_list[-1], fontsize=fontsize)

        axs[3].set_xlabel('Local time [h]', fontsize=fontsize)

        # save figure if requested
        if save:
            save_path = Path(save_folderpath)
            save_path.mkdir(parents=True, exist_ok=True)
            for ext in ['png', 'pdf']:
                    plt.savefig(save_path / f'weather_data.{ext}', bbox_inches='tight')

        fig.canvas.manager.set_window_title('Weather data')


def sim_vs_obs_error_metrics(
    obs,
    pred,
    align="to_obs",            # "to_obs" | "to_pred" | "union"
    interpolate="time",        # pandas interpolate method; "time" for DateTimeIndex
    limit_area="inside",       # avoid extrapolation
    column_map=None,
    group_size=None, # e.g., 6 if pred has 6 vars per simulation
    obs_select=None, # if comparing many pred columns to one obs column
    plot_versus=None,
    plot_var_name='',
    plot_save=False,
    plot_save_folderpath=None,
) -> pd.DataFrame:
    """
    Compute RMSE, MAE, MBE between obs and pred with smart column pairing.

    Pairing rules (in order of precedence):
      1) `column_map` provided: use it.  (pred_col -> obs_col)
      2) If `obs_select` is set or obs has a single column: compare each pred col to that obs col.
      3) If `group_size` is set and len(pred)%group_size==0 and len(obs)==group_size:
           map each group of pred columns (by order) to obs columns (by order).
      4) If len(obs.columns) == len(pred.columns): map by positional order.
      5) Fallback: canonical-name match (lowercase, strip spaces/units/punct).

    Returns a DataFrame indexed by ["RMSE","MAE","MBE"] with one column per
    compared pred column (labelled 'pred ~vs~ obs' when needed).
    """

    # --- Normalize inputs ---
    if isinstance(obs, pd.Series):
        obs = obs.to_frame()
    else:
        obs = obs.copy()
    if isinstance(pred, pd.Series):
        pred = pred.to_frame()
    else:
        pred = pred.copy()

    # --- Timezone sanity ---
    if (obs.index.tz is None) != (pred.index.tz is None):
        raise ValueError("Both indexes must be either tz-aware or tz-naive.")
    if obs.index.tz is not None and pred.index.tz is not None and obs.index.tz != pred.index.tz:
        pred = pred.tz_convert(obs.index.tz)

    # --- Build column pairs ---
    def canon(s: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    pairs = []  # (pred_col, obs_col)

    if column_map is not None:
        if isinstance(column_map, dict):
            pairs = [(p, o) for p, o in column_map.items() if p in pred.columns and o in obs.columns]
        else:
            pairs = [(p, o) for (p, o) in column_map if p in pred.columns and o in obs.columns]

    if not pairs:
        if obs_select is not None:
            if obs_select not in obs.columns:
                raise KeyError(f"obs_select '{obs_select}' not found in obs columns.")
            pairs = [(p, obs_select) for p in pred.columns]
        elif obs.shape[1] == 1:
            oc = obs.columns[0]
            pairs = [(p, oc) for p in pred.columns]
        elif group_size is not None and len(pred.columns) % group_size == 0 and obs.shape[1] == group_size:
            # map each group of pred to obs by order
            oc = list(obs.columns)
            pcs = list(pred.columns)
            for g in range(len(pcs)//group_size):
                for i in range(group_size):
                    pairs.append((pcs[g*group_size + i], oc[i]))
        elif obs.shape[1] == pred.shape[1]:
            # positional order
            pairs = list(zip(pred.columns.tolist(), obs.columns.tolist()))
        else:
            # canonical name match
            obs_c = {canon(c): c for c in obs.columns}
            for p in pred.columns:
                cp = canon(p)
                if cp in obs_c:
                    pairs.append((p, obs_c[cp]))

    if not pairs:
        raise ValueError("Could not infer column pairing between obs and pred.")

    # --- Helpers to align with interpolation ---
    def interp_to(index_target: pd.DatetimeIndex, s: pd.Series) -> pd.Series:
        # add target times, interpolate, then select exactly target times
        s2 = (s.sort_index()
               .reindex(s.index.union(index_target))
               .interpolate(method=interpolate, limit_area=limit_area)
               .reindex(index_target))
        return s2

    # Choose alignment index
    if align == "to_obs":
        base_index = obs.index
    elif align == "to_pred":
        base_index = pred.index
    elif align == "union":
        base_index = obs.index.union(pred.index)
    else:
        raise ValueError("align must be 'to_obs', 'to_pred', or 'union'.")

    # --- Compute metrics per pair ---
    results = {}
    df_glob_list = []
    for pcol, ocol in pairs:
        # Select series
        s_obs = obs[ocol].astype(float)
        s_pred = pred[pcol].astype(float)

        # Align/interpolate
        if align == "to_obs":
            idx = s_obs.index
            sp = interp_to(idx, s_pred)
            so = s_obs.reindex(idx)  # keep raw obs values
        elif align == "to_pred":
            idx = s_pred.index
            so = interp_to(idx, s_obs)
            sp = s_pred.reindex(idx)
        else:  # union
            idx = base_index
            so = interp_to(idx, s_obs)
            sp = interp_to(idx, s_pred)

        # Drop rows with NaNs in either series
        valid = ~(so.isna() | sp.isna())
        so = so[valid]
        sp = sp[valid]
        if so.empty:
            continue

        err = sp - so
        rmse = np.sqrt((err**2).mean())
        mae  = err.abs().mean()
        mbe  = err.mean()

        # Column label
        label = f"{pcol} ~vs~ {ocol}"
        results[label] = pd.Series({"RMSE": rmse, "MAE": mae, "MBE": mbe})

        df_glob = pd.concat([so, sp], axis=1)
        df_glob_list.append(df_glob)

    if not results:
        raise ValueError("No valid pairs produced non-empty overlap after alignment.")

    if plot_versus:
        n = len(df_glob_list)
        if n > 1:
            # make just enough axes; 2 columns layout
            fig, axs = plt.subplots(3, 2, layout='constrained', figsize=(8.2, 10))
            axs = np.array(axs).ravel()  # <-- flatten to 1-D list of axes
            fontsize = 10
        else:
            fig, ax = plt.subplots(layout='constrained', figsize=(5, 4))
            axs = np.array([ax])
            fontsize = 12

        for i, ax in enumerate(axs[:n]):  # <-- only as many axes as datasets
            df_glob = df_glob_list[i].copy()
            xcol, ycol = df_glob.columns[0], df_glob.columns[1]

            # 1:1 reference line
            xmin, xmax = df_glob[xcol].min(), df_glob[xcol].max()
            ax.plot([xmin, xmax], [xmin, xmax], 'k-', linewidth=1)

            # Color by hour if you have a DateTimeIndex; otherwise no hue
            if hasattr(df_glob.index, "hour"):
                hue_vec = df_glob.index.hour
                norm = plt.Normalize(hue_vec.min(), hue_vec.max())
                cmap = 'viridis'
                sns.scatterplot(data=df_glob, ax=ax, x=xcol, y=ycol, hue=hue_vec,
                                palette=cmap, s=40, edgecolor="none")
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cb = fig.colorbar(sm, ax=ax)
                cb.set_label('Time [h]', fontsize=fontsize)
                ax.get_legend().remove()
            else:
                sns.scatterplot(data=df_glob, ax=ax, x=xcol, y=ycol, s=40, edgecolor="none")

            ax.set_xlabel('Observation', fontsize=fontsize)
            ax.set_ylabel('Prediction', fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize)
            if n > 1:
                ax.set_title(df_glob.columns[0].replace('obs_', ''), fontweight='bold', fontsize=fontsize)
            ax.set_aspect('equal')

            if plot_save:
                save_path = Path(plot_save_folderpath)
                save_path.mkdir(parents=True, exist_ok=True)
                for ext in ['png', 'pdf']:
                    plt.savefig(save_path / f'{plot_var_name}_sim_vs_obs.{ext}', bbox_inches='tight')

        # Use the first panel's var names for the window title (optional)
        fig.canvas.manager.set_window_title(f'{plot_var_name}_sim_vs_obs')

    out = pd.DataFrame(results).reindex(["RMSE","MAE","MBE"])
    return out

def sim_vs_sim_error_metrics(s1, s2):
    """Return min/max diff and RMSE of (s1 - s2) over aligned non-NaN times."""
    pair = pd.concat([pd.Series(s1), pd.Series(s2)], axis=1).dropna()
    if pair.empty:
        return pd.Series({"min_diff": np.nan, "max_diff": np.nan, "rmse": np.nan})
    diff = pair.iloc[:, 0] - pair.iloc[:, 1]
    return pd.Series({
        "min_diff": float(diff.min()),
        "max_diff": float(diff.max()),
        "rmse": float(np.sqrt(np.mean(np.square(diff.values))))
    })


def plot_comparative(
    scene_list,
    out_type,
    obj_name,
    var_name,
    legend_label_list,
    separate_plots=False,
    plot_versus=False,
    obs_df=None,
    xlim='auto',
    ylim='auto',
    remove_legend=False,
    figsize=(6, 5),
    fontsize=12,
    linewidth=2.0,
    sunrise_time_tuple=(6, 30),
    sunset_time_tuple=(20, 30),
    save=False,
    save_folderpath=''
):
    # ---- 0) Validate arguments ------------------------------------------------
    if len(scene_list) != len(legend_label_list):
        raise ValueError("scene_list and legend_label_list must have the same length.")

    for scene in scene_list:
        available_objs = {obj.name for obj, _ in scene.output_dict[out_type].values()}
        available_vars = {var for _, var in scene.output_dict[out_type].keys()}
        if obj_name not in available_objs:
            raise ValueError(f"Object '{obj_name}' not found in a scene. Available: {sorted(available_objs)}")
        if var_name not in available_vars:
            raise ValueError(f"Variable '{var_name}' not found in a scene. Available: {sorted(available_vars)}")

    out_time_list = [scene.time_management_dict['target_date_list'].tz_localize(None) for scene in scene_list]
    reference_time = out_time_list[0]
    if not all(t.equals(reference_time) for t in out_time_list):
        raise ValueError("Not all DateTimeIndex entries are identical across scenes.")
    out_time = reference_time

    # ---- 1) Mappings (labels/palettes) ---------------------------------------
    dir_list = ['top', 'bottom', 'north', 'south', 'east', 'west']
    is_rad = var_name in ['sw_flux_arr', 'lw_flux_arr']

    cmap_dict = {
        'hc'              : 'Oranges',
        'temperature'     : 'Purples',
        'wind_speed'      : 'Reds',
        'avg_temperature' : 'Oranges',
        'avg_lw_rad_flux' : 'Purples',
        'avg_sw_rad_flux' : 'Reds',
        'avg_cond_flux'   : 'Blues',
        'avg_conv_flux'   : 'Greens',
        'comfort_index'   : ['g'],
        'tmrt'            : 'plasma',
        'lw_flux_arr'     : 'tab10',
        'sw_flux_arr'     : 'tab10',
    }
    ylabel_dict = {
        'temperature'      : r'$T~[°C]$',
        'hc'               : r'$h_c~[W.m^{-2}.K^{-1}]$',
        'wind_speed'       : r'$U_w~[m.s^{-1}]$',
        'avg_temperature'  : r'$T~[°C]$',
        'avg_lw_rad_flux'  : r'$L^{*}~[W.m^{-2}]$',
        'avg_sw_rad_flux'  : r'$K^{*}~[W.m^{-2}]$',
        'avg_cond_flux'    : r'$Q_G~[W.m^{-2}]$',
        'avg_conv_flux'    : r'$Q_H~[W.m^{-2}]$',
        'avg_lw_radiosity' : r'$\mathcal{J}_{LW}~[W.m^{-2}]$',
        'avg_sw_radiosity' : r'$\mathcal{J}_{SW}~[W.m^{-2}]$',
        'comfort_index'    : r'PET [°C]',
        'tmrt'             : r'$T_{mrt}~[°C]$',
        'lw_flux_arr'      : r'$L^{\downarrow}~[W.m^{-2}]$',
        'sw_flux_arr'      : r'$K^{\downarrow}~[W.m^{-2}]$',
    }
    if var_name not in cmap_dict:
        raise ValueError(f"No cmap defined for variable '{var_name}'.")
    if var_name not in ylabel_dict:
        raise ValueError(f"No ylabel defined for variable '{var_name}'.")

    # Dash patterns for simulations in separate_plots mode
    dash_cycle = ['', (3, 1), (1, 1), (5, 2), (1, 1), (6, 2)]
    if is_rad:
        n = 6
    else:
        n = 1
    sim_color_separate = sns.color_palette(cmap_dict[var_name], n) # simulations color when separate_plots=True

    # ---- 2) Build DataFrames --------------------------------------------------
    df_list = []
    if is_rad:
        for i, scene in enumerate(scene_list):
            out_vals = scene.output_dict[out_type][(obj_name, var_name)][-1]  # (T, 6)
            col_names = [f"{legend_label_list[i]}_{d}" for d in dir_list]
            df_list.append(pd.DataFrame(index=out_time, data=out_vals, columns=col_names))
    else:
        for i, scene in enumerate(scene_list):
            out_vals = scene.output_dict[out_type][(obj_name, var_name)][-1]  # (T,)
            df_list.append(pd.DataFrame(index=out_time, data=out_vals, columns=[legend_label_list[i]]))

    df_glob = pd.concat(df_list, axis=1)

    # Prepare observations (copy)
    obs_df_proc = None
    if obs_df is not None:
        obs_df_proc = obs_df.copy()
        if is_rad:
            obs_df_proc.columns = [f"obs_{d}" for d in dir_list]
        else:
            obs_df_proc.columns = ["obs"]

    # ---- 3) Plotting ----------------------------------------------------------
    if separate_plots and is_rad:
        # One subplot per direction
        fig, axs = plt.subplots(3, 2, figsize=figsize, layout='constrained')
        axs = np.asarray(axs).ravel()

        for i_dir, ax in enumerate(axs):
            dir_name = dir_list[i_dir]

            # SIMULATIONS: same color, different dash per scene
            for j, (df_scene, label) in enumerate(zip(df_list, legend_label_list)):
                col = f"{label}_{dir_name}"
                sns.lineplot(
                    x=df_scene.index, y=df_scene[col],
                    ax=ax, linewidth=linewidth, label=label,
                    color=sim_color_separate[i_dir], dashes=dash_cycle[j % len(dash_cycle)]
                )

            # OBSERVATIONS: markers only (no line)
            if obs_df_proc is not None:
                col_obs = f"obs_{dir_name}"
                if col_obs in obs_df_proc.columns:
                    sns.lineplot(
                        x=obs_df_proc.index, y=obs_df_proc[col_obs],
                        ax=ax, marker='o', linestyle='None',
                        color=sim_color_separate[i_dir], label='obs'
                    )

            ax.set_title(dir_name.capitalize(), fontsize=fontsize, fontweight='bold')

        # Legend handling
        if remove_legend:
            for ax in axs:
                leg = ax.get_legend()
                if leg: leg.remove()
        else:
            axs[-1].legend(fontsize=fontsize-2, frameon=True)

    else:
        # Single combined axes (either non-directional or combined view)
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        axs = [ax]

        # SIMULATIONS: palette + solid lines
        sns.lineplot(
            data=df_glob, ax=ax, linewidth=linewidth,
            palette=sim_color_separate, linestyle='-', dashes=None
        )

        # OBSERVATIONS: markers + dashed line (2,2)
        if obs_df_proc is not None:
            if separate_plots:
                linewidth = 0
                dashes = [''] * n
            else:
                dashes = [(2, 2)] * n
            # Plot in black to stand out, one line per obs column (OK for directional)
            sns.lineplot(
                data=obs_df_proc, ax=ax,
                linewidth=linewidth, marker='o',
                palette=sim_color_separate, dashes=dashes
            )

        # if not remove_legend:
        #     ax.legend(fontsize=fontsize-2, frameon=True)

    # ---- 4) Common axis formatting + night shading ----------------------------
    sunrise_m = sunrise_time_tuple[0] * 60 + sunrise_time_tuple[1]
    sunset_m  = sunset_time_tuple[0] * 60 + sunset_time_tuple[1]
    minutes_since_midnight = out_time.hour * 60 + out_time.minute
    night_mask = (minutes_since_midnight < sunrise_m) | (minutes_since_midnight >= sunset_m)

    for ax in axs:
        ax.fill_between(
            out_time, 0, 1,
            where=night_mask,
            color='gainsboro',
            alpha=0.8,
            transform=ax.get_xaxis_transform()
        )

        ax.grid(True)
        if remove_legend:
            leg = ax.get_legend()
            if leg:
                leg.remove()

        ax.set_xlabel('Local time [h]', fontsize=fontsize)
        ax.set_ylabel(ylabel_dict[var_name], fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)

        if xlim != 'auto':
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([out_time[0], out_time[-1]])

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))

        if ylim != 'auto':
            ax.set_ylim(ylim)

    try:
        fig.canvas.manager.set_window_title(f'{out_type} - {obj_name} - {var_name}')
    except Exception:
        pass

    # ---- 5) Save --------------------------------------------------------------
    if save:
        save_path = Path(save_folderpath)
        save_path.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(save_path / f'{out_type}_{obj_name}_{var_name}.{ext}', bbox_inches='tight')

    # ---- 6) Metrics: simulation–simulation --------------------------------------
    # Build pairwise metrics across simulations.
    rows = []
    if is_rad:
        # Per direction, compare same-direction series across scenes
        for d in dir_list:
            # collect columns like "<label>_top" for all labels
            cols = [f"{lab}_{d}" for lab in legend_label_list if f"{lab}_{d}" in df_glob.columns]
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    a, b = cols[i], cols[j]
                    met = sim_vs_sim_error_metrics(df_glob[a], df_glob[b])
                    rows.append({
                        "direction": d,
                        "scene_a"  : a.split("_" + d)[0],
                        "scene_b"  : b.split("_" + d)[0],
                        **met.to_dict()
                    })
    else:
        # Single series per scene: compare all pairs
        cols = [lab for lab in legend_label_list if lab in df_glob.columns]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                met = sim_vs_sim_error_metrics(df_glob[a], df_glob[b])
                rows.append({
                    "direction": None,
                    "scene_a"  : a,
                    "scene_b"  : b,
                    **met.to_dict()
                })

    sim_sim_metrics = pd.DataFrame(rows, columns=["direction", "scene_a", "scene_b", "min_diff", "max_diff", "rmse"])

    # ---- 6) Metrics: simulation-observation -----------------------------------------------------------
    if obs_df is not None:
        if is_rad:
            col_map = {}
            for lab in legend_label_list:
                for d in dir_list:
                    sim_col = f"{lab}_{d}"
                    obs_col = f"obs_{d}"
                    if obs_col in obs_df_proc.columns:
                        col_map[sim_col] = obs_col
        else:
            col_map = {lab: "obs" for lab in legend_label_list}

        sim_obs_metrics = sim_vs_obs_error_metrics(obs_df_proc,
                                                df_glob,
                                                column_map=col_map,
                                                align="to_obs",
                                                plot_versus=plot_versus,
                                                plot_var_name=var_name,
                                                plot_save=save,
                                                plot_save_folderpath=save_folderpath)

    # ---- 7) Return results -------------------------------------------------------
    result = {"sim_sim_metrics": sim_sim_metrics}

    if obs_df is not None:
        result["sim_obs_metrics"] = sim_obs_metrics

    return result


def plot_abricocoda_stack(
    scene,
    tree,
    figsize=(7, 5),
    fontsize=12,
    sunrise_time_tuple=(6,30),
    sunset_time_tuple=(20,30),
    ylim='auto',
    save=False,
    save_folderpath='',
    area_factor=1.0,             # NEW: multiply counts -> area (m²)
    shelter_area_m2=None,        # NEW: optional horizontal line at this area
    remove_legend=False,
):
    """
    Stacked *area* over time of:
      - Abricocoda & shaded (green)
      - Abricocoda & not shaded (blue)
      - Not Abricocoda & shaded (orange)

    Y-axis is area [m²]. Each probe contributes `area_factor` m².
    Example: if probes are on a 2 m grid and each represents 2 m² in your convention,
    set area_factor=2. (More generally, set it to the area per probe.)

    Parameters
    ----------
    area_factor : float
        Multiplier converting probe counts to area (m²).
    shelter_area_m2 : float or None
        If provided, draw a dotted horizontal line at this area (m²).
    """

    if scene is None or tree is None:
        raise ValueError("Both 'scene' and 'tree' must be provided.")
    if area_factor <= 0:
        raise ValueError("area_factor must be positive.")

    # --- Time axis ---
    out_time_scene = scene.time_management_dict['target_date_list'].tz_localize(None)
    out_time_tree  = tree.time_management_dict['target_date_list'].tz_localize(None)
    if not out_time_scene.equals(out_time_tree):
        raise ValueError("Scene and tree time axes differ.")
    out_time = out_time_scene

    # --- Tree comfort (first comfort_index found) ---
    tree_probe_keys = list(tree.output_dict['probe'].keys())
    tree_key = next(k for k in tree_probe_keys if k[-1] == 'comfort_index')
    tree_ser = pd.Series(
        np.asarray(tree.output_dict['probe'][tree_key][-1]).reshape(-1),
        index=out_time,
        name='tree'
    )

    # --- Scene probes (comfort & sun) ---
    scene_keys = list(scene.output_dict['probe'].keys())
    probe_names = sorted({k[0] for k in scene_keys if k[-1] == 'comfort_index'})
    if not probe_names:
        raise ValueError("No comfort_index probes found in scene.")
    n_probes = len(probe_names)

    col_comfort, col_sun = {}, {}
    for name in probe_names:
        comfort_idx = scene.output_dict['probe'][(name, 'comfort_index')][-1]
        sun_exp     = scene.output_dict['probe'][(name, 'sun_exposure')][-1]
        col_comfort[name] = np.asarray(comfort_idx).reshape(-1)
        col_sun[name]     = np.asarray(sun_exp).reshape(-1)

    comfort_df = pd.DataFrame(col_comfort, index=out_time)
    sun_df     = pd.DataFrame(col_sun, index=out_time)

    # --- Booleans per probe/time ---
    abricocoda = comfort_df.le(tree_ser, axis=0)   # comfort <= tree
    shade      = (sun_df == 0)                     # shaded

    # --- Counts per time ---
    A_shade   = (abricocoda & shade).sum(axis=1)      # green
    A_sun     = (abricocoda & ~shade).sum(axis=1)     # blue
    NA_shade  = (~abricocoda & shade).sum(axis=1)     # orange

    counts_df = pd.DataFrame(
        {'A_shade': A_shade, 'A_sun': A_sun, 'NA_shade': NA_shade},
        index=out_time
    )

    # --- Convert counts -> area ---
    areas_df = counts_df * float(area_factor)
    total_probe_area = n_probes * float(area_factor)

    # --- Plot (stacked area) ---
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')

    ax.stackplot(
        areas_df.index,
        areas_df['A_shade'],
        areas_df['A_sun'],
        areas_df['NA_shade'],
        labels=['Abricocoda & shade', 'Abricocoda & sun', 'Not Abricocoda & shade'],
        colors=['forestgreen', 'royalblue', 'orange'],
        alpha=0.9
    )

    # Night background
    sunrise_time = sunrise_time_tuple[0] * 60 + sunrise_time_tuple[1]
    sunset_time = sunset_time_tuple[0] * 60 + sunset_time_tuple[1]
    time = out_time.hour * 60 + out_time.minute
    night_mask = (time < sunrise_time) | (time >= sunset_time)
    ax.fill_between(
        out_time, 0, 1,
        where=night_mask,
        color='gainsboro', alpha=0.6,
        transform=ax.get_xaxis_transform()
    )

    # --- Horizontal dotted lines (no legend entries) ---
    ax.axhline(y=total_probe_area, linestyle=':', linewidth=2, color='black')
    shelter_val = None
    if shelter_area_m2 is not None:
            shelter_val = float(shelter_area_m2)
            ax.axhline(y=shelter_val, linestyle=':', linewidth=2, color='black')

    # Axes/labels/formatting
    ax.set_ylabel('Area [m²]', fontsize=fontsize)
    ax.set_xlabel('Local time [h]', fontsize=fontsize)
    ax.grid(True, which='major')
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlim([out_time[0], out_time[-1]])

    # Ensure y-limit includes both lines
    stack_max_area = areas_df.sum(axis=1).max()
    ymax_candidate = total_probe_area
    if shelter_area_m2 is not None:
        ymax_candidate = max(ymax_candidate, float(shelter_area_m2))
    y_top = max(stack_max_area, ymax_candidate)
    ax.set_ylim(0, y_top * 1.05 if y_top > 0 else 1)

    # --- Put the labels on the Y-AXIS as tick labels ---
    current_ticks = ax.get_yticks()
    extras = [total_probe_area, shelter_val]

    # Merge current ticks with our two special values, then format labels.
    merged = list(current_ticks)
    for v in extras:
            if v is None:
                    continue
            if not any(abs(v - b) < 1e-6 for b in merged):
                    merged.append(v)
    new_ticks = np.array(sorted(merged))

    # Build labels: replace those two values with S_tot / S_shelter, keep others numeric
    labels = []
    for y in new_ticks:
            if abs(y - total_probe_area) < 1e-6:
                    labels.append(r'$S_{\mathrm{tot}}$')
            elif (shelter_val is not None) and abs(y - shelter_val) < 1e-6:
                    labels.append(r'$S_{\mathrm{shelter}}$')
            else:
                    labels.append(f'{y:g}')
    ax.set_yticks(new_ticks)
    ax.set_yticklabels(labels)

    if ylim != 'auto':
            ax.set_ylim(ylim)

    # Legend (only the stacked areas)
    if not remove_legend:
        ax.legend(loc='upper left', fontsize=fontsize-1, frameon=True)

    if save:
        save_path = Path(save_folderpath)
        save_path.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            plt.savefig(save_path / f'abricocoda_stack_{scene.general_dict['case_name']}.{ext}', bbox_inches='tight')

    fig.canvas.manager.set_window_title('abricocoda_stack ' + scene.general_dict['case_name'])


def plot_abricocoda_diff_stack(
    scene,
    scene_ref,
    tree,
    title="Differential stacked areas (with − without) by category",
    figsize=(7, 5),
    fontsize=12,
    area_factor=1.0,          # area each probe represents [m²]
    ylim='auto',
    remove_legend=False,
    colors=('forestgreen', 'royalblue', 'orange'),  # (A&shade, A&sun, ~A&shade)
    alpha_pos=0.95,
    alpha_neg=0.55,
    show_night=True,
    sunrise_time_tuple=(6,30),
    sunset_time_tuple=(20,30),
    save=False,
    save_folderpath=''
):
    """
    Differential stack plot between a scene WITH a shelter and the same scene WITHOUT a shelter,
    for three categories at each probe and time:
      C1: Abricocoda & shaded
      C2: Abricocoda & sun
      C3: Not Abricocoda & shaded

    We plot Δ = WITH − WITHOUT as:
      - stacked positives (benefits) above 0
      - stacked negatives (losses) below 0
    """
    if any(s is None for s in (scene, scene_ref, tree)):
        raise ValueError("Provide scene, scene_ref, and tree.")
    if area_factor <= 0:
        raise ValueError("area_factor must be positive.")

    # --- Time axis (must match) ---
    t_with = scene.time_management_dict['target_date_list'].tz_localize(None)
    t_wo   = scene_ref.time_management_dict['target_date_list'].tz_localize(None)
    t_tree = tree.time_management_dict['target_date_list'].tz_localize(None)
    if not (t_with.equals(t_wo) and t_with.equals(t_tree)):
        raise ValueError("Time axes differ across inputs.")
    out_time = t_with

    # --- Helper: probe names (must match) ---
    def _probe_names(scene):
        keys = list(scene.output_dict['probe'].keys())
        return sorted({k[0] for k in keys if k[-1] == 'comfort_index'})
    names_with = _probe_names(scene)
    names_wo   = _probe_names(scene_ref)
    if not names_with:
        raise ValueError("No comfort_index probes in 'scene'.")
    if set(names_with) != set(names_wo):
        raise ValueError("Probe names must match between with/without scenes.")

    # --- Tree comfort series (first comfort_index found) ---
    tree_keys = list(tree.output_dict['probe'].keys())
    tree_key = next(k for k in tree_keys if k[-1] == 'comfort_index')
    tree_ser = pd.Series(
        np.asarray(tree.output_dict['probe'][tree_key][-1]).reshape(-1),
        index=out_time, name='tree'
    )

    # --- Helper: build category counts/areas for a scene ---
    def _categories_area(scene, probe_names):
        col_comfort, col_sun = {}, {}
        for name in probe_names:
            col_comfort[name] = np.asarray(scene.output_dict['probe'][(name, 'comfort_index')][-1]).reshape(-1)
            col_sun[name]     = np.asarray(scene.output_dict['probe'][(name, 'sun_exposure')][-1]).reshape(-1)

        comfort_df = pd.DataFrame(col_comfort, index=out_time)
        sun_df     = pd.DataFrame(col_sun, index=out_time)

        A = comfort_df.le(tree_ser, axis=0)   # Abricocoda
        S = (sun_df == 0)                     # shade

        C1 = (A & S)           # Abricocoda & shade
        C2 = (A & ~S)          # Abricocoda & sun
        C3 = (~A & S)          # Not Abricocoda & shade

        a1 = C1.sum(axis=1) * float(area_factor)
        a2 = C2.sum(axis=1) * float(area_factor)
        a3 = C3.sum(axis=1) * float(area_factor)
        return a1, a2, a3

    # Areas for each category
    A_with_C1, A_with_C2, A_with_C3       = _categories_area(scene, names_with)
    A_without_C1, A_without_C2, A_without_C3 = _categories_area(scene_ref, names_with)

    # Deltas per category (with − without)
    dC1 = A_with_C1 - A_without_C1    # Abricocoda & shade
    dC2 = A_with_C2 - A_without_C2    # Abricocoda & sun
    dC3 = A_with_C3 - A_without_C3    # Not Abricocoda & shade

    # Split into positive and negative stacks
    dC1_pos = dC1.clip(lower=0.0)
    dC2_pos = dC2.clip(lower=0.0)
    dC3_pos = dC3.clip(lower=0.0)

    dC1_neg = (-dC1.clip(upper=0.0))  # magnitudes
    dC2_neg = (-dC2.clip(upper=0.0))
    dC3_neg = (-dC3.clip(upper=0.0))

    # Totals for y-limit
    dCpos_sum = (dC1_pos + dC2_pos + dC3_pos)
    dCneg_sum = (dC1_neg + dC2_neg + dC3_neg)

    df = pd.DataFrame({
        'A_with_C1': A_with_C1, 'A_with_C2': A_with_C2, 'A_with_C3': A_with_C3,
        'A_without_C1': A_without_C1, 'A_without_C2': A_without_C2, 'A_without_C3': A_without_C3,
        'dC1': dC1, 'dC2': dC2, 'dC3': dC3,
        'dCpos_sum': dCpos_sum, 'dCneg_sum': dCneg_sum
    }, index=out_time)

    # --- Plot ---
    created_fig = False
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')

    c1, c2, c3 = colors

    # Positive stack (benefits above zero)
    # -> removed abricocoda & sun
    ax.stackplot(
        # out_time, dC1_pos, dC2_pos, dC3_pos,
        out_time, dC1_pos, dC3_pos,
        # colors=[c1, c2, c3],
        colors=[c1, c3],
        alpha=alpha_pos,
        labels=['Abricocoda & shade', 'Abricocoda & sun', 'no Abricocoda & shade'],
        zorder=3
    )

    # Negative stack (losses below zero) — pass negatives so they sit below baseline
    # -> removed abricocoda & sun
    ax.stackplot(
        # out_time, -dC1_neg, -dC2_neg, -dC3_neg,
        out_time, -dC1_neg, -dC3_neg,
        # colors=[c1, c2, c3],
        colors=[c1, c3],
        alpha=alpha_neg,
        # no labels to avoid duplicates; legend is from positive stack
        zorder=2
    )

    # Zero baseline
    ax.axhline(0, color='black', linewidth=1.2, zorder=4)

    # Optional night background
    if show_night:
        sunrise_time = sunrise_time_tuple[0] * 60 + sunrise_time_tuple[1]
        sunset_time = sunset_time_tuple[0] * 60 + sunset_time_tuple[1]
        time = out_time.hour * 60 + out_time.minute
        night_mask = (time < sunrise_time) | (time >= sunset_time)
        ax.fill_between(
            out_time, 0, 1, where=night_mask,
            color='gainsboro', alpha=0.8,
            transform=ax.get_xaxis_transform(), zorder=1
        )

    # Axes / formatting
    ax.set_ylabel('Δ area [m²]', fontsize=fontsize)
    ax.set_xlabel('Local time [h]', fontsize=fontsize)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlim([out_time[0], out_time[-1]])

    # Symmetric y-limits around 0
    y_top = max(
        float(pd.to_numeric(dCpos_sum, errors='coerce').max(skipna=True) or 0.0),
        float(pd.to_numeric(dCneg_sum, errors='coerce').max(skipna=True) or 0.0)
    )
    # if np.isfinite(y_top) and y_top > 0:
    #     ax.set_ylim(-1.05 * y_top, 1.05 * y_top)
    # else:
    #     ax.set_ylim(-1, 1)

    if title:
        ax.set_title(title, fontsize=fontsize)

    if ylim != 'auto':
            ax.set_ylim(ylim)

    if not remove_legend:
        ax.legend(loc='upper left', fontsize=fontsize-1, frameon=True)

    if save:
        save_path = Path(save_folderpath)
        save_path.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            plt.savefig(save_path / f'abricocoda_diff_stack_{scene.general_dict['case_name']}-{scene_ref.general_dict['case_name']}.{ext}', bbox_inches='tight')

    fig.canvas.manager.set_window_title('abricocoda_diff_stack ' + scene.general_dict['case_name']
                                        + ' - ' + scene_ref.general_dict['case_name'])



def plot_shelter_shade_quality(
    scene_list,
    scene_ref_list,
    tree,
    shelter_area_m2_list,          # <-- constant denominator (must be > 0)
    legend_label_list,
    figsize=(7, 5),
    fontsize=12,
    area_factor=1.0,          # m² represented by one probe
    linewidth=3.0,
    ylim = 'auto',
    show_night=True,
    sunrise_time_tuple=(6,30),
    sunset_time_tuple=(20,30),
    save=False,
    save_folderpath='',
    save_filename='Shade_quality'
):
    """
    Plot ratio(t) = Δ(Abricocoda & shade area) / S_shelter,
    where Δ = (with shelter) - (without shelter), and S_shelter is provided.

    Returns
    -------
    fig, ax, df
      df columns: ['AS_with_m2','AS_without_m2','dAS_m2','ratio']
    """
    # --- Plot ---
    # Get color palette with N distinct colors (one per shelter)
    num_scenes = len(scene_list)
    cmap = mpl.colormaps['Oranges']
    if num_scenes == 1:
        line_color_list = [cmap(0.5)]  # center of colormap
    else:
        # Avoid extremes: spread between 0.3 and 0.9
        color_positions = np.linspace(0.3, 0.9, num_scenes)
        line_color_list = [cmap(pos) for pos in color_positions]

    fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')

    for scene, scene_ref, shelter_area_m2, legend_label, line_color in zip(scene_list, scene_ref_list, shelter_area_m2_list, legend_label_list, line_color_list):

        # --- Checks ---
        if any(s is None for s in (scene, scene_ref, tree)):
            raise ValueError("Provide scene, scene_ref, and tree.")
        if area_factor <= 0:
            raise ValueError("area_factor must be positive.")
        if shelter_area_m2 is None or float(shelter_area_m2) <= 0:
            raise ValueError("shelter_area_m2 must be a positive number.")

        # --- Time axis (must match) ---
        t_with = scene.time_management_dict['target_date_list'].tz_localize(None)
        t_wo   = scene_ref.time_management_dict['target_date_list'].tz_localize(None)
        t_tree = tree.time_management_dict['target_date_list'].tz_localize(None)
        if not (t_with.equals(t_wo) and t_with.equals(t_tree)):
            raise ValueError("Time axes differ across inputs.")
        out_time = t_with

        # --- Probe names (must match) ---
        keys_with = list(scene.output_dict['probe'].keys())
        keys_wo   = list(scene_ref.output_dict['probe'].keys())
        probe_names_with = sorted({k[0] for k in keys_with if k[-1] == 'comfort_index'})
        probe_names_wo   = sorted({k[0] for k in keys_wo   if k[-1] == 'comfort_index'})
        if not probe_names_with:
            raise ValueError("No comfort_index probes in 'scene'.")
        if set(probe_names_with) != set(probe_names_wo):
            raise ValueError("Probe names must match between with/without scenes.")

        # --- Tree comfort series (first comfort_index found) ---
        tree_keys = list(tree.output_dict['probe'].keys())
        tree_key = next(k for k in tree_keys if k[-1] == 'comfort_index')
        tree_ser = pd.Series(
            np.asarray(tree.output_dict['probe'][tree_key][-1]).reshape(-1),
            index=out_time, name='tree'
        )

        # --- Scene WITH: build comfort & sun DataFrames ---
        col_c_with, col_s_with = {}, {}
        for name in probe_names_with:
            col_c_with[name] = np.asarray(scene.output_dict['probe'][(name, 'comfort_index')][-1]).reshape(-1)
            col_s_with[name] = np.asarray(scene.output_dict['probe'][(name, 'sun_exposure')][-1]).reshape(-1)
        comfort_with = pd.DataFrame(col_c_with, index=out_time)
        sun_with     = pd.DataFrame(col_s_with, index=out_time)

        # --- Scene WITHOUT: build comfort & sun DataFrames ---
        col_c_wo, col_s_wo = {}, {}
        for name in probe_names_with:  # same names checked above
            col_c_wo[name] = np.asarray(scene_ref.output_dict['probe'][(name, 'comfort_index')][-1]).reshape(-1)
            col_s_wo[name] = np.asarray(scene_ref.output_dict['probe'][(name, 'sun_exposure')][-1]).reshape(-1)
        comfort_wo = pd.DataFrame(col_c_wo, index=out_time)
        sun_wo     = pd.DataFrame(col_s_wo, index=out_time)

        # --- Masks ---
        A_with = comfort_with.le(tree_ser, axis=0)   # Abricocoda
        S_with = (sun_with == 0)                     # shade
        AS_with = (A_with & S_with)

        A_wo = comfort_wo.le(tree_ser, axis=0)
        S_wo = (sun_wo == 0)
        AS_wo = (A_wo & S_wo)

        # --- Areas (m²) ---
        AS_with_m2    = AS_with.sum(axis=1) * float(area_factor)
        AS_without_m2 = AS_wo.sum(axis=1)   * float(area_factor)
        dAS_m2        = AS_with_m2 - AS_without_m2

        # --- Ratio over time (constant denominator) ---
        S_shelter = float(shelter_area_m2)
        ratio = dAS_m2 / S_shelter * 100

        df = pd.DataFrame({
            'AS_with_m2': AS_with_m2,
            'AS_without_m2': AS_without_m2,
            'dAS_m2': dAS_m2,
            'ratio': ratio
        }, index=out_time)

        ax.plot(df.index, df['ratio'], label=legend_label, color=line_color, linewidth=linewidth)

    # Optional night shading
    if show_night:
        sunrise_time = sunrise_time_tuple[0] * 60 + sunrise_time_tuple[1]
        sunset_time = sunset_time_tuple[0] * 60 + sunset_time_tuple[1]
        time = out_time.hour * 60 + out_time.minute
        night_mask = (time < sunrise_time) | (time >= sunset_time)
        ax.fill_between(
            out_time, 0, 1,
            where=night_mask, color='gainsboro', alpha=0.8,
            transform=ax.get_xaxis_transform()
        )

    # Axes & formatting
    ax.set_ylabel(r'Shelter shade quality [%]', fontsize=fontsize)
    ax.set_xlabel('Local time [h]', fontsize=fontsize)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlim([out_time[0], out_time[-1]])
    ax.legend(fontsize=fontsize - 5)

    if ylim != 'auto':
        ax.set_ylim(ylim)

    # save if needed
    if save:
        save_path = Path(save_folderpath)
        save_path.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            plt.savefig(save_path / f'{save_filename}.{ext}', bbox_inches='tight')

    fig.canvas.manager.set_window_title(save_filename)


def plot_comfort_distribution(scene_list,
                              tree,
                              legend_label_list,
                              sel_out_time_list,
                              slice_dir='x',
                              slice_coord_list = [0.],
                              invert_xaxis=False,
                              xlim='auto',
                              ylim='auto',
                              figsize=(10, 7),
                              markersize=3,
                              linewidth=2,
                              fontsize=12,
                              sunrise_time_tuple=(6,30),
                              sunset_time_tuple=(20,30),
                              save=False,
                              save_folderpath='',
                              save_filename='Comfort_distribution'):

        # check if tree object is specified
        if tree is None:
                raise ValueError('Tree must be specified to plot the abricocoda area.')

        # ensure all scenes and tree share the same time index
        out_time_list = [
                scene.time_management_dict['target_date_list'].tz_localize(None)
                for scene in scene_list
        ] + [tree.time_management_dict['target_date_list'].tz_localize(None)]
        reference_time = out_time_list[0]
        if not all(t.equals(reference_time) for t in out_time_list):
                raise ValueError("Not all DateTimeIndex entries are identical across scenes and tree.")
        out_time = reference_time

        # gather tree confort_index data in a dataframe
        tree_probe_list = list({probe for probe, _ in tree.output_dict['probe'].values()})
        tree_value_list = [
                tree.output_dict['probe'][(probe.name, 'comfort_index')][-1]
                for probe in tree_probe_list
        ]
        tree_df = pd.DataFrame(index=out_time, data=np.array(tree_value_list).T)

        dir_key = {'x': 0, 'y': 1, 'z': 2}.get(slice_dir, 1)
        other_dir = 'y' if slice_dir == 'x' else 'x'
        other_dir_key = {'x': 0, 'y': 1, 'z': 2}.get(other_dir, 1)

        scene_df_list = []
        for idx, scene in enumerate(scene_list):
                # scene comfort index in dataframes
                scene_probe_list = list({probe for probe, _ in scene.output_dict['probe'].values()})
                out_val_list = []
                col_name_list = []
                for probe in scene_probe_list:
                        if np.round(probe.coord[other_dir_key], 2) == slice_coord_list[idx]:
                                out_val_list.append(scene.output_dict['probe'][(probe.name, 'comfort_index')][-1])
                                col_name_list.append(tuple(probe.coord))
                scene_df = pd.DataFrame(index=out_time, data=np.array(out_val_list).T, columns=col_name_list)
                surface = next((s for s in scene.surface_list_dict['all'] if s.type == 'shelter'), None)
                scene_df.columns.name = (surface, scene.general_dict['case_name'])
                scene_df_list.append(scene_df)

        # plotting
        # Get color palette with N distinct colors (one per shelter)
        num_scenes = len(scene_list)
        cmap = mpl.colormaps['Oranges']

        if num_scenes == 1:
                marker_color_list = [cmap(0.5)]  # center of colormap
        else:
                # Avoid extremes: spread between 0.3 and 0.9
                color_positions = np.linspace(0.3, 0.9, num_scenes)
                marker_color_list = [cmap(pos) for pos in color_positions]

        # sel_out_time = out_time[(out_time.hour > 5) & (out_time.hour % 2 == 0) & (out_time.minute == 0)]
        conds = [(out_time.hour == h) & (out_time.minute == m) for h, m in sel_out_time_list]
        sel_out_time = out_time[np.logical_or.reduce(conds)]

        # rows = int(np.ceil(np.sqrt(len(sel_out_time))))
        # cols = int(np.ceil(len(sel_out_time) / rows))
        rows = 1
        cols = len(sel_out_time)
        fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True, layout='constrained')

        for ax in axs.ravel()[len(sel_out_time):]:
                ax.set_visible(False)

        for i, date in enumerate(sel_out_time):
            row, col = divmod(i, cols)
            if rows > 1 and cols > 1:
                    ax = axs[row, col]
            else:
                if rows == 1:
                    ax = axs[col]
                else:
                    ax = axs[row]

            # shade night
            sunrise_time = sunrise_time_tuple[0] * 60 + sunrise_time_tuple[1]
            sunset_time = sunset_time_tuple[0] * 60 + sunset_time_tuple[1]
            date_time = date.hour * 60 + date.minute
            if (date_time < sunrise_time) | (date_time >= sunset_time):
                    ax.set_facecolor('gainsboro')

            # plot each scene
            for j, scene_df in enumerate(scene_df_list):
                    surface = scene_df.columns.name[0]
                    x_vals = [coord[dir_key] for coord in scene_df.columns]
                    y_vals = scene_df.loc[date].values.flatten()
                    label = legend_label_list[j]
                    d_min, d_max = surface.mesh.points[:, dir_key].min(), surface.mesh.points[:, dir_key].max()
                    # color_list = ['sandybrown', 'sandybrown', 'sandybrown', 'sandybrown'] # 'per_material'
                    # color_list = ['sandybrown', 'indianred', 'goldenrod', 'yellowgreen'] # 'per_size'
                    color_list = ['gainsboro', 'gainsboro', 'gainsboro', 'gainsboro']

                    ax.plot(x_vals, y_vals, 'o', markersize=markersize, label=label, color=marker_color_list[j])
                    ax.axvspan(d_min, d_max, color=color_list[j % len(color_list)], alpha=0.4)

            # plot tree reference
            ax.axhline(tree_df.loc[date][0], color='g', ls='-', linewidth=linewidth, label=r'$PET_{CTC}$')

            # configure axis
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=fontsize)
            # Only label outer axes
            if row != rows - 1:
                    ax.set_xlabel('')
            else:
                    ax.set_xlabel(f'${slice_dir}~[m]$', fontsize=fontsize)

            if col != 0:
                    ax.set_ylabel('')
            else:
                    ax.set_ylabel('PET [°C]', fontsize=fontsize)
            ax.set_title(date.strftime("%I %p"), fontsize=fontsize, fontweight='bold')

            if xlim != 'auto':
                    ax.set_xlim(xmin=xlim[0], xmax=xlim[1])
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            # ax.tick_params(axis='x', labelrotation=30)
            if invert_xaxis:
                    ax.invert_xaxis()
            if ylim != 'auto':
                    ax.set_ylim(ylim)
            if i == len(sel_out_time)-1:
                    ax.legend(fontsize=fontsize-2)

        # save if needed
        if save:
                save_path = Path(save_folderpath)
                save_path.mkdir(parents=True, exist_ok=True)
                for ext in ['png', 'pdf']:
                        plt.savefig(save_path / f'{save_filename}.{ext}', bbox_inches='tight')

        fig.canvas.manager.set_window_title(save_filename)


def plot_shade_performance(exposed_scene,
                           tunnel_scene,
                           scene_list,
                           legend_label_list,
                           linecolor_list,
                           dashes_list,
                           probe_name,
                           xlim='auto',
                           ylim='auto',
                           remove_legend=False,
                           figsize=(6, 5),
                           fontsize=12,
                           linewidth=2,
                           sunrise_time_tuple=(6,30),
                           sunset_time_tuple=(20,30),
                           save=False,
                           save_folderpath=''):
    all_scene_list = [exposed_scene, tunnel_scene] + scene_list
    all_legend_label_list = ['exposed', 'tunnel'] + legend_label_list
    # validate probe and variable names
    for scene in all_scene_list:
        available_probes = {probe.name for probe, _ in scene.output_dict['probe'].values()}
        available_vars = {var for _, var in scene.output_dict['probe'].keys()}

        if probe_name not in available_probes:
            raise ValueError(f"Object 'probe' not found in scene. Available: {available_probes} for scene {scene.general_dict['case_name']}")

        if 'tmrt' not in available_vars:
            raise ValueError(f"Variable 'tmrt' not found in scene. Available: {available_vars} for scene {scene.general_dict['case_name']}")

    # ensure all scenes share the same time index
    out_time_list = [
        scene.time_management_dict['target_date_list'].tz_localize(None)
        for scene in all_scene_list
    ]
    reference_time = out_time_list[0]
    if not all(t.equals(reference_time) for t in out_time_list):
        raise ValueError("Not all DateTimeIndex entries are identical across scenes.")
    out_time = reference_time

    # build DataFrame from scenes
    out_val_list = []
    col_name_list = []
    ref_val = exposed_scene.output_dict['probe'][(probe_name, ('tmrt'))][-1]
    for i, scene in enumerate(all_scene_list):
        out_val = np.array(scene.output_dict['probe'][(probe_name, ('tmrt'))][-1])
        out_val_list.append(out_val - ref_val)
        col_name_list.append(all_legend_label_list[i])
    df_glob = pd.DataFrame(index=out_time, data=np.array(out_val_list).T, columns=col_name_list)

    # plotting
    all_linecolor_list = ['y', 'k'] + linecolor_list
    all_dashes_list = ['', ''] + dashes_list
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    sns.lineplot(data=df_glob, ax=ax, palette=all_linecolor_list, linewidth=linewidth, dashes=all_dashes_list)

    # add night background shading
    sunrise_time = sunrise_time_tuple[0] * 60 + sunrise_time_tuple[1]
    sunset_time = sunset_time_tuple[0] * 60 + sunset_time_tuple[1]
    time = out_time.hour * 60 + out_time.minute
    night_mask = (time < sunrise_time) | (time >= sunset_time)
    ax.fill_between(
            out_time, 0, 1,
            where=night_mask,
            color='gainsboro',
            alpha=0.8,
            transform=ax.get_xaxis_transform()
    )

    # configure axis
    ax.grid(True)
    if remove_legend and ax.get_legend():
        ax.get_legend().remove()
    # else:
    #     ax.legend(fontsize=fontsize - 2, frameon=True)


    ax.set_xlabel('Local time [h]', fontsize=fontsize)
    ax.set_ylabel(r'$\Delta T_{mrt}~[°C]$', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    if xlim != 'auto':
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([out_time[0], out_time[-1]])
    # ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
    if ylim != 'auto':
        ax.set_ylim(ylim)

    # save if needed
    if save:
        save_path = Path(save_folderpath)
        save_path.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            plt.savefig(save_path / f'shade_performance_curves_{probe_name}_{scene.general_dict['case_name']}.{ext}', bbox_inches='tight')

    fig.canvas.manager.set_window_title(f'shade_performance_curves_{probe_name}_{scene.general_dict['case_name']}')