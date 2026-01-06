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

import numpy as np
import pyvista as pv
import pandas as pd
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (
    QAction, QComboBox, QLabel, QCheckBox, QVBoxLayout,
    QDockWidget, QWidget, QToolBar
)
from PyQt5.QtCore import QTimer, Qt


def visualize_fields(scene, tree=None):
    """
    Visualize time-varying scalar fields on PyVista surface meshes + probes.
    """

    # ---- Constants for scalar bar management ----
    SURFACE_BAR_TITLE = "__surface_field__"
    PROBE_BAR_TITLE = "__probe_field__"

    SURFACE_BAR_POS = dict(position_x=0.5, position_y=0.9, width=0.18, height=0.08)
    PROBE_BAR_POS = dict(position_x=0.72, position_y=0.9, width=0.18, height=0.08)

    # ---------------------------
    # --------- INPUTS ----------
    # ---------------------------
    surface_fields = scene.output_dict["surface_field"]
    timestep_seconds = float(scene.time_management_dict["dt"])

    # Build per-variable surface data
    field_data = {}
    surface_names = set()

    surface_fields_key_list = list(surface_fields.keys())
    surface_name_set = set([surf for surf, var in surface_fields_key_list])

    for (surface_name, variable), (surface, values_over_time) in surface_fields.items():
        if "_cover_back" in surface_name:
            continue
        if len(surface_name_set) > 20:
            if "_back" in surface_name:
                continue
        surface_names.add(surface_name)
        field_data.setdefault(variable, {})[surface_name] = {
            "surface": surface,
            "values": np.stack(values_over_time)  # (T, N)
        }

    field_data = {var: d for var, d in field_data.items() if d}
    if not field_data:
        raise ValueError("No surfaces to plot after filtering names containing '_cover_back' or '_back''.")

    variable_list = sorted(field_data.keys())

    # Synthetic surface-only option (solid white geometry)
    SURFACE_ONLY = "surface"
    variable_choices = [SURFACE_ONLY] + variable_list

    # Current state
    current_variable = variable_choices[0]
    current_step = 0
    is_playing = False
    timer = QTimer()

    # Bookkeeping
    surface_meshes = {var: {} for var in variable_list}
    surface_actors = {}
    surface_checkboxes = {}
    text_actor = None

    # Global min/max per variable for surfaces
    field_clims = {
        var: [
            float(min(np.min(d["values"]) for d in field_data[var].values())),
            float(max(np.max(d["values"]) for d in field_data[var].values())),
        ]
        for var in variable_list
    }

    # Map meshes
    for variable in variable_list:
        for surface_name, data in field_data[variable].items():
            surface_meshes[variable][surface_name] = data["surface"].mesh

    # Any variable will do to get n_steps
    sample_surface = next(iter(field_data[variable_list[0]].values()))
    n_steps = int(sample_surface["values"].shape[0])

    # Colormaps for surfaces
    cmap_dict = {
        "temperature": "coolwarm",
        "sw_rad_flux": "inferno",
        "lw_rad_flux": "plasma",
        "conv_flux": "cividis",
        "cond_flux": "viridis",
        "lw_radiosity": "magma",
        "sw_radiosity": "YlOrRd",
        "sun_exposure": ["k", "w"],
    }

    # ---------------------------
    # --------- PLOTTING --------
    # ---------------------------
    plotter = BackgroundPlotter()

    # === LEFT DOCK PANEL ===
    left_layout = QVBoxLayout()

    # Surface visibility toggles
    left_layout.addWidget(QLabel("Surfaces:"))
    for sname in sorted(surface_names):
        cb = QCheckBox(sname)
        cb.setChecked(True)
        surface_checkboxes[sname] = cb
        left_layout.addWidget(cb)

    # Probes toggle (always present; may just do nothing if no probes)
    left_layout.addWidget(QLabel("Comfort Probes:"))
    comfort_checkbox = QCheckBox("Show")
    comfort_checkbox.setChecked(True)
    left_layout.addWidget(comfort_checkbox)

    # Surface field selector
    left_layout.addWidget(QLabel("Field:"))
    combo_box = QComboBox()
    combo_box.addItems(variable_choices)
    left_layout.addWidget(combo_box)

    # Surface color scale mode
    left_layout.addWidget(QLabel("Color Scale:"))
    color_mode_box = QComboBox()
    color_mode_box.addItems(["Global", "Current"])
    color_mode_box.setCurrentText("Global")
    left_layout.addWidget(color_mode_box)

    # Probe field selector
    left_layout.addWidget(QLabel("Probe Field:"))
    probe_field_box = QComboBox()
    left_layout.addWidget(probe_field_box)

    # Probe color scale mode (for continuous probe fields)
    left_layout.addWidget(QLabel("Probe Color Scale:"))
    probe_color_mode_box = QComboBox()
    probe_color_mode_box.addItems(["Global", "Current"])
    probe_color_mode_box.setCurrentText("Global")
    left_layout.addWidget(probe_color_mode_box)

    left_layout.addStretch()

    # Add left dock
    left_widget = QWidget()
    left_widget.setLayout(left_layout)
    left_dock = QDockWidget("Controls", plotter.app_window)
    left_dock.setWidget(left_widget)
    left_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
    plotter.app_window.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

    # ------ PROBES (MERGED) ----
    # ---------------------------
    probe_actor = None
    merged = None
    probe_names = []
    sphere_npts = None

    reference_comfort_df = None
    probe_data = {}  # dict[var_name][probe_name] -> {position, values(df)}
    probe_global_minmax = {}  # dict[var_name] -> (vmin, vmax)
    probe_field_options = []
    comfort_vs_tree_name = "comfort_vs_tree"

    orientations = ["top", "bottom", "north", "south", "east", "west"]

    # Time index for probes (from scene if available)
    out_time_list_scene = scene.time_management_dict.get("target_date_list")
    if out_time_list_scene is not None:
        out_time_list_scene = out_time_list_scene.tz_localize(None)

    probe_time_index = None

    # Base scalar probe variables (1D)
    base_probe_vars = [
        "comfort_index",
        "tmrt",
        "air_temperature",
        "sun_exposure",
    ]

    # Collect scene probes
    for key, val in scene.output_dict.get("probe", {}).items():
        probe_name = key[0]
        var_name = key[-1]
        position = val[0].coord
        raw = val[-1]

        # 1D variables
        if var_name in base_probe_vars:
            arr = np.asarray(raw).reshape(-1, 1)
            n_steps_probe = arr.shape[0]

            if probe_time_index is None:
                if out_time_list_scene is not None and len(out_time_list_scene) == n_steps_probe:
                    probe_time_index = out_time_list_scene
                else:
                    probe_time_index = pd.RangeIndex(n_steps_probe)

            df = pd.DataFrame(index=probe_time_index, data=arr)
            var_dict = probe_data.setdefault(var_name, {})
            var_dict[probe_name] = {
                "position": position,
                "values"  : df,
            }

        # 2D flux arrays -> split into 6 oriented variables
        elif var_name in ("sw_flux_arr", "lw_flux_arr"):
            arr = np.asarray(raw)  # expected shape (n_steps, 6)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            n_steps_probe = arr.shape[0]

            if arr.shape[1] != 6:
                # unexpected shape, skip to avoid inconsistent dimensions
                continue

            if probe_time_index is None:
                if out_time_list_scene is not None and len(out_time_list_scene) == n_steps_probe:
                    probe_time_index = out_time_list_scene
                else:
                    probe_time_index = pd.RangeIndex(n_steps_probe)

            base_name = "sw_flux" if var_name == "sw_flux_arr" else "lw_flux"

            for j, ori in enumerate(orientations):
                oriented_name = base_name + "_" + ori  # e.g. sw_flux_top
                col = arr[:, j].reshape(-1, 1)
                df = pd.DataFrame(index=probe_time_index, data=col)

                var_dict = probe_data.setdefault(oriented_name, {})
                var_dict[probe_name] = {
                    "position": position,
                    "values"  : df,
                }

    # Reference comfort index from tree, if provided
    if tree is not None:
        out_time_list_tree = tree.time_management_dict["target_date_list"].tz_localize(None)
        for key, val in tree.output_dict.get("probe", {}).items():
            if key[-1] == "comfort_index":
                reference_comfort_df = pd.DataFrame(index=out_time_list_tree, data=val[-1])
                break

    # Build probe field options and global min/max for continuous fields
    for name, var_dict in probe_data.items():
        probe_field_options.append(name)

        # categorical / boolean fields skip min/max
        if name in ("sun_exposure",) or name == comfort_vs_tree_name:
            continue

        mins = []
        maxs = []
        for pdata in var_dict.values():
            arr = np.asarray(pdata["values"])
            if arr.size:
                mins.append(np.nanmin(arr))
                maxs.append(np.nanmax(arr))
        if mins and maxs:
            probe_global_minmax[name] = (
                float(np.min(mins)),
                float(np.max(maxs)),
            )

    # Extra comparison field when tree is provided
    if tree is not None and "comfort_index" in probe_data and "sun_exposure" in probe_data and reference_comfort_df is not None:
        probe_field_options.append(comfort_vs_tree_name)

    # Fill probe field combo box
    if probe_field_options:
        probe_field_box.addItems(sorted(probe_field_options))
        if "comfort_index" in probe_field_options:
            probe_field_box.setCurrentText("comfort_index")

    # Build merged spheres if we have at least one probe variable
    if probe_data:
        probe_positions = {}
        for var_dict in probe_data.values():
            for pname, pdata in var_dict.items():
                probe_positions.setdefault(pname, pdata["position"])

        for pname in sorted(probe_positions):
            sphere = pv.Sphere(radius=0.3, center=probe_positions[pname])
            if sphere_npts is None:
                sphere_npts = sphere.n_points
            if merged is None:
                merged = sphere
            else:
                merged = merged.append_polydata(sphere)
            probe_names.append(pname)

        if merged is not None:
            merged.point_data["probe_scalar"] = np.zeros(merged.n_points)
            probe_actor = plotter.add_mesh(
                    merged,
                    scalars="probe_scalar",
                    cmap="viridis",
                    show_scalar_bar=False,
            )

    # ---------------------------
    # --------- HELPERS ---------
    # ---------------------------
    def _set_surface_scalar_bar(title_text, show):
        try:
            plotter.remove_scalar_bar(SURFACE_BAR_TITLE)
        except Exception:
            pass
        if show:
            sb = plotter.add_scalar_bar(
                title=SURFACE_BAR_TITLE,
                font_family="times",
                label_font_size=24,
                title_font_size=24,
                **SURFACE_BAR_POS
            )
            try:
                sb.SetTitle(title_text)
            except Exception:
                pass

    def _remove_probe_scalar_bar():
        try:
            plotter.remove_scalar_bar(PROBE_BAR_TITLE)
        except Exception:
            pass

    def _probe_unit_for(var):
        if var in ("temperature", "tmrt", "air_temperature"):
            return " [°C]"
        if var.startswith("sw_flux") or var.startswith("lw_flux") \
           or var in ("sw_rad_flux", "lw_rad_flux", "conv_flux", "cond_flux"):
            return " [W/m²]"
        if var in ("sun_exposure", "comfort_index"):
            return " []"
        return ""

    def _add_probe_scalar_bar(field_name):
        if probe_actor is None:
            return
        _remove_probe_scalar_bar()
        sb = plotter.add_scalar_bar(
            title=PROBE_BAR_TITLE,
            mapper=probe_actor.mapper,
            font_family="times",
            label_font_size=24,
            title_font_size=24,
            **PROBE_BAR_POS
        )
        sb.SetTitle(field_name + _probe_unit_for(field_name))

    def _add_probe_legend_for_comfort():
        _remove_probe_scalar_bar()
        if hasattr(plotter, "_comfort_legend"):
            try:
                plotter.remove_legend()
            except Exception:
                pass
        labels = [
            ["Abricocoda & shade", "forestgreen"],
            ["Abricocoda & sun", "royalblue"],
            ["no Abricocoda & shade", "orange"],
        ]
        plotter._comfort_legend = plotter.add_legend(
            labels,
            bcolor="w",
            face="circle",
            size=(0.2, 0.15),
            loc="lower right",
        )

    def _remove_probe_legend():
        if hasattr(plotter, "_comfort_legend"):
            try:
                plotter.remove_legend()
            except Exception:
                pass
            del plotter._comfort_legend

    def _probe_cmap(field_name):
        if field_name == "tmrt":
            return "YlOrRd"
        if field_name == 'air_temperature':
            return "coolwarm"
        if field_name == "sun_exposure":
            return ["black", "white"]
        if "lw_flux" in field_name:
            return "plasma"
        if "sw_flux" in field_name:
            return "inferno"
        return "viridis"


    def _unit_for(var):
        return " [°C]" if var == "temperature" else " []" if var == "sun_exposure" else " [W/m²]"

    def _visible_surfaces():
        return [s for s, cb in surface_checkboxes.items() if cb.isChecked()]

    def update_comfort_probes(step):
        if probe_actor is None or merged is None or sphere_npts is None:
            return

        if comfort_checkbox is not None and not comfort_checkbox.isChecked():
            probe_actor.SetVisibility(False)
            _remove_probe_scalar_bar()
            _remove_probe_legend()
            return

        if probe_field_box is None or probe_field_box.count() == 0:
            return

        probe_actor.SetVisibility(True)
        sel = probe_field_box.currentText()

        # ---- comparison field (categorical, only if tree is given) ----
        if sel == comfort_vs_tree_name:
            if tree is None or reference_comfort_df is None:
                return
            if "comfort_index" not in probe_data or "sun_exposure" not in probe_data:
                return

            _remove_probe_scalar_bar()
            global_arr = np.zeros(merged.n_points)
            offset = 0

            for pname in probe_names:
                ci_dict = probe_data["comfort_index"].get(pname)
                sun_dict = probe_data["sun_exposure"].get(pname)
                if ci_dict is None or sun_dict is None:
                    color_val = 1.0
                else:
                    ci_arr = ci_dict["values"]
                    sun_arr = sun_dict["values"]
                    if step >= len(ci_arr):
                        color_val = 1.0
                    else:
                        probe_ci = float(ci_arr.iloc[step, 0])
                        probe_sun = float(sun_arr.iloc[step, 0])
                        if step < len(reference_comfort_df):
                            tree_ci = float(reference_comfort_df.iloc[step, 0])
                        else:
                            tree_ci = probe_ci
                        if probe_ci > tree_ci:
                            color_val = 0.0 if probe_sun == 1.0 else 0.35
                        else:
                            color_val = 0.6 if probe_sun == 1.0 else 1.0

                global_arr[offset:offset + sphere_npts] = color_val
                offset += sphere_npts

            merged.point_data["probe_scalar"] = global_arr
            merged.set_active_scalars("probe_scalar")
            try:
                probe_actor.mapper.lookup_table = pv.LookupTable(
                    ["white", "gold", "dodgerblue", "lime"]
                )
            except Exception:
                pass
            probe_actor.mapper.ScalarVisibilityOn()
            probe_actor.mapper.SetScalarRange(0.0, 1.0)
            probe_actor.mapper.Modified()
            _add_probe_legend_for_comfort()
            return

        # ---- continuous fields (including raw comfort_index and others) ----
        _remove_probe_legend()

        mode = probe_color_mode_box.currentText() if probe_color_mode_box else "Global"
        if mode == "Current":
            vals = []
            for pname in probe_names:
                var_dict = probe_data.get(sel, {})
                pdata = var_dict.get(pname)
                if pdata is not None:
                    df = pdata["values"]
                    if step < len(df):
                        v = float(df.iloc[step, 0])
                        if not np.isnan(v):
                            vals.append(v)
            if vals:
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
            else:
                vmin, vmax = probe_global_minmax.get(sel, (0.0, 1.0))
        else:
            vmin, vmax = probe_global_minmax.get(sel, (0.0, 1.0))

        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5

        global_arr = np.zeros(merged.n_points)
        offset = 0
        for pname in probe_names:
            var_dict = probe_data.get(sel, {})
            pdata = var_dict.get(pname)
            if pdata is not None:
                df = pdata["values"]
                val = float(df.iloc[step, 0]) if step < len(df) else np.nan
            else:
                val = np.nan
            if np.isnan(val):
                val = vmin
            global_arr[offset:offset + sphere_npts] = val
            offset += sphere_npts

        merged.point_data["probe_scalar"] = global_arr
        merged.set_active_scalars("probe_scalar")
        try:
            probe_actor.mapper.lookup_table = pv.LookupTable(_probe_cmap(sel))
        except Exception:
            pass
        probe_actor.mapper.ScalarVisibilityOn()
        probe_actor.mapper.SetScalarRange(vmin, vmax)
        probe_actor.mapper.Modified()
        _add_probe_scalar_bar(sel)

    def update_text():
        nonlocal text_actor
        total_seconds = int(current_step * timestep_seconds)
        h, r = divmod(total_seconds, 3600)
        m = r // 60
        if text_actor:
            plotter.remove_actor(text_actor)
        text_actor = plotter.add_text(
            f"{h:02d}:{m:02d}",
            position="upper_left",
            font_size=14,
        )

    def update_scalar_field(variable, step):
        """
        Update surfaces for given variable and time step, including scalar bar.
        """
        # -- geometry only (white) --
        if variable == SURFACE_ONLY:
            for sname in surface_names:
                example_var = variable_list[0]
                mesh = surface_meshes[example_var].get(sname)
                if mesh is None:
                    continue
                actor = surface_actors.get(sname)
                if actor is None:
                    actor = plotter.add_mesh(
                        mesh,
                        color="white",
                        show_edges=True,
                        show_scalar_bar=False,
                    )
                    surface_actors[sname] = actor
                actor.mapper.ScalarVisibilityOff()
                actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                actor.SetVisibility(surface_checkboxes[sname].isChecked())

            # Hide surface scalar bar
            _set_surface_scalar_bar("", False)

            # Update text and probes
            update_text()
            update_comfort_probes(step)
            plotter.update()
            return

        # -- scalar fields on surfaces --
        # Compute color range
        if color_mode_box.currentText() == "Current":
            all_vals = [field_data[variable][s]["values"][step] for s in _visible_surfaces()]
            if all_vals:
                vmin = float(np.min(np.concatenate(all_vals)))
                vmax = float(np.max(np.concatenate(all_vals)))
            else:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = field_clims[variable]

        # Update actors
        for sname, mesh in surface_meshes[variable].items():
            values = field_data[variable][sname]["values"][step]
            mesh[variable] = values
            actor = surface_actors.get(sname)
            if actor is None:
                actor = plotter.add_mesh(
                    mesh,
                    scalars=variable,
                    show_edges=True,
                    cmap=cmap_dict.get(variable, "viridis"),
                    show_scalar_bar=False,
                )
                surface_actors[sname] = actor
            actor.SetVisibility(surface_checkboxes[sname].isChecked())
            actor.mapper.ScalarVisibilityOn()
            actor.mapper.scalar_range = [vmin, vmax]

        # Dummy anchor for consistent scalar bar even if meshes hidden
        if "__dummy__" not in surface_actors:
            dummy_mesh = pv.Sphere(radius=0.01, center=(0, 0, 0))
            dummy_mesh["dummy_scalar"] = np.array([0.0] * dummy_mesh.n_cells)
            dummy_actor = plotter.add_mesh(
                dummy_mesh,
                scalars="dummy_scalar",
                cmap=cmap_dict.get(variable, "viridis"),
                show_scalar_bar=False,
            )
            surface_actors["__dummy__"] = dummy_actor

        anchor = surface_actors["__dummy__"]
        anchor.mapper.ScalarVisibilityOn()
        anchor.mapper.scalar_range = [vmin, vmax]
        anchor.SetVisibility(True)

        # Surface scalar bar (managed by title key; placed top-right)
        _set_surface_scalar_bar(variable + _unit_for(variable), True)

        update_text()
        update_comfort_probes(step)
        plotter.update()

    # ---------------------------
    # --------- CALLBACKS -------
    # ---------------------------
    def update_time_step(value):
        nonlocal current_step
        current_step = int(value)
        update_scalar_field(current_variable, current_step)

    def update_frame():
        nonlocal current_step
        if current_step >= n_steps - 1:
            update_scalar_field(current_variable, current_step)
            stop_animation()
            return
        update_scalar_field(current_variable, current_step)
        current_step += 1

    def toggle_animation():
        nonlocal is_playing, current_step
        if is_playing:
            stop_animation()
        else:
            if current_step >= n_steps:
                current_step = 0
            timer.start(100)
            play_action.setText("⏸ Pause")
            is_playing = True

    def stop_animation():
        nonlocal is_playing
        timer.stop()
        play_action.setText("▶ Play")
        is_playing = False

    def on_variable_change():
        nonlocal current_variable, surface_actors
        was_playing = is_playing
        stop_animation()

        current_variable = combo_box.currentText()

        # Clear old actors (including dummy)
        for actor in list(surface_actors.values()):
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        surface_actors.clear()

        # Ensure camera widget added once
        if not hasattr(plotter, "_camera_widget_added"):
            plotter.add_camera_orientation_widget()
            plotter._camera_widget_added = True

        update_scalar_field(current_variable, current_step)
        if was_playing:
            toggle_animation()

    def on_color_mode_change():
        update_scalar_field(current_variable, current_step)

    def on_probe_field_change():
        update_comfort_probes(current_step)
        plotter.update()

    def on_probe_color_mode_change():
        update_comfort_probes(current_step)
        plotter.update()

    def flip_camera():
        cam = plotter.camera
        pos = np.array(cam.GetPosition())
        focal = np.array(cam.GetFocalPoint())
        view_up = np.array(cam.GetViewUp())
        direction = pos - focal
        angle_rad = np.deg2rad(90)
        Rz = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ])
        new_dir = Rz @ direction
        new_up = Rz @ view_up
        cam.SetPosition(*(focal + new_dir))
        cam.SetViewUp(*new_up)
        plotter.render()

    # ---------------------------
    # --------- WIDGETS ---------
    # ---------------------------
    slider = plotter.add_slider_widget(
        callback=update_time_step,
        rng=[0, n_steps - 1],
        value=0,
        pointa=(0.1, 0.92),
        pointb=(0.45, 0.92),
    )

    def add_action(toolbar, label, method, main_window):
        action = QAction(label, main_window)
        action.triggered.connect(method)
        toolbar.addAction(action)
        return action

    user_toolbar = plotter.app_window.addToolBar("Controls")
    play_action = add_action(user_toolbar, "▶ Play", toggle_animation, plotter.app_window)
    flip_view_action = add_action(user_toolbar, "🔄 Flip View 90°", flip_camera, plotter.app_window)

    combo_box.currentIndexChanged.connect(on_variable_change)
    color_mode_box.currentIndexChanged.connect(on_color_mode_change)
    timer.timeout.connect(update_frame)
    probe_field_box.currentIndexChanged.connect(on_probe_field_change)
    probe_color_mode_box.currentIndexChanged.connect(on_probe_color_mode_change)

    # ---------------------------
    # ---------- START ----------
    # ---------------------------
    on_variable_change()
    update_comfort_probes(current_step)
