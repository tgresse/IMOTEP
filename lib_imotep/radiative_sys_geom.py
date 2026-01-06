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

from numba import njit
import numpy as np
import quadpy
import trimesh

from lib_imotep.utils import global_timer

class RadiativeSystemGeometry:

    def __init__(self, transmissivity_sw_list, opposite_transmissivity_sw_list, opposite_facet_index_list, probe_opposite_transmissivity_sw_list,
                      tripoints_list,  trinormal_list, triarea_list,
                      procoord_list,
                      combined_mesh):
        self.transmissivity_sw_list = transmissivity_sw_list
        self.opposite_transmissivity_sw_list = opposite_transmissivity_sw_list
        self.opposite_facet_index_list = opposite_facet_index_list
        self.probe_opposite_transmissivity_sw_list = probe_opposite_transmissivity_sw_list

        self.tripoints_list = tripoints_list
        self.trinormal_list = trinormal_list
        self.triarea_list = triarea_list
        self.n_triangles = len(tripoints_list)
        self.n_facets = self.n_triangles + 1 # add the sky that is not a triangle

        self.procoord_list = procoord_list
        self.n_probes = len(procoord_list)

        self.pronormal_list = [np.array([0., 0., 1.]), np.array([0., 0., -1.]),  # top, bottom
                               np.array([0., 1., 0.]), np.array([0., -1., 0.]),  # north, south
                               np.array([1., 0., 0.]), np.array([-1., 0., 0.])]  # east, west

        tmesh = trimesh.Trimesh(vertices=combined_mesh.points,
                                faces=combined_mesh.regular_faces,
                                process=False)
        self.ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(tmesh)

        # to compute facet view factors
        scheme = quadpy.t2.get_good_scheme(12)
        self.scheme_points = scheme.points
        self.scheme_weights = scheme.weights

    def compute_visibility_matrix(self):
        print(f"    Computing the visibility matrix...")

        global_timer.start('process')

        n_f = self.n_facets

        tripoints_list = self.tripoints_list
        trinormal_list = self.trinormal_list
        opposite_facet_index_list = self.opposite_facet_index_list

        ray_intersector = self.ray_intersector

        # initialize the visibility matrices at False
        fully_visibility_matrix = np.zeros((n_f, n_f), dtype=np.bool_)
        partially_visibility_matrix = np.zeros((n_f, n_f), dtype=np.bool_)

        # precompute all (i, j) index pairs where j > i
        # span until n_f-1 because the last facet is the sky which is not a triangle
        pairs = [(i, j) for i in range(n_f - 1) for j in range(i + 1, n_f - 1)]

        # loop through (i,j) pairs
        for count, (i, j) in enumerate(pairs):
            if count % 10000 == 0:
                print(f"        ...processing pair {count} / {len(pairs)} ({i}-{j})")

            # check visibility without obstruction
            # 0: not visible
            # 1: fully visible
            # ]0,1[: partially visible
            vis_ij = get_visibility(tripoints_list[i], trinormal_list[i],
                                    tripoints_list[j], trinormal_list[j])

            # if partially visible
            if 0. < vis_ij < 1.:
                partially_visibility_matrix[i, j] = True
                partially_visibility_matrix[j, i] = True

            # if fully visible
            elif vis_ij == 1:
                # check obstruction
                # obs_ij between 0 and 1
                # no obstruction: obs_ij == 0.0
                obs_ij = get_obstruction(i, opposite_facet_index_list[i],
                                         j, opposite_facet_index_list[j],
                                         tripoints_list[i], tripoints_list[j],
                                         ray_intersector)

                if obs_ij == 0.:
                    fully_visibility_matrix[i, j] = True
                    fully_visibility_matrix[j, i] = True

                elif 0. < obs_ij < 1.:
                    partially_visibility_matrix[i, j] = True
                    partially_visibility_matrix[j, i] = True

        global_timer.stop('process', print_type='second')

        self.fully_visibility_matrix = fully_visibility_matrix
        self.partially_visibility_matrix = partially_visibility_matrix

    def compute_viewfactor_matrix(self):
        """
        Compute view factors between all domain facets using analytical integration
        and partial visibility projection-based clipping when needed.
        """
        print(f"    Computing the view factor matrix...")
        global_timer.start('process')

        n_f = self.n_facets
        VF_matrix = np.zeros((n_f, n_f))

        tripoints_list = self.tripoints_list
        trinormal_list = self.trinormal_list
        triarea_list = self.triarea_list
        opposite_facet_index_list = self.opposite_facet_index_list

        scheme_points = self.scheme_points
        scheme_weights = self.scheme_weights

        ray_intersector = self.ray_intersector

        # span until nf-1 because the last facet is the sky which is not a triangle
        fully_visible_pairs = [(i, j) for i in range(n_f - 1) for j in range(i + 1, n_f - 1)
                               if self.fully_visibility_matrix[i, j]]
        partially_visible_pairs = [(i, j) for i in range(n_f - 1) for j in range(i + 1, n_f - 1)
                                   if self.partially_visibility_matrix[i, j]]

        # compute view factor with integral method for fully visible and not obstructed facets
        print(f"      fully visible pairs (contour integral): {len(fully_visible_pairs)}")
        for count, (i, j) in enumerate(fully_visible_pairs):
            if count % 10000 == 0:
                print(f"        ...processing pair {count} / {len(fully_visible_pairs)} ({i}-{j})")

            vf_ij = compute_viewfactor_integral_njit(tripoints_list[i], trinormal_list[i],
                                                     tripoints_list[j],
                                                     scheme_points, scheme_weights)

            VF_matrix[i, j] = vf_ij
            VF_matrix[j, i] = vf_ij * triarea_list[i] / triarea_list[j]

        global_timer.stop('process', print_type='third')

        # compute view factor with Monte-Carlo method for partially visible or obstructed facets
        print(f"      partially visible pairs (Monte Carlo): {len(partially_visible_pairs)}")
        global_timer.start('process')

        for count, (i, j) in enumerate(partially_visible_pairs):
            if count % 10000 == 0:
                print(f"        ...processing pair {count} / {len(partially_visible_pairs)} ({i}-{j})")

            vf_ij = compute_viewfactor_monte_carlo(
                    idx1=i, idx1_opp=opposite_facet_index_list[i],
                    idx2=j, idx2_opp=opposite_facet_index_list[j],
                    tripoints1=tripoints_list[i], trinormal1=trinormal_list[i],
                    tripoints2=tripoints_list[j], trinormal2=trinormal_list[j],
                    triarea2=triarea_list[j],
                    ray_intersector=ray_intersector,
                    precision=1e-2,
                    max_samples=5_000_000
            )

            VF_matrix[i, j] = vf_ij
            VF_matrix[j, i] = vf_ij * triarea_list[i] / triarea_list[j]

        global_timer.stop('process', print_type='third')

        # Sky view factors
        tol = 1e-4
        tol_warning = 1e-2
        for i in range(self.n_facets - 1):
            s = VF_matrix[i, :-1].sum()
            if s > 1 + tol:
                if s > 1 + tol_warning:
                    print(f"        [WARNING] Sum of viewfactors (except with sky) at triangle {tripoints_list[i]}, normal {trinormal_list[i]} exceed 1: {s:.3f} -> facet viewfactors renormalized.")
                # renormalize facet VFs to sum to 1
                VF_matrix[i, :-1] *= 1.0 / s
                vf_sky = 0.0
            else:
                vf_sky = max(0.0, 1.0 - s)
            VF_matrix[i, -1] = vf_sky

            # vf_sky = 1.0 - np.sum(VF_matrix[i, :-1])
            # if vf_sky < -1e-2:
            #     print(f"        [WARNING] Significantly negative sky VF for facet at coord {tripoints_list[i]} and normal {trinormal_list[i]}: {vf_sky} -> set to zero")
            # VF_matrix[i, -1] = max(0.0, vf_sky)

        # Note: the last row of the view factor matrix (sky -> facets) is kept null because not needed

        return VF_matrix

    def compute_visible_probes(self):
        print("    Computing visible probes...")
        global_timer.start('process')

        # each list contains one element per triangle
        # each element is a list of tuples
        # each tuple contains the probe index followed by the normal indices for which the triangle is either fully or
        # partially visible from the probe
        fully_visible_probe_list = []
        partially_visible_probe_list = []

        len_tripoints_list = len(self.tripoints_list)
        opposite_facet_index_list = self.opposite_facet_index_list
        ray_intersector = self.ray_intersector

        procoord_arr = np.array(self.procoord_list)
        pronormal_arr = np.array(self.pronormal_list)

        for tri_idx, (tripoints, trinormal) in enumerate(zip(self.tripoints_list, self.trinormal_list)):
            if tri_idx % 100 == 0:
                print(f"        ...processing triangle {tri_idx} / {len_tripoints_list}")
            fully_visible_per_triangle = []
            partially_visible_per_triangle = []
            opp_idx = opposite_facet_index_list[tri_idx]

            for pro_idx, propoint in enumerate(procoord_arr):
                # visibility check
                fully_mask, partial_mask = get_probe_visibility(propoint, pronormal_arr, tripoints, trinormal)

                # array of normal direction indices for which the triangle is visible (fully or partially) from the probe
                fully_normal_indices = np.where(fully_mask)[0]
                partial_normal_indices = np.where(partial_mask)[0]

                # the triangle is not visible by the probe in any direction (case when the probe is behind the triangle)
                # -> go to next probe
                if len(fully_normal_indices) == 0 and len(partial_normal_indices) == 0:
                    continue

                # check if there is an obstruction between the probe and the triangle
                # obstruction between 0 and 1
                # no obstruction: obstruction == 0.0
                obstruction = get_probe_obstruction(tri_idx, opp_idx, propoint, tripoints, ray_intersector)

                # if the visibility is fully obstructed between the triangle and the probe -> go to next probe
                if obstruction == 1.0:
                    continue

                # fully visible case..
                if len(fully_normal_indices) > 0:
                    # .. and no obstruction -> fully visibility confirmed
                    if obstruction == 0.0:
                        fully_visible_per_triangle.append((pro_idx, *fully_normal_indices))
                    # .. with partial obstruction -> partial visibility
                    else:
                        partially_visible_per_triangle.append((pro_idx, *fully_normal_indices, *partial_normal_indices))
                        # continue to prevent going through the partially visible case as the partial_normal_indices are already added
                        continue

                # partially visible case -> confirmed whatever the obstruction status
                if len(partial_normal_indices) > 0:
                    partially_visible_per_triangle.append((pro_idx, *partial_normal_indices))

            fully_visible_probe_list.append(fully_visible_per_triangle)
            partially_visible_probe_list.append(partially_visible_per_triangle)

        self.fully_visible_probe_list = fully_visible_probe_list
        self.partially_visible_probe_list = partially_visible_probe_list

        global_timer.stop('process', print_type='second')

    def compute_probe_viewfactor_matrix(self):
        """
        Compute view factors between all domain facets using analytical integration
        and partial visibility projection-based clipping when needed.
        """
        print("    Computing probe view factor matrix...")
        global_timer.start('process')

        n_facets = self.n_facets
        n_probes = self.n_probes
        VF_matrix = np.zeros((n_probes, 6, n_facets))

        len_fully_visible_probe_list = len(self.fully_visible_probe_list)
        len_partially_visible_probe_list = len(self.partially_visible_probe_list)

        tripoints_list = self.tripoints_list
        trinormal_list = self.trinormal_list
        triarea_list = self.triarea_list
        opposite_facet_index_list = self.opposite_facet_index_list

        procoord_list = self.procoord_list
        pronormal_list = self.pronormal_list

        ray_intersector = self.ray_intersector

        # Fully visible triangles
        print(f"      fully visible triangles (contour integral): {len_fully_visible_probe_list}")
        for tri_idx, probe_list in enumerate(self.fully_visible_probe_list):
            if tri_idx % 100 == 0:
                print(f"        ...processing triangle {tri_idx} / {len_fully_visible_probe_list}")

            tripoints = tripoints_list[tri_idx]

            for probe_info in probe_list:
                probe_idx, *normal_indices = probe_info
                normals = np.array([pronormal_list[j] for j in normal_indices])

                vf_values = compute_probe_viewfactor_integral_jit(
                        propoint=procoord_list[probe_idx],
                        pronormals=normals,
                        tripoints=tripoints
                )

                VF_matrix[probe_idx, normal_indices, tri_idx] = vf_values

        global_timer.stop('process', print_type='third')

        # Partially visible triangles
        print(f"      partially visible triangles (Monte Carlo): {len_partially_visible_probe_list}")
        global_timer.start('process')

        for tri_idx, probe_list in enumerate(self.partially_visible_probe_list):
            if tri_idx % 100 == 0:
                print(f"        ...processing triangle {tri_idx} / {len_partially_visible_probe_list}")

            tripoints = tripoints_list[tri_idx]
            trinormal = trinormal_list[tri_idx]
            triarea = triarea_list[tri_idx]
            opp_idx = opposite_facet_index_list[tri_idx]

            for probe_info in probe_list:
                probe_idx, *normal_indices = probe_info
                normals = np.array([pronormal_list[j] for j in normal_indices])

                vf_values = compute_probe_viewfactor_monte_carlo(
                        idx1=None,
                        idx1_opp=None,
                        idx2=tri_idx,
                        idx2_opp=opp_idx,
                        propoint1=procoord_list[probe_idx],
                        pronormals1=normals,
                        tripoints2=tripoints,
                        trinormal2=trinormal,
                        triarea2=triarea,
                        ray_intersector=ray_intersector,
                        precision=1e-2,
                        max_samples=5_000_000
                )

                VF_matrix[probe_idx, normal_indices, tri_idx] = vf_values

        global_timer.stop('process', print_type='third')

        # Sky view factor
        tol = 1e-4
        tol_warning = 1e-2
        for i in range(n_probes):
            for j in range(6):
                s = VF_matrix[i, j, :-1].sum()
                if s > 1 + tol:
                    if s > 1 + tol_warning:
                        print(f"        [WARNING] Sum of viewfactors (except with sky) at probe {procoord_list[i]}, normal {pronormal_list[j]} exceed 1: {s:.3f} -> facet viewfactors renormalized.")
                    # renormalize facet VFs to sum to 1
                    VF_matrix[i, j, :-1] *= 1.0 / s
                    vf_sky = 0.0
                else:
                    vf_sky = max(0.0, 1.0 - s)
                VF_matrix[i, j, -1] = vf_sky

                # VF_matrix[i, j, -1] = vf_sky
                # vf_sky = 1.0 - np.sum(VF_matrix[i, j, :-1])
                # if vf_sky < -1e-2:
                #     print(f"        [WARNING] Significantly negative sky VF at probe {procoord_list[i]}, normal {pronormal_list[j]}: {vf_sky:.3f} -> clamped to 0")
                # VF_matrix[i, j, -1] = max(0.0, vf_sky)

        # Note: the last row of the view factor matrix (sky -> facets) is kept null because not needed

        return VF_matrix

    def compute_primary_direct_irradiance_prefactor(self, sunray_direction_list):
        """Compute the primary shortwave heat flux prefactors on facets considering obstructions."""
        print("    Computing primary direct irradiance prefactors...")
        global_timer.start('process')

        n_f = self.n_facets
        n_dir = len(sunray_direction_list)

        batch_size = 2000

        opposite_facet_index_list = self.opposite_facet_index_list
        opposite_transmissivity_sw_list = self.opposite_transmissivity_sw_list
        ray_intersector = self.ray_intersector

        primary_direct_irradiance_partial_obstruction_per_sundir_arr = np.full((n_dir, n_f), False)
        primary_direct_irradiance_prefactor_per_sundir_arr = np.zeros((n_dir, n_f))

        for i, sun_dir in enumerate(sunray_direction_list):
            if sun_dir[2] >= .0:
                continue

            sun_dir = np.asarray(sun_dir)
            neg_sun_dir = -sun_dir
            direction_arr = np.tile(neg_sun_dir, (batch_size, 1))

            for j, (tripoints, trinormal, trans) in enumerate(zip(self.tripoints_list, self.trinormal_list, self.transmissivity_sw_list)):
                j_opp = opposite_facet_index_list[j]
                # STEP 1: no obstruction
                # compute the primary irradiance prefactor uniquely based on face orientation (no obstruction)
                prefactor = np.dot(trinormal, neg_sun_dir)

                # if the incident flux is not null then the facet might see the sun (no obstruction)
                if prefactor <= 1e-9:
                    continue

                # STEP 2: obstruction
                # compute the visibility ratio between the sun and facet i (from the sun point of view) pondered by the transmissivity of obstruction facets
                # check for obstruction
                r = np.random.rand(2, batch_size)
                origin_arr = get_random_points_on_triangle(tripoints, r) + 1e-2 * neg_sun_dir

                # ray_hit_arr ((m,) int): Indexes of combined_mesh.faces
                # ray_id_arr ((m,) int): Indexes of ray
                # both arrays can be empty if no obstruction
                ray_hit_arr, ray_id_arr = ray_intersector.intersects_id(ray_origins=origin_arr,
                                                                        ray_directions=direction_arr)

                transmissivity_partial_obstruction = False
                if len(ray_id_arr) == 0:
                    transmissivity_upsteam = 1.

                else:
                    transmissivity_arr = np.ones(batch_size)
                    for k, (ray_hit, ray_id) in enumerate(zip(ray_hit_arr, ray_id_arr)):
                        # the facet that obstruct the visibility (of index ray_hit), might be different from the index of the origin facet or its opposite
                        if ray_hit not in [j, j_opp]:
                            transmissivity_arr[ray_id] *= opposite_transmissivity_sw_list[ray_hit]
                            if 0. < opposite_transmissivity_sw_list[ray_hit] < 1.:
                                transmissivity_partial_obstruction = True
                    transmissivity_upsteam = np.sum(transmissivity_arr) / batch_size

                if transmissivity_partial_obstruction:
                    primary_direct_irradiance_partial_obstruction_per_sundir_arr[i, j] = True

                primary_direct_irradiance_prefactor_per_sundir_arr[i, j] = prefactor * transmissivity_upsteam

        global_timer.stop('process', print_type='second')

        return (primary_direct_irradiance_prefactor_per_sundir_arr,
                primary_direct_irradiance_partial_obstruction_per_sundir_arr)

    def compute_probe_primary_direct_irradiance_prefactor(self, sunray_direction_list):
        """Compute the primary shortwave heat flux prefactors on probes considering obstructions."""
        print("    Computing probe primary direct irradiance prefactors...")
        global_timer.start('process')

        n_p = self.n_probes
        n_dir = len(sunray_direction_list)

        pronormal_list = self.pronormal_list
        procoord_list = self.procoord_list
        ray_intersector = self.ray_intersector
        probe_opposite_transmissivity_sw_list = self.probe_opposite_transmissivity_sw_list

        probe_primary_direct_irradiance_partial_obstruction_per_sundir_arr = np.full((n_dir, n_p), False)
        probe_primary_direct_irradiance_prefactor_per_sundir_arr = np.zeros((n_dir, n_p, 6))

        origin_arr = np.array([pt1 for pt1 in procoord_list])

        for i, sun_dir in enumerate(sunray_direction_list):
            if sun_dir[2] >= .0:
                continue

            sun_dir = np.asarray(sun_dir)
            neg_sun_dir = -sun_dir

            # STEP 1: no obstruction, no transmission
            # compute the primary irradiance prefactor uniquely based on face orientation (no obstruction)
            prefactor_arr = np.dot(pronormal_list, neg_sun_dir)
            unvisible = prefactor_arr < 0.0
            prefactor_arr[unvisible] = 0.0

            # STEP 2: obstruction
            # check for obstruction
            direction_arr = np.tile(neg_sun_dir, (n_p, 1))

            # ray_hit_arr ((m,) int): Indexes of combined_mesh.faces
            # ray_id_arr ((m,) int): Indexes of ray
            # both arrays can be empty if no obstruction
            ray_hit_arr, ray_id_arr = ray_intersector.intersects_id(ray_origins=origin_arr,
                                                                    ray_directions=direction_arr)

            transmissivity_partial_obstruction_arr = np.full(n_p, False)
            transmissivity_arr = np.ones(n_p)
            for ray_hit, ray_id in zip(ray_hit_arr, ray_id_arr):
                transmissivity_arr[ray_id] *= probe_opposite_transmissivity_sw_list[ray_hit]
                if 0. < probe_opposite_transmissivity_sw_list[ray_hit] < 1.:
                    transmissivity_partial_obstruction_arr[ray_id] = True

            for j, trans in enumerate(transmissivity_arr):
                if transmissivity_partial_obstruction_arr[j]:
                    probe_primary_direct_irradiance_partial_obstruction_per_sundir_arr[i, j] = True
                for k, prefactor in enumerate(prefactor_arr):
                    probe_primary_direct_irradiance_prefactor_per_sundir_arr[i, j, k] = prefactor * trans

        global_timer.stop('process', print_type='second')

        return (probe_primary_direct_irradiance_prefactor_per_sundir_arr,
                probe_primary_direct_irradiance_partial_obstruction_per_sundir_arr)

#-----------------------------------------------------------------------------------------------------------------------
# useful functions

@njit
def get_visibility(tripoints1, trinormal1, tripoints2, trinormal2):
    """
    Estimate visibility based on mutual orientation of triangle vertices.
    Returns a score between 0 (not visible) and 1 (fully visible).
    The function takes special care of the points that are located in the plane of the other triangle.
    This point should be considered as visible if the remaining parts of the triangles are visible to each other.
    It should be considered as non-visible if the remaining part of the triangles are hidden to each other
    """
    # warning: the visibility check is done without obstruction check
    # visibility_count : count the number of vertices that are visible to each other in the direction of the triangle normals
    # point_in_plane_count : count the number of vertex pairs that indicates that one of the two vertices is in the plane of the other triangle
    visibility_count = 0
    point_in_plane_count = 0

    for pt1 in tripoints1:
        for pt2 in tripoints2:
            # vector vertex triangle1 -> vertex triangle2
            v12 = pt2 - pt1

            # dot1: v12 . triangle1 normal
            # dot1: v12 . triangle2 normal
            dot1 = v12[0]*trinormal1[0] + v12[1]*trinormal1[1] + v12[2]*trinormal1[2]
            dot2 = v12[0]*trinormal2[0] + v12[1]*trinormal2[1] + v12[2]*trinormal2[2]

            # if dot1 positive and dot2 negative, the two vertices are clearly visible
            if dot1 > 1e-6 and dot2 < -1e-6:
                visibility_count += 1
            # if dot1=0, vertex triangle2 in the plane of triangle1
            # if dot2=0, vertex triangle1 in the plane of triangle2
            elif np.abs(dot1) < 1e-6 or np.abs(dot2) < 1e-6:
                point_in_plane_count += 1

    # no vertex pairs clearly visible -> point_in_plane are ignored, no visibility between the triangles
    if visibility_count == 0:
        return 0.0
    # otherwise, compute the visibility ]0, 1]
    else:
        return (visibility_count + point_in_plane_count) / 9.

def get_obstruction(idxtri1, idxtri1_opp, idxtri2, idxtri2_opp,
                    tripoints1, tripoints2, ray_intersector):
    batch_size = 100

    # generate batch_size random points in the triangles
    r = np.random.rand(4, batch_size)
    s1 = get_random_points_on_triangle(tripoints1, r[:2])
    s2 = get_random_points_on_triangle(tripoints2, r[2:])

    # vectors triangle1 random point -> triangle2 random point
    v12 = s2 - s1
    dist = np.linalg.norm(v12, axis=1)
    valid_rays = dist > 1e-2
    num_valid_rays = valid_rays.sum()
    s1 = s1[valid_rays]
    v12 = v12[valid_rays]
    dist = dist[valid_rays]
    # unit vectors triangle1 random point -> triangle2 random point
    dir_v = v12 / dist[:, None]

    # ray tracing for the finding obstructed rays
    # the function returns an array with one value per ray:
    # the index of the first triangle intersected
    # -1 if no intersection found (should never happen because the rays are supposed to intersect at least the destination triangle)
    # An offset of 1e-2 was intentionally implemented to avoid intersecting source triangle or its opposite. Its relatively high value
    # is necessary especially in the case of tilted surfaces to avoid self intersecting for most of the cases.
    ray_hits = ray_intersector.intersects_first(ray_origins=s1 + 1e-2 * dir_v, ray_directions=dir_v)

    # detect rays that intersect the origin facets or their opposites
    origin_intersect_rays = (ray_hits == idxtri1) | (ray_hits == idxtri1_opp)
    num_origin_intersect_rays = origin_intersect_rays.sum()
    if num_origin_intersect_rays > 0.05 * num_valid_rays:
        print(f"        Many detections of rays that intersect the origin facets or their opposites for obstruction calculation ({num_origin_intersect_rays} detected over {num_valid_rays} rays).\n"
              f"            [Info: tripoints1= [{tripoints1[0]}, {tripoints1[1]}, {tripoints1[2]}] /\n"
              f"                   tripoints2= [{tripoints2[0]}, {tripoints2[1]}, {tripoints2[2]}]]")
    if num_origin_intersect_rays > 0.5 * num_valid_rays:
        # raise error if too many self-intersections
        raise ValueError(f" Too many detections of rays that intersect the origin facets or their opposites for obstruction calculation ({num_origin_intersect_rays} detected over {num_valid_rays} rays).\n"
                         f"     [Info: tripoints1= [{tripoints1[0]}, {tripoints1[1]}, {tripoints1[2]}] /\n"
                         f"            tripoints2= [{tripoints2[0]}, {tripoints2[1]}, {tripoints2[2]}]]")

    # filter rays that hit on the destination triangle or its opposite
    # note: if the destination triangle has no opposite triangle, idxtri2_opp is None and the second condition is True
    unobstructed = (ray_hits == idxtri2) | (ray_hits == idxtri2_opp)
    num_unobstructed = unobstructed.sum()

    return (num_valid_rays - num_origin_intersect_rays - num_unobstructed) / (num_valid_rays - num_origin_intersect_rays)

@njit
def get_probe_visibility(propoint, pronormals, tripoints, trinormal):
    # warning: the visibility check is done without obstruction check
    # number of normals for the probe
    n_n = pronormals.shape[0]
    # initialize boolean masks for directions where the triangle is fully and partially visible from the probe
    fully_visible_mask = np.zeros(n_n, dtype=np.bool_)
    partially_visible_mask = np.zeros(n_n, dtype=np.bool_)

    # loop through normals
    for i in range(n_n):
        # nb of visible triangle vertices
        visibility_count = 0
        # nb of triangle vertices seen from the probe in a direction perpendicular to the normal
        point_in_perpendicular_direction_count = 0

        # loop through triangle vertices
        for j in range(3):
            # vector probe -> vertex
            v21 = tripoints[j] - propoint

            # dot1 = v21 . probe normal
            # dot2 = v21 . triangle normal
            dot1 = v21[0] * pronormals[i][0] + v21[1] * pronormals[i][1] + v21[2] * pronormals[i][2]
            dot2 = v21[0] * trinormal[0] + v21[1] * trinormal[1] + v21[2] * trinormal[2]

            # if dot2=0, the probe is in the plane of the triangle -> no visiblity
            if np.abs(dot2) < 1e-6:
                break

            # if dot1 positive and dot2 negative -> triangle is facing the probe in the selected direction
            if dot1 > 1e-6 and dot2 < -1e-6:
                visibility_count += 1

            # if dot1=0, the vertex is seen from the probe in a direction perpendicular to the normal
            elif np.abs(dot1) < 1e-6:
                point_in_perpendicular_direction_count += 1

        # no visible vertex -> step to the next normal
        if visibility_count == 0:
            continue

        # if all vertices are clearly visible, or at least one vertex is clearly visible and the other ones
        # are in perpendicular direction -> the triangle is considered as fully visible
        if visibility_count == 3 or visibility_count + point_in_perpendicular_direction_count == 3:
            fully_visible_mask[i] = True
        # otherwise (there is still at least one fully visible vertex) -> the triangle is partially visible
        else:
            partially_visible_mask[i] = True

    return fully_visible_mask, partially_visible_mask

def get_probe_obstruction(idxtri, idxtri_opp, propoint1, tripoints2, ray_intersector):
    batch_size = 100

    # duplicate probe coordinates batch_size times
    s1 = np.tile(propoint1, (batch_size, 1))

    # generate batch_size random points in the triangle
    r = np.random.rand(2, batch_size)
    s2 = get_random_points_on_triangle(tripoints2, r)

    # vectors probe -> triangle random point
    v21 = s2 - s1
    dist = np.linalg.norm(v21, axis=1)

    # remove random points if too close to the probe to avoid dividing by zero
    valid = dist > 1e-9
    s1 = s1[valid]
    v21 = v21[valid]
    dist = dist[valid]
    # unit vectors probe -> triangle random point
    dir_v = v21 / dist[:, None]

    # ray tracing for the finding obstructed rays
    # the function returns an array with one value per ray:
    # the index of the first triangle intersected
    # -1 if no intersection found (should never happen because the rays are supposed to intersect at least the destination triangle)
    ray_hits = ray_intersector.intersects_first(ray_origins=s1, ray_directions=dir_v)
    # filter rays that hit on triangles other that the destination triangle or its opposite
    obstructed = (ray_hits != idxtri) & (ray_hits != idxtri_opp)

    return obstructed.sum() / batch_size

@njit
def get_random_points_on_triangle(points, r):
    sqrt_r1 = np.sqrt(r[0])
    sqrt_r1_r2 = sqrt_r1 * r[1]

    return ((1 - sqrt_r1)[:, None] * points[0] +
          (sqrt_r1 - sqrt_r1_r2)[:, None] * points[1] +
          sqrt_r1_r2[:, None] * points[2])

@njit
def compute_differential_viewfactor(trinormal1, trinormal2, area2, dir_v, dist):
    # scalar products
    cos_i = dir_v @ trinormal1
    cos_j = -dir_v @ trinormal2

    pos_mask = (cos_i > 0) & (cos_j > 0)
    if not np.any(pos_mask):
        return 0.0, 0.0

    cos_i_pos = cos_i[pos_mask]
    cos_j_pos = cos_j[pos_mask]
    dist_sq = dist[pos_mask] ** 2

    factor = cos_i_pos * cos_j_pos / (np.pi * dist_sq)

    dF = area2 * np.sum(factor)
    dF2 = np.sum((area2 * factor) ** 2)

    return dF, dF2

@njit
def compute_probe_differential_viewfactor(trinormals1, trinormal2, area2, dir_v, dist, eps=0.0):
    n_normals = trinormals1.shape[0]
    if dir_v.shape[0] == 0:  # no rays
        return np.zeros(n_normals), np.zeros(n_normals)

    cos_j = -np.dot(dir_v, trinormal2)
    mask_j = cos_j > 0.0
    if not np.any(mask_j):
        return np.zeros(n_normals), np.zeros(n_normals)

    dir_v_valid = dir_v[mask_j]
    cos_j_valid = cos_j[mask_j]
    dist_sq_valid = dist[mask_j]**2 + eps*eps

    # rows=i normals, cols=samples
    cos_i_mat = np.dot(dir_v_valid, trinormals1.T).T
    mask_i = cos_i_mat > 0.0
    cos_i_valid = cos_i_mat * mask_i

    cj = cos_j_valid[np.newaxis, :]
    inv_kernel = (cos_i_valid * cj) / (np.pi * dist_sq_valid[np.newaxis, :])

    per_sample = area2 * inv_kernel  # per-ray contributions
    dF_arr  = np.sum(per_sample, axis=1)
    dF2_arr = np.sum(per_sample**2, axis=1)
    return dF_arr, dF2_arr

def compute_viewfactor_monte_carlo(
    idx1, idx1_opp, idx2, idx2_opp,
    tripoints1, trinormal1, tripoints2, trinormal2,
    triarea2, ray_intersector,
    precision=1e-3, max_samples=50_000_000
):
    batch_size = 10000
    view_factor_sum = 0.0
    squared_sum = 0.0
    total_samples = 0
    check_obstruction = True
    print_warning = True

    while total_samples < max_samples:
        r = np.random.rand(4, batch_size)
        s1 = get_random_points_on_triangle(tripoints1, r[:2])
        s2 = get_random_points_on_triangle(tripoints2, r[2:])

        v21 = s2 - s1
        dist = np.linalg.norm(v21, axis=1)
        valid_rays = dist > 1e-2
        num_valid_rays = valid_rays.sum()
        if not np.any(valid_rays):
            continue

        s1 = s1[valid_rays]
        v21 = v21[valid_rays]
        dist = dist[valid_rays]
        dir_v = v21 / dist[:, None]

        # filter rays for which the positions on both facets are visible to each other
        # dir from s1 -> s2 must lie in the +hemisphere of n1 and the -hemisphere of n2
        # i.e., dot(dir_v, n1) >= eps  and  dot(-dir_v, n2) >= eps  <=>  dot(dir_v, n2) <= -eps
        visible_rays = (dir_v @ trinormal1 >= 0.) & (dir_v @ trinormal2 <= 0.)
        s1 = s1[visible_rays]
        dist = dist[visible_rays]
        dir_v = dir_v[visible_rays]

        if check_obstruction:
            # An offset of 1e-2 was intentionally implemented to avoid intersecting source triangle or its opposite. Its relatively high value
            # is necessary especially in the case of tilted surfaces to avoid self intersecting for most of the cases.
            ray_hits = ray_intersector.intersects_first(ray_origins=s1 + 1e-2 * dir_v, ray_directions=dir_v)

            # detect rays that intersect the origin facets
            origin_intersect_rays = (ray_hits == idx1) | (ray_hits == idx1_opp)
            num_origin_intersect_rays = origin_intersect_rays.sum()
            if num_origin_intersect_rays > 0.05 * num_valid_rays:
                if print_warning:
                    print(f"        Many detections of rays that intersect the origin facets or their opposites for viewfactor calculation ({num_origin_intersect_rays} detected over {num_valid_rays} rays).\n"
                          f"            [Info: tripoints1= [{tripoints1[0]}, {tripoints1[1]}, {tripoints1[2]}], trinormal1= {trinormal1} /\n"
                          f"                   tripoints2= [{tripoints2[0]}, {tripoints2[1]}, {tripoints2[2]}], trinormal2= {trinormal2}]")
                    print_warning = False
            if num_origin_intersect_rays > 0.5 * num_valid_rays:
                # raise error if too many self-intersections
                raise ValueError(f" Too many detections of rays that intersect the origin facets or their opposites for viewfactor calculation ({num_origin_intersect_rays} detected over {num_valid_rays} rays).\n"
                                 f"     [Info: tripoints1= [{tripoints1[0]}, {tripoints1[1]}, {tripoints1[2]}], trinormal1= {trinormal1} /\n"
                                 f"            tripoints2= [{tripoints2[0]}, {tripoints2[1]}, {tripoints2[2]}], trinormal2= {trinormal2}]")

            unobstructed = (ray_hits == idx2) | (ray_hits == idx2_opp)
            # once check obstruction is False at the end of the first loop, the obstruction test will not be performed in remaining loops
            if np.all(unobstructed):
                check_obstruction = False
        else:
            unobstructed = np.ones(len(s1), dtype=bool)
            num_origin_intersect_rays = 0

        # if all rays are obstructed, the VF must remain zero
        if not np.any(unobstructed):
            return 0.0

        dir_v = dir_v[unobstructed]
        dist = dist[unobstructed]

        dF, dF2 = compute_differential_viewfactor(trinormal1, trinormal2, triarea2, dir_v, dist)

        # if systematic negative cos -> no visibility -> VF = 0 (should never happen)
        if dF == 0.0:
            return 0.0

        view_factor_sum += dF
        squared_sum += dF2
        total_samples += (batch_size - num_origin_intersect_rays)

        estimate = view_factor_sum / total_samples
        variance = (squared_sum - (view_factor_sum ** 2) / total_samples) / (total_samples - 1)
        std_err = np.sqrt(variance / total_samples)

        if std_err / estimate < precision:
            break


    if std_err / estimate > precision:
        print(f"        [WARNING] View factor not converged (samples= {total_samples}, error= {std_err / estimate:.2e})")

    return view_factor_sum / total_samples

@njit
def compute_viewfactor_integral_njit(tripoints1, trinormal1, tripoints2, scheme_points, scheme_weights):
    num_quad_points = scheme_points.shape[1]  # 33
    cartesian_points = np.zeros((num_quad_points, 3))  # (33, 3)

    # Barycentric to Cartesian: cartesian_points = scheme_points.T @ tripoints1
    for i in range(num_quad_points):
        for d in range(3):  # x, y, z
            for b in range(3):  # barycentric coordinate index
                cartesian_points[i, d] += scheme_points[b, i] * tripoints1[b, d]

    result = 0.0
    for j in range(3):
        M_km = tripoints2[j]
        M_kp = tripoints2[(j + 1) % 3]

        for i in range(num_quad_points):
            M_0 = cartesian_points[i]

            M_0M_km = M_0 - M_km
            M_0M_kp = M_0 - M_kp

            cross_prod = np.cross(M_0M_km, M_0M_kp)
            norm_cross = np.linalg.norm(cross_prod)

            if norm_cross == 0.0:
                continue

            dot_prod = np.dot(M_0M_km, M_0M_kp)
            beta_k = np.arctan2(norm_cross, dot_prod)
            dot_trinormal_cross = np.dot(trinormal1, cross_prod)

            fval = beta_k * dot_trinormal_cross / norm_cross

            result += scheme_weights[i] * fval

    return -result / (2.0 * np.pi)

def compute_probe_viewfactor_monte_carlo(
    idx1, idx1_opp, idx2, idx2_opp,
    propoint1, pronormals1, tripoints2, trinormal2,
    triarea2, ray_intersector,
    precision=1e-3, max_samples=50_000_000
):
    batch_size = 10000
    n_n = len(pronormals1)
    view_factor_sum_arr = np.zeros(n_n)
    squared_sum_arr = np.zeros(n_n)
    total_samples_arr = np.zeros(n_n, dtype=np.int32)
    estimate_arr = np.zeros(n_n)
    variance_arr = np.zeros(n_n)
    std_err_arr = np.zeros(n_n)
    converged_mask = np.zeros(n_n, dtype=bool)
    check_obstruction = True

    s1 = np.tile(propoint1, (batch_size, 1))

    while np.any(~converged_mask) and np.max(total_samples_arr) < max_samples:
        r = np.random.rand(2, batch_size)
        s2 = get_random_points_on_triangle(tripoints2, r)

        v21 = s2 - s1
        dist = np.linalg.norm(v21, axis=1)
        valid = dist > 1e-9
        if not np.any(valid):
            continue

        s1 = s1[valid]
        v21 = v21[valid]
        dist = dist[valid]
        dir_v = v21 / dist[:, None]

        if check_obstruction:
            ray_hits = ray_intersector.intersects_first(ray_origins=s1, ray_directions=dir_v)
            unobstructed = (ray_hits == idx2) | (ray_hits == idx2_opp)
            if np.all(unobstructed):
                check_obstruction = False
        else:
            unobstructed = np.ones(len(s1), dtype=bool)

        if not np.any(unobstructed):
            return np.zeros(n_n)

        dir_v = dir_v[unobstructed]
        dist = dist[unobstructed]

        # Slice only active normals
        active_mask = ~converged_mask
        active_normals = pronormals1[active_mask]

        dF_active, dF2_active = compute_probe_differential_viewfactor(active_normals, trinormal2, triarea2, dir_v, dist)

        # Update only active indices
        view_factor_sum_arr[active_mask] += dF_active
        squared_sum_arr[active_mask] += dF2_active
        total_samples_arr[active_mask] += batch_size

        active = total_samples_arr > 0
        estimate_arr[active] = view_factor_sum_arr[active] / total_samples_arr[active]
        variance_arr[active] = (squared_sum_arr[active] - (view_factor_sum_arr[active] ** 2) / total_samples_arr[active]) / (total_samples_arr[active] - 1)
        std_err_arr[active] = np.sqrt(variance_arr[active] / total_samples_arr[active])

        converged_mask |= (std_err_arr / (estimate_arr + 1e-12)) < precision

    if not np.all(converged_mask):
        print(f"        [WARNING] View factor not converged (max_samples= {np.max(total_samples_arr)}, max_error= {np.max(std_err_arr / (estimate_arr + 1e-12)):.2e})")

    return view_factor_sum_arr / total_samples_arr


@njit
def compute_probe_viewfactor_integral_jit(propoint, pronormals, tripoints):
    dir_tripro_arr = np.empty((3, 3), dtype=np.float64)

    # Normalize vectors from probe to triangle points
    for i in range(3):
        v0 = tripoints[i][0] - propoint[0]
        v1 = tripoints[i][1] - propoint[1]
        v2 = tripoints[i][2] - propoint[2]
        norm = np.sqrt(v0*v0 + v1*v1 + v2*v2)
        if norm > 1e-12:
            dir_tripro_arr[i, 0] = v0 / norm
            dir_tripro_arr[i, 1] = v1 / norm
            dir_tripro_arr[i, 2] = v2 / norm
        else:
            dir_tripro_arr[i, :] = 0.0

    # Dot products between edge vectors
    cos_alpha_arr = np.empty(3, dtype=np.float64)
    pairs = [(0, 2), (2, 1), (1, 0)]
    for i in range(3):
        a = dir_tripro_arr[pairs[i][0]]
        b = dir_tripro_arr[pairs[i][1]]
        dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        cos_alpha_arr[i] = min(1.0, max(-1.0, dot))  # clamp

    # Compute alpha angles
    alpha_arr = np.empty(3, dtype=np.float64)
    for i in range(3):
        s = np.sqrt(1.0 - cos_alpha_arr[i] * cos_alpha_arr[i])
        alpha_arr[i] = np.arctan2(s, cos_alpha_arr[i])

    # Normalized cross products (triangle edge normals)
    normal_tripro_arr = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        a = dir_tripro_arr[pairs[i][0]]
        b = dir_tripro_arr[pairs[i][1]]
        cx = a[1]*b[2] - a[2]*b[1]
        cy = a[2]*b[0] - a[0]*b[2]
        cz = a[0]*b[1] - a[1]*b[0]
        norm = np.sqrt(cx*cx + cy*cy + cz*cz)
        if norm > 1e-12:
            normal_tripro_arr[i, 0] = cx / norm
            normal_tripro_arr[i, 1] = cy / norm
            normal_tripro_arr[i, 2] = cz / norm
        else:
            normal_tripro_arr[i, :] = 0.0

    # Dot product with pronormals
    N = pronormals.shape[0]
    vf_arr = np.empty(N, dtype=np.float64)
    for i in range(N):
        acc = 0.0
        for j in range(3):
            dot = (pronormals[i, 0] * normal_tripro_arr[j, 0] +
                   pronormals[i, 1] * normal_tripro_arr[j, 1] +
                   pronormals[i, 2] * normal_tripro_arr[j, 2])
            acc += dot * alpha_arr[j]
        vf_arr[i] = acc / (2.0 * np.pi)

    return vf_arr


if __name__ == "__main__":
    from _input_shelter_old import general_dict, weather_def_dict, panel_def_dict_list, airzone_def_dict_list, probe_set_def_list, output_def_dict
    from imotep import IMOTEP

    imotep = IMOTEP(general_dict,
                    weather_def_dict,
                    panel_def_dict_list,
                    airzone_def_dict_list,
                    probe_set_def_list,
                    output_def_dict)

    imotep._generate_model_objects()

    imotep._generate_rad_utility_lists()

    system_geometry = RadiativeSystemGeometry(imotep.rad_opposite_facet_index_list,
                                              imotep.rad_transmissivity_sw_list,
                                              imotep.rad_opposite_transmissivity_sw_list,
                                              imotep.rad_tripoints_list,
                                              imotep.rad_trinormal_list,
                                              imotep.rad_triarea_list,
                                              imotep.rad_procoord_list,
                                              imotep.mesh_list_dict['combined_domain'][0])

    system_geometry.compute_visibility_matrix()

    # import numpy as np
    # propoint = np.array([-2, 0, 1], dtype=np.float64)
    # pronormals = np.array([[1, 0, 0]], dtype=np.float64)
    # tripoints = np.array([ [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
    # trinormal = np.array([0, 0, 1], dtype=np.float64)
    # triarea = float(0.5)
    #
    # vf_mc = compute_probe_viewfactor_monte_carlo(
    #     None, None, None, None,
    #     propoint, pronormals, tripoints, trinormal,
    #     triarea, None,
    #     precision=1e-3, max_samples=50_000_000
    # )
    #
    # vf_int = compute_probe_viewfactor_integral(propoint, pronormals, tripoints)