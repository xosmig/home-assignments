from typing import List, Tuple, Optional

import time
from copy import deepcopy

import numpy as np
import sortednp as snp
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares, OptimizeResult

from _corners import filter_frame_corners_on_id
from corners import FrameCorners
from _camtrack import (
    PointCloudBuilder,
    compute_reprojection_errors,
    view_mat3x4_to_rodrigues_and_translation,
    rodrigues_and_translation_to_view_mat3x4
)


def _view_mat_to_param_vector(view_mat: np.ndarray) -> np.ndarray:
    assert view_mat.shape == (3, 4)
    r_vec, t_vec = view_mat3x4_to_rodrigues_and_translation(view_mat)
    return np.concatenate((r_vec.flatten(), t_vec.flatten()))


def _param_vector_to_view_mat(params: np.ndarray) -> np.ndarray:
    assert params.shape == (6,)
    r_vec = params[:3].reshape(3, 1)
    t_vec = params[3:].reshape(3, 1)
    return rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)


def _to_param_vector(view_mats: List[np.ndarray], points_3d: List[np.ndarray]) -> np.ndarray:
    all_view_mat_parameters = np.concatenate(list(map(_view_mat_to_param_vector, view_mats)))
    all_points_parameters = np.concatenate(list(map(lambda point: point.flatten(),  points_3d)))
    return np.concatenate((all_view_mat_parameters, all_points_parameters))


def _from_param_vector(n_cameras: int, n_points: int, params: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    assert len(params) == n_cameras * 6 + n_points * 3

    all_view_mat_parameters = params[:n_cameras * 6]
    view_mats = list(map(_param_vector_to_view_mat, all_view_mat_parameters.reshape(-1, 6)))
    assert len(view_mats) == n_cameras

    all_points_parameters = params[n_cameras * 6:]
    assert len(all_points_parameters) == n_points * 3
    points_3d = all_points_parameters.reshape(-1, 3)
    assert len(points_3d) == n_points

    return view_mats, points_3d


def _calculate_frame_residuals(
        view_mat: np.ndarray,
        points_3d: np.ndarray,
        intrinsic_mat: np.ndarray,
        frame_corners: FrameCorners,
        points_3d_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    _, (idx_2d, idx_3d) = snp.intersect(frame_corners.ids.flatten(), points_3d_ids, indices=True)

    assert view_mat.shape == (3, 4)
    proj_mat = intrinsic_mat @ view_mat

    frame_residuals = compute_reprojection_errors(points_3d[idx_3d], frame_corners.points[idx_2d], proj_mat).flatten()
    return frame_residuals, idx_2d, idx_3d

def _calculate_residuals_from_params(
        params: np.ndarray,
        intrinsic_mat: np.ndarray,
        list_of_corners: List[FrameCorners],
        points_3d_ids: np.ndarray,
        n_cameras: int,
        n_points: int,
        max_inlier_reprojection_error: float) -> np.ndarray:

    view_mats, points_3d = _from_param_vector(n_cameras, n_points, params)
    assert len(view_mats) == len(list_of_corners)
    points_3d = np.vstack(points_3d)
    assert len(points_3d) == len(points_3d_ids)

    all_residuals = []
    num_of_residuals = 0
    for view_mat, frame_corners in zip(view_mats, list_of_corners):
        frame_residuals, _, _ = _calculate_frame_residuals(
            view_mat,
            points_3d,
            intrinsic_mat,
            frame_corners,
            points_3d_ids)
        frame_residuals = np.minimum(frame_residuals, max_inlier_reprojection_error)
        num_of_residuals += len(frame_residuals)
        all_residuals.append(frame_residuals)

    all_residuals = np.concatenate(all_residuals)
    assert all_residuals.shape == (num_of_residuals,)

    return all_residuals


def _bundle_adjustment_sparsity(
        list_of_corners: List[FrameCorners],
        points_3d_ids: np.ndarray,
        n_cameras: int,
        n_points: int) -> lil_matrix:

    row_count = 0
    for frame, frame_corners in enumerate(list_of_corners):
        _, (idx_2d, idx_3d) = snp.intersect(frame_corners.ids.flatten(), points_3d_ids, indices=True)
        row_count += len(idx_3d)
    col_count = n_cameras * 6 + n_points * 3

    matrix = lil_matrix((row_count, col_count), dtype=int)

    print("Jacobian shape: {}x{}, nonzero values: {}".format(row_count, col_count, row_count * 9))

    cur_row = 0
    for frame, frame_corners in enumerate(list_of_corners):
        _, (idx_2d, idx_3d) = snp.intersect(frame_corners.ids.flatten(), points_3d_ids, indices=True)

        # NB: This frame will produce exactly `len(idx_3d)` residuals.
        frame_residuals_ids = cur_row + np.arange(len(idx_3d))
        assert len(frame_residuals_ids) == len(idx_3d)

        frame_camera_param_ids = frame * 6 + np.arange(6)
        assert len(frame_camera_param_ids) == 6

        for residual_id in frame_residuals_ids:
            matrix[residual_id, frame_camera_param_ids] = 1

        for i, point_idx in enumerate(idx_3d):
            frame_point_residual_id = cur_row + i
            point_param_ids = (n_cameras * 6 + point_idx * 3) + np.arange(3)
            assert len(point_param_ids) == 3
            matrix[frame_point_residual_id, point_param_ids] = 1

        cur_row += len(idx_3d)

    for row in range(row_count):
        assert matrix[row].count_nonzero() == 9
        assert matrix[row, :n_cameras * 6].count_nonzero() == 6
        assert matrix[row, n_cameras * 6:].count_nonzero() == 3

    return matrix


def run_bundle_adjustment(
        intrinsic_mat: np.ndarray,
        list_of_corners: List[FrameCorners],
        max_inlier_reprojection_error: float,
        view_mats: List[np.ndarray],
        pc_builder: PointCloudBuilder) -> List[np.ndarray]:

    assert len(list_of_corners) == len(view_mats)
    n_cameras = len(view_mats)
    n_points = len(pc_builder.ids)

    n_corners_initial = sum(len(frame_corners.ids) for frame_corners in list_of_corners)
    list_of_corners = deepcopy(list_of_corners)

    n_corners_deleted = 0
    for frame, (view_mat, frame_corners) in enumerate(zip(view_mats, list_of_corners)):
        frame_residuals, idx_2d, _ = _calculate_frame_residuals(
            view_mat,
            pc_builder.points,
            intrinsic_mat,
            frame_corners,
            pc_builder.ids.flatten())

        assert len(idx_2d) == len(frame_residuals)
        outliers = set(frame_corners.ids[idx_2d[frame_residuals > max_inlier_reprojection_error]].flatten())

        list_of_corners[frame] = filter_frame_corners_on_id(frame_corners, lambda id: id not in outliers)
        n_corners_deleted += np.count_nonzero(frame_residuals > max_inlier_reprojection_error)

    n_corners_filtered = sum(len(frame_corners.ids) for frame_corners in list_of_corners)
    assert n_corners_initial - n_corners_filtered == n_corners_deleted

    print("Filtered out {} corners out of {}, {} left".format(n_corners_deleted, n_corners_initial, n_corners_filtered))

    jacobian_begin = time.time()

    jacobian_sparsity = _bundle_adjustment_sparsity(
        list_of_corners,
        pc_builder.ids.flatten(),
        n_cameras,
        n_points)

    jacobian_end = time.time()

    print("Evaluated jacobian sparsity in {0:.0f} seconds".format(jacobian_end - jacobian_begin))

    print("Running optimization...")

    ba_begin = time.time()

    # NB: PyTypeChecker seems to produce invalid result for the least_squares method
    # noinspection PyTypeChecker
    optimize_result = least_squares(
        _calculate_residuals_from_params,
        _to_param_vector(view_mats, list(pc_builder.points)),
        jac_sparsity=jacobian_sparsity,
        verbose=2,  # Displays progress during iterations.
        x_scale='jac',  #
        ftol=1e-4,
        max_nfev=400,
        method='trf',  #
        loss='soft_l1',
        kwargs={
            "intrinsic_mat": intrinsic_mat,
            "list_of_corners": list_of_corners,
            "n_cameras": n_cameras,
            "n_points": n_points,
            "points_3d_ids": pc_builder.ids.flatten(),
            "max_inlier_reprojection_error": max_inlier_reprojection_error,
        })  # type: OptimizeResult

    ba_end = time.time()

    print("Optimization took {0:.0f} seconds".format(ba_end - ba_begin))

    new_view_mats, new_points_3d = _from_param_vector(n_cameras, n_points, optimize_result.x)
    pc_builder.update_points(pc_builder.ids, np.vstack(new_points_3d))
    return new_view_mats

