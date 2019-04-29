#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import sys
from collections import namedtuple
from typing import List, Tuple, Set, Optional

import numpy as np
import sortednp as snp

import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *
from _corners import filter_frame_corners, filter_frame_corners_on_id

from ba import run_bundle_adjustment

############################################
# Parameters
_PROB = 0.99999
_TRIANGULATION_PARAMETERS = TriangulationParameters(
    max_reprojection_error=1.0,
    min_triangulation_angle_deg=5,
    min_depth=0.5,
)
############################################

_InitializationResult = namedtuple(
    '_InitializationResult',
    ('frame1', 'frame2', 'points_3d', 'point_ids', 'frame2_pose', 'quality')
)


def _track_2_frames(corners_1: FrameCorners, corners_2: FrameCorners, intrinsic_mat: np.ndarray):
    correspondences = build_correspondences(corners_1, corners_2)

    essential_matrix, inliers_mask = cv2.findEssentialMat(
        correspondences.points_1,
        correspondences.points_2,
        intrinsic_mat,
        method=cv2.RANSAC,
        prob=_PROB,
        threshold=_TRIANGULATION_PARAMETERS.max_reprojection_error,
    )

    correspondences = apply_correspondences_mask(correspondences, inliers_mask)

    _, homography_inliers_mask = cv2.findHomography(
        correspondences.points_1,
        correspondences.points_2,
        method=cv2.RANSAC)

    if np.mean(homography_inliers_mask) >= 0.6:
        return None

    r_candidate_1, r_candidate_2, t_candidate = cv2.decomposeEssentialMat(essential_matrix)
    pose_candidates = [view_mat3x4_to_pose(np.hstack((view_r_mat, view_t_vec)))
                       for view_r_mat in (r_candidate_1, r_candidate_2)
                       for view_t_vec in (t_candidate, -t_candidate)]

    def try_pose(pose):
        triangulation_result = triangulate_correspondences(
            correspondences,
            view_mat_1=eye3x4(),
            view_mat_2=pose_to_view_mat3x4(pose),
            intrinsic_mat=intrinsic_mat,
            parameters=_TRIANGULATION_PARAMETERS)
        return triangulation_result, pose, len(triangulation_result[0])

    (points_3d, ids), pose, _ = max([try_pose(pose) for pose in pose_candidates], key=lambda x: x[-1])

    # TODO: check quality
    quality = len(points_3d)  # TODO

    return points_3d, ids, pose, quality


def _initialize_tracking(corner_storage: CornerStorage, intrinsic_mat: np.ndarray) -> Optional[_InitializationResult]:
    frame1_candidates = [0]  # TODO
    frame2_candidates = list(range(1, len(corner_storage)))

    best_result = max(
        [(frame1, frame2, _track_2_frames(corner_storage[frame1], corner_storage[frame2], intrinsic_mat))
         for frame1 in frame1_candidates
         for frame2 in frame2_candidates],
        key=lambda x: x[-1][-1] if x[-1] is not None else 0
    )

    if best_result is None:
        raise ValueError("No sufficient initialization pairs")

    frame1, frame2, (points_3d, ids, pose, quality) = best_result
    return _InitializationResult(frame1, frame2, points_3d, ids, pose, quality)


def _select_corners_by_indices(frame_corners: FrameCorners, indices: np.ndarray) -> FrameCorners:
    mask = np.repeat(False, len(frame_corners.ids))
    mask[indices] = True
    return filter_frame_corners(frame_corners, mask)


def _cv2_to_numpy(t: Tuple) -> Tuple:
    return tuple(val.get() if isinstance(val, cv2.UMat) else val for val in t)


def _solve_pnp(
        frame_corners: FrameCorners,
        prev_view_matrix: np.ndarray,
        intrinsic_mat: np.ndarray,
        points_3d: PointCloudBuilder) -> Optional[Tuple[np.ndarray, Set[int]]]:

    _, (idx_2d, idx_3d) = snp.intersect(
        frame_corners.ids.flatten(),
        points_3d.ids.flatten(),
        indices=True)

    pnp_frame_corners = _select_corners_by_indices(frame_corners, idx_2d)
    pnp_3d_points = points_3d.points[idx_3d]

    if len(pnp_frame_corners.ids) < 6:
        raise ValueError("Not enough corners")

    prev_r_vector, prev_t_vector = view_mat3x4_to_rodrigues_and_translation(prev_view_matrix)

    ransac_success, r_vector, t_vector, inliers = _cv2_to_numpy(cv2.solvePnPRansac(
        pnp_3d_points,
        pnp_frame_corners.points,
        intrinsic_mat,
        distCoeffs=None,  # TODO: what is it?
        rvec=prev_r_vector,
        tvec=prev_t_vector,
        useExtrinsicGuess=True,
        reprojectionError=_TRIANGULATION_PARAMETERS.max_reprojection_error,
    ))

    if not ransac_success:
        raise ValueError("Ransac failure")

    frame_view_matrix = rodrigues_and_translation_to_view_mat3x4(r_vector, t_vector)
    new_outliers = np.delete(pnp_frame_corners.ids.flatten(), inliers.flatten())

    return frame_view_matrix, set(new_outliers)


def _try_track_camera(
        corner_storage: CornerStorage,
        intrinsic_mat: np.ndarray) -> Tuple[List[np.ndarray], PointCloudBuilder]:

    init_res = _initialize_tracking(corner_storage, intrinsic_mat)

    assert init_res.frame1 == 0
    print("Initialized using frames {} and {}".format(init_res.frame1, init_res.frame2), file=sys.stderr)
    print("Number of 3d points: {}".format(len(init_res.points_3d)), file=sys.stderr)

    view_matrices = [None] * len(corner_storage)  # type: List[Optional[np.ndarray]]
    view_matrices[init_res.frame1] = eye3x4()
    view_matrices[init_res.frame2] = pose_to_view_mat3x4(init_res.frame2_pose)

    point_cloud_builder = PointCloudBuilder()
    point_cloud_builder.add_points(init_res.point_ids, init_res.points_3d)

    outliers = set()

    def _remove_outliers(corners: FrameCorners) -> FrameCorners:
        return filter_frame_corners_on_id(corners, lambda id: id not in outliers)

    list_of_inliers = [None] * len(corner_storage)  # type: List[Optional[FrameCorners]]

    for frame in range(len(corner_storage)):
        if view_matrices[frame] is not None:
            list_of_inliers[frame] = _remove_outliers(corner_storage[frame])
            continue

        print(("Processing frame {frame_number} / {total_frames} ("
               "PointCloudSize: {point_cloud_size}, "
               "OutliersCount: {outliers_count})")
              .format(frame_number=frame + 1,
                      total_frames=len(corner_storage),
                      point_cloud_size=len(point_cloud_builder.ids),
                      outliers_count=len(outliers)))

        assert frame > 0
        assert view_matrices[frame - 1] is not None
        frame_view_matrix, new_outliers = _solve_pnp(
            _remove_outliers(corner_storage[frame]),
            view_matrices[frame - 1],
            intrinsic_mat,
            point_cloud_builder)

        view_matrices[frame] = frame_view_matrix
        outliers.update(new_outliers)

        list_of_inliers[frame] = _remove_outliers(corner_storage[frame])

        # if frame > 10 and np.random.random() < 0.15:
        #     ba_start = max(0, frame - 20)
        #     ba_end = frame + 1
        #     view_matrices[ba_start:ba_end] = run_bundle_adjustment(
        #         intrinsic_mat,
        #         list_of_inliers[ba_start:ba_end],
        #         max_inlier_reprojection_error=5,
        #         view_mats=view_matrices[ba_start:ba_end],
        #         pc_builder=point_cloud_builder,
        #         verbose=1)

        for past_frame in range(frame):
            correspondences = build_correspondences(
                _remove_outliers(corner_storage[past_frame]),
                _remove_outliers(corner_storage[frame]),
                ids_to_remove=point_cloud_builder.ids)

            assert view_matrices[past_frame].shape == (3, 4), "past_frame == {}".format(past_frame)
            assert view_matrices[frame].shape == (3, 4), "frame == {}".format(frame)
            if len(correspondences.ids) == 0:
                continue

            new_points_3d, new_ids = triangulate_correspondences(
                correspondences,
                view_mat_1=view_matrices[past_frame],
                view_mat_2=view_matrices[frame],
                intrinsic_mat=intrinsic_mat,
                parameters=_TRIANGULATION_PARAMETERS)

            point_cloud_builder.add_points(new_ids, new_points_3d)

    view_matrices = run_bundle_adjustment(
        intrinsic_mat,
        list_of_inliers,
        max_inlier_reprojection_error=_TRIANGULATION_PARAMETERS.max_reprojection_error,
        view_mats=view_matrices,
        pc_builder=point_cloud_builder,
        verbose=2,
        # Experimental parameters:
        # x_scale="custom",
        # method="dogbox",
        # loss="cauchy",
        # enable_bounds=True,
        bound_error=True,
        filter_corners=True,
        # max_nfev=400,
    )
    return view_matrices, point_cloud_builder


def _track_camera(
        corner_storage: CornerStorage,
        intrinsic_mat: np.ndarray,
        max_retries: int=4) -> Tuple[List[np.ndarray], PointCloudBuilder]:

    def _weaken_parameters():
        global _PROB
        _PROB /= 1.5
        global _TRIANGULATION_PARAMETERS
        _TRIANGULATION_PARAMETERS = TriangulationParameters(
            max_reprojection_error=_TRIANGULATION_PARAMETERS.max_reprojection_error * 1.5,
            min_triangulation_angle_deg=_TRIANGULATION_PARAMETERS.min_triangulation_angle_deg,  # / 1.1,
            min_depth=_TRIANGULATION_PARAMETERS.min_depth,  # / 1.1,
        )

    res = None
    for retry_id in range(max_retries+1):
        try:
            res = _try_track_camera(corner_storage, intrinsic_mat)
        except ValueError as ex:
            if retry_id < max_retries:
                print("Failed")
                print("Error:", ex)
                print("Retrying {} / {}".format(retry_id+1, max_retries))
                _weaken_parameters()
                continue
            else:
                raise
        break
    return res


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat)
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
