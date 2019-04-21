#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np
import sortednp as snp

import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *

############################################
# Parameters
_TRIANGULATION_PARAMETERS = TriangulationParameters(
    max_reprojection_error=5,  # TODO
    min_triangulation_angle_deg=5,  # TODO
    min_depth=0.5  # TODO
)
############################################


def _track_2_frames(corners_1: FrameCorners, corners_2: FrameCorners, intrinsic_mat: np.ndarray):
    correspondences = build_correspondences(corners_1, corners_2)

    essential_matrix, inliers_mask = cv2.findEssentialMat(
        correspondences.points_1,
        correspondences.points_2,
        intrinsic_mat,
        method=cv2.RANSAC,
        # prob=None,  # TODO
        # threshold=None  # TODO
    )

    correspondences = apply_correspondences_mask(correspondences, inliers_mask)

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


def _initialize_tracking(corner_storage: CornerStorage, intrinsic_mat: np.ndarray):
    frame_1_candidates = [0]  # TODO
    frame_2_candidates = list(range(1, len(corner_storage)))

    frame_1, frame_2, (points_3d, ids, pose, quality) = max(
        [(frame_1, frame_2, _track_2_frames(corner_storage[frame_1], corner_storage[frame_2], intrinsic_mat))
         for frame_1 in frame_1_candidates
         for frame_2 in frame_2_candidates],
        key=lambda x: x[-1][-1]
    )
    print(len(points_3d))

    # TODO: Warning if the quality is too low

    return frame_1, frame_2, points_3d, ids, pose, quality


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    frame_1, frame_2, points_3d, ids, pose, quality = _initialize_tracking(corner_storage, intrinsic_mat)
    assert frame_1 == 0

    view_matrices = [eye3x4()] * frame_2 + [pose_to_view_mat3x4(pose)] * (len(corner_storage) - frame_2)

    point_cloud_builder = PointCloudBuilder()
    point_cloud_builder.add_points(ids, points_3d)

    return view_matrices, point_cloud_builder


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
        intrinsic_mat
    )
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
