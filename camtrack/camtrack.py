#! /usr/bin/env python3
from numpy.core._multiarray_umath import ndarray

__all__ = [
    'track_and_calc_colors'
]

import sys
from collections import namedtuple
from typing import List, Tuple, Set, Callable

import numpy as np
import sortednp as snp

import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *
from _corners import filter_frame_corners

############################################
# Parameters
_TRIANGULATION_PARAMETERS = TriangulationParameters(
    max_reprojection_error=5,  # TODO
    min_triangulation_angle_deg=5,  # TODO
    min_depth=0.5  # TODO
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

    # TODO: validate with homography (whatever that means).

    # TODO: check quality
    quality = len(points_3d)  # TODO

    return points_3d, ids, pose, quality


def _initialize_tracking(corner_storage: CornerStorage, intrinsic_mat: np.ndarray) \
        -> _InitializationResult:
    frame_1_candidates = [0]  # TODO
    frame_2_candidates = list(range(1, len(corner_storage)))
    # frame_2_candidates = list(range(50, len(corner_storage)))

    frame_1, frame_2, (points_3d, ids, pose, quality) = max(
        [(frame_1, frame_2, _track_2_frames(corner_storage[frame_1], corner_storage[frame_2], intrinsic_mat))
         for frame_1 in frame_1_candidates
         for frame_2 in frame_2_candidates],
        key=lambda x: x[-1][-1]
    )
    print(len(points_3d))

    # TODO: Warning if the quality is too low

    return _InitializationResult(frame_1, frame_2, points_3d, ids, pose, quality)


def _filter_corners_on_id(frame_corners: FrameCorners, id_predicate: Callable[[int], bool]) -> FrameCorners:
    mask = np.vectorize(lambda id: id_predicate(id))(frame_corners.ids.flatten())
    assert len(mask) == len(frame_corners.ids)
    assert mask.dtype == np.bool
    return filter_frame_corners(frame_corners, mask)


def _select_corners_by_indices(frame_corners: FrameCorners, indices: np.ndarray) -> FrameCorners:
    mask = np.repeat(False, len(frame_corners.ids))
    mask[indices] = True
    return filter_frame_corners(frame_corners, mask)


def _rotation_matrix_to_vector(r_matrix: np.ndarray) -> np.ndarray:
    res, _ = cv2.Rodrigues(r_matrix)
    return res


def _rotation_vector_to_matrix(r_vector: np.ndarray) -> np.ndarray:
    res, _ = cv2.Rodrigues(r_vector)
    return res


def _components_to_view_mat3x4(r_matrix: np.ndarray, t_vector: np.ndarray) -> np.ndarray:
    assert r_matrix.shape == (3, 3)
    assert t_vector.shape == (3, 1)
    return np.hstack((r_matrix, t_vector))


def _cv2_to_numpy(t: Tuple) -> Tuple:
    return tuple(val.get() if isinstance(val, cv2.UMat) else val for val in t)


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    init_res = _initialize_tracking(corner_storage, intrinsic_mat)
    assert init_res.frame1 == 0
    print("Initialized using frames {} and {}".format(init_res.frame1, init_res.frame2), file=sys.stderr)

    view_matrices = list(None for _ in range(len(corner_storage)))
    view_matrices[init_res.frame1] = eye3x4()
    view_matrices[init_res.frame2] = pose_to_view_mat3x4(init_res.frame2_pose)

    point_cloud_builder = PointCloudBuilder()
    point_cloud_builder.add_points(init_res.point_ids, init_res.points_3d)

    outliers = set()

    for frame in range(len(corner_storage)):
        if view_matrices[frame] is not None:
            continue

        pnp_frame_corners = _filter_corners_on_id(corner_storage[frame], lambda id: id not in outliers)

        _, (idx_2d, idx_3d) = snp.intersect(
            pnp_frame_corners.ids.flatten(),
            point_cloud_builder.ids.flatten(),
            indices=True)

        pnp_frame_corners = _select_corners_by_indices(pnp_frame_corners, idx_2d)
        pnp_3d_points = point_cloud_builder.points[idx_3d]

        print(("Processing frame {frame_number} ("
               "PointCloudSize: {point_cloud_size}, "
               "OutliersCount: {outliers_count})")
              .format(frame_number=frame,
                      point_cloud_size=len(point_cloud_builder.ids),
                      outliers_count=len(outliers)),
              file=sys.stderr)

        if len(pnp_frame_corners.ids) < 6:
            # TODO: handle this case
            print("FOOBAR: finished on frame {}".format(frame), file=sys.stderr)
            assert frame > 0
            for f in range(frame, len(corner_storage)):
                view_matrices[f] = view_matrices[frame - 1]
            break

        assert frame > 0
        assert view_matrices[frame - 1] is not None
        prev_r_vector = _rotation_matrix_to_vector(view_matrices[frame - 1][:, :3])
        prev_t_vector = view_matrices[frame - 1][:, 3]

        ransac_success, r_vector, t_vector, inliers = _cv2_to_numpy(cv2.solvePnPRansac(
            pnp_3d_points,
            pnp_frame_corners.points,
            intrinsic_mat,
            distCoeffs=None,  # TODO: wth is it?
            rvec=prev_r_vector,
            tvec=prev_t_vector,
            useExtrinsicGuess=True,
            # reprojectionError=, # TODO
        ))

        # TODO: handle failure
        assert ransac_success

        view_matrices[frame] = _components_to_view_mat3x4(_rotation_vector_to_matrix(r_vector), t_vector)
        new_outliers = np.delete(pnp_frame_corners.ids.flatten(), inliers.flatten())

        print("FOOBAR2: number of points: {}, number of inliers: {}, number of outliers: {}"
              .format(len(pnp_frame_corners.ids), len(inliers), len(new_outliers)), file=sys.stderr)

        # TODO: check the number of new outliers?
        outliers.update(list(new_outliers))

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
