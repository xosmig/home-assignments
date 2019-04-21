#! /usr/bin/env python3
from numpy.core._multiarray_umath import ndarray

__all__ = [
    'track_and_calc_colors'
]

import sys
from collections import namedtuple
from typing import List, Tuple, Set, Callable, Optional

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
    frame1_candidates = [0]  # TODO
    frame2_candidates = list(range(1, len(corner_storage)))

    frame1, frame2, (points_3d, ids, pose, quality) = max(
        [(frame1, frame2, _track_2_frames(corner_storage[frame1], corner_storage[frame2], intrinsic_mat))
         for frame1 in frame1_candidates
         for frame2 in frame2_candidates],
        key=lambda x: x[-1][-1]
    )

    # TODO: Warning if the quality is too low

    return _InitializationResult(frame1, frame2, points_3d, ids, pose, quality)


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


def _solve_pnp(
        frame_corners: FrameCorners,
        prev_view_matrix: np.ndarray,
        intrinsic_mat: np.ndarray,
        points_3d: PointCloudBuilder) -> Tuple[np.ndarray, Set[int]]:

    _, (idx_2d, idx_3d) = snp.intersect(
        frame_corners.ids.flatten(),
        points_3d.ids.flatten(),
        indices=True)

    pnp_frame_corners = _select_corners_by_indices(frame_corners, idx_2d)
    pnp_3d_points = points_3d.points[idx_3d]

    if len(pnp_frame_corners.ids) < 6:
        # # TODO: handle this case
        # print("FOOBAR: finished on frame {}".format(frame), file=sys.stderr)
        # assert frame > 0
        # for f in range(frame, len(corner_storage)):
        #     view_matrices[f] = view_matrices[frame - 1]
        # break
        raise ValueError()  # FIXME

    prev_r_vector = _rotation_matrix_to_vector(prev_view_matrix[:, :3])
    prev_t_vector = prev_view_matrix[:, 3]

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

    frame_view_matrix = _components_to_view_mat3x4(_rotation_vector_to_matrix(r_vector), t_vector)
    new_outliers = np.delete(pnp_frame_corners.ids.flatten(), inliers.flatten())

    return frame_view_matrix, set(new_outliers)


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:

    init_res = _initialize_tracking(corner_storage, intrinsic_mat)
    assert init_res.frame1 == 0
    print("Initialized using frames {} and {}".format(init_res.frame1, init_res.frame2), file=sys.stderr)
    print("Number of 3d points: {}".format(len(init_res.points_3d)), file=sys.stderr)

    view_matrices = list(None for _ in range(len(corner_storage)))
    view_matrices[init_res.frame1] = eye3x4()
    view_matrices[init_res.frame2] = pose_to_view_mat3x4(init_res.frame2_pose)

    point_cloud_builder = PointCloudBuilder()
    point_cloud_builder.add_points(init_res.point_ids, init_res.points_3d)

    outliers = set()

    def _remove_outliers(corners):
        return _filter_corners_on_id(corners, lambda id: id not in outliers)

    for frame in range(len(corner_storage)):
        if view_matrices[frame] is not None:
            continue

        print(("Processing frame {frame_number} / {total_frames} ("
               "PointCloudSize: {point_cloud_size}, "
               "OutliersCount: {outliers_count})")
              .format(frame_number=frame,
                      total_frames=len(corner_storage),
                      point_cloud_size=len(point_cloud_builder.ids),
                      outliers_count=len(outliers)),
              file=sys.stderr)

        assert frame > 0
        assert view_matrices[frame - 1] is not None
        frame_view_matrix, new_outliers = _solve_pnp(
            _remove_outliers(corner_storage[frame]),
            view_matrices[frame - 1],
            intrinsic_mat,
            point_cloud_builder)
        assert frame_view_matrix is not None  # TODO: handle failure

        view_matrices[frame] = frame_view_matrix
        outliers.update(new_outliers)

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
