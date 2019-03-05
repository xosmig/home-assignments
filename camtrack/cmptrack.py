#! /usr/bin/env python3

__all__ = [
    'start_from_origin',
    'get_all_translations',
    'get_all_rotation_mats',
    'calc_track_length',
    'calc_translation_errors',
    'calc_rotation_error_rad',
    'calc_rotation_errors_rad',
    'calc_errors'
]

from typing import List, Tuple

import click
import numpy as np
from transforms3d.axangles import mat2axangle

from data3d import Pose, read_poses


def _to_mat4x4(pose):
    return np.vstack((np.hstack((pose.r_mat, pose.t_vec.reshape(-1, 1))),
                      np.array([0, 0, 0, 1.0])))


def _to_pose_from_mat4x4(mat):
    return Pose(mat[:3, :3], mat[:3, 3].flatten())


def start_from_origin(poses: List[Pose]) -> List[Pose]:
    mat_0_inv = np.linalg.inv(_to_mat4x4(poses[0]))
    return [_to_pose_from_mat4x4(mat_0_inv @ _to_mat4x4(p)) for p in poses]


def get_all_translations(poses: List[Pose]) -> np.ndarray:
    return np.array([p.t_vec for p in poses])


def get_all_rotation_mats(poses: List[Pose]) -> np.ndarray:
    return np.array([p.r_mat for p in poses])


def calc_track_length(t_vecs: np.ndarray) -> float:
    diffs = t_vecs[1:, :] - t_vecs[:-1, :]
    return np.linalg.norm(diffs, axis=1).sum()


def calc_translation_errors(ground_truth_t_vecs: np.ndarray,
                            estimate_t_vecs: np.ndarray) -> np.ndarray:
    scale, _, _, _ = np.linalg.lstsq(
        estimate_t_vecs.reshape((-1, 1)),
        ground_truth_t_vecs.flatten(),
        rcond=None
    )
    scale = scale.item()
    scaled_estimate_t_vecs = scale * estimate_t_vecs
    ground_truth_track_length = calc_track_length(ground_truth_t_vecs)
    return np.linalg.norm(ground_truth_t_vecs - scaled_estimate_t_vecs,
                          axis=1) / ground_truth_track_length


def calc_rotation_error_rad(r_mat_1: np.ndarray, r_mat_2: np.ndarray) -> float:
    r_mat_diff = r_mat_2 @ r_mat_1.T
    _, angle = mat2axangle(r_mat_diff)
    return np.abs(angle)


def calc_rotation_errors_rad(r_mats_1: np.ndarray,
                             r_mats_2: np.ndarray) -> np.ndarray:
    return np.array([calc_rotation_error_rad(r_mat_1, r_mat_2)
                     for r_mat_1, r_mat_2 in zip(r_mats_1, r_mats_2)])


def calc_errors(ground_truth_track: List[Pose],
                estimate_track: List[Pose]) -> Tuple[np.ndarray, np.ndarray]:
    ground_truth_track = start_from_origin(ground_truth_track)
    estimate_track = start_from_origin(estimate_track)
    r_errors = calc_rotation_errors_rad(
        get_all_rotation_mats(ground_truth_track),
        get_all_rotation_mats(estimate_track),
    )
    t_errors = calc_translation_errors(
        get_all_translations(ground_truth_track),
        get_all_translations(estimate_track),
    )
    return r_errors, t_errors


# pylint:disable=no-value-for-parameter
@click.command()
@click.argument('ground_truth_file', type=click.File('r'))
@click.argument('estimate_file', type=click.File('r'))
@click.option('--plot', '-p', is_flag=True, help='Plot frame errors')
def _cli(ground_truth_file, estimate_file, plot):
    gt_track = read_poses(ground_truth_file)
    e_track = read_poses(estimate_file)
    r_errors, t_errors = calc_errors(gt_track, e_track)
    r_errors = np.degrees(r_errors)  # pylint:disable=assignment-from-no-return

    r_max = r_errors.max()
    r_median = np.median(r_errors)
    t_max = t_errors.max()
    t_median = np.median(t_errors)
    click.echo('Rotation errors (degrees)\n  max = {}\n  median = {}'.format(
        r_max,
        r_median
    ))
    click.echo('Translation errors\n  max = {}\n  median = {}'.format(
        t_max,
        t_median
    ))

    if not plot:
        return

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(211)
    plt.plot(r_errors)
    plt.stem(r_errors, linefmt='g:')
    plt.xlabel('frame')
    plt.ylabel('Rotation error (degrees)')
    plt.subplot(212)
    plt.plot(t_errors)
    plt.stem(t_errors, linefmt='g:')
    plt.xlabel('frame')
    plt.ylabel('Translation error')
    plt.show()


if __name__ == '__main__':
    _cli()
