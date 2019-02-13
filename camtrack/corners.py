#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _float_img_to_u8(img):
    return np.uint8(np.round(img * 255))


def _get_width(img):
    return img.shape[1]


def _get_height(img):
    return img.shape[0]


def _get_pixel(img, point):
    point = point.reshape(2)
    x = min(int(round(point[0])), _get_width(img) - 1)
    y = min(int(round(point[1])), _get_height(img) - 1)
    return img[y][x]


def _mask_points_neighborhoods(img, points, radius):
    mask = np.ones_like(img).astype(np.uint8)
    for point in points.reshape(-1, 2):
        cv2.circle(mask, tuple(point), radius, color=0, thickness=-1)

    return mask


def _quality_filter(img, points, new_points, block_size, quality_level):
    quality_matrix = cv2.cornerMinEigenVal(img, block_size, ksize=block_size)

    def get_quality(point):
        return _get_pixel(quality_matrix, point)

    if points is None or len(points) == 0:
        return new_points == new_points

    best_quality = max(get_quality(point) for point in points.reshape(-1, 2))
    quality_threshold = best_quality * quality_level

    return np.array([get_quality(point) >= quality_threshold for point in new_points])


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    assert frame_sequence is not None and len(frame_sequence) > 0

    max_corners = 10000
    quality_level = 0.01
    next_id = 0
    block_size = 5
    min_distance = 10
    corner_size = 10

    # params for ShiTomasi corner detection
    feature_params = {
        "qualityLevel": quality_level,
        "minDistance": min_distance,
        "blockSize": block_size,
        "gradientSize": block_size,
        "useHarrisDetector": False,
    }

    # Parameters for lucas kanade optical flow
    lk_params = {
        "winSize": (15, 15),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }

    points = cv2.goodFeaturesToTrack(frame_sequence[0],
                                     maxCorners=max_corners,
                                     mask=None,
                                     **feature_params)
    assert points is not None and len(points) > 0
    sizes = np.asarray([corner_size for _ in range(len(points))])
    ids = np.asarray([next_id + i for i in range(len(points))])
    next_id += len(ids)

    builder.set_corners_at_frame(0, FrameCorners(ids, points, sizes))
    previous_frame = frame_sequence[0]

    for frame_idx, current_frame in enumerate(frame_sequence[1:], 1):
        current_frame = current_frame

        # calculate optical flow
        points, status, err = cv2.calcOpticalFlowPyrLK(
            _float_img_to_u8(previous_frame),
            _float_img_to_u8(current_frame),
            prevPts=points,
            nextPts=None,
            **lk_params)

        # filter out lost corners
        found = (status.reshape(-1) == 1)
        assert len(found) == len(points) == len(ids)
        points = points[found]
        sizes = sizes[found]
        ids = ids[found]
        err = err[found]

        # filter out corners with high error
        err = err.reshape(-1)
        assert len(err) == len(points)
        error_threshold = 5 * np.quantile(err, q=0.8)
        error_filter = err <= error_threshold
        points = points[error_filter]
        sizes = sizes[error_filter]
        ids = ids[error_filter]

        # filter out too low-quality corners
        quality_filter = _quality_filter(current_frame, points, points, block_size, quality_level / 100)
        points = points[quality_filter]
        sizes = sizes[quality_filter]
        ids = ids[quality_filter]

        if len(points) < max_corners:
            new_points = cv2.goodFeaturesToTrack(
                current_frame, maxCorners=max_corners, mask=None, **feature_params)

            if new_points is not None and len(new_points) > 0:
                mask=_mask_points_neighborhoods(current_frame, points, min_distance)
                mask_filter = np.array([_get_pixel(mask, point) != 0 for point in new_points])
                new_points = new_points[mask_filter]

            # don't add too much
            new_points = new_points[:max_corners-len(points)]

            # add new corners to the list
            if new_points is not None and len(new_points) > 0:
                new_sizes = np.array([corner_size for _ in range(len(new_points))])
                new_ids = np.array([next_id + i for i in range(len(new_points))])
                next_id += len(new_ids)

                points = np.append(points, new_points, axis=0)
                sizes = np.append(sizes, new_sizes, axis=0)
                ids = np.append(ids, new_ids, axis=0)

        previous_frame = current_frame
        builder.set_corners_at_frame(frame_idx, FrameCorners(ids, points, sizes))


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
