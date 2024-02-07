"""
Module for Utility Functions

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: Dec 18, 2023
"""
__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import cv2
import numpy as np


def cv_imread(file_path):
    """读取带中文路径的图片文件
    Args:
        file_path (_type_): _description_
    Returns:
        _type_: _description_
    """
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)


def cv_imwrite(file_path, img):
    """保存带中文路径的图片文件
    Args:
        file_path (_type_): _description_
        img (_type_): _description_
    """
    cv2.imencode(".png", img)[1].tofile(file_path)


def merge_two_contours(contour1, contour2):
    """
    Merge two contours into one, from the nearest points.

    Args:
        contour1: First contour.
        contour2: Second contour.

    Returns:
        Merged contour.
    """
    (
        _,
        contour1_nearest_point_idx,
        contour2_nearest_point_idx,
    ) = get_distance_between_two_contours(contour1, contour2)

    # reorganize contour1's contour points from the nearest point
    reorganized_contour1 = np.roll(contour1, -contour1_nearest_point_idx, axis=0)
    return np.insert(contour2, contour2_nearest_point_idx, reorganized_contour1, axis=0)


def merge_two_contours_by_npi(
    contour1, contour2, contour1_nearest_point_idx, contour2_nearest_point_idx
):
    """
    Merge two contours into one, by the nearest points index.
    "npi" means "nearest point index" of a contour.

    Args:
        contour1: First contour.
        contour2: Second contour.
        contour1_nearest_point_idx: Nearest point index of contour1.
        contour2_nearest_point_idx: Nearest point index of contour2.

    Returns:
        Merged contour.
    """
    # reorganize contour1's contour points from the nearest point
    reorganized_contour1 = np.roll(contour1, -contour1_nearest_point_idx, axis=0)
    return np.insert(contour2, contour2_nearest_point_idx, reorganized_contour1, axis=0)


def contour_closest_to_which_contour(given_contour, contours):
    """
    Find the contour in contours that is closest to the given contour.

    Args:
        contour: Given contour.
        contours: Contours to search from.

    Returns:
        Index of the closest contour in contours.
        Distance to the closest contour.
        Nearest point index of the given contour.
        Nearest point index of the closest contour.
    """
    min_distance = float("inf")
    closest_contour_idx = 0
    nearest_point_idx_on_given_contour = 0
    nearest_point_idx_on_closest_contour = 0
    for idx, cnt in enumerate(contours):
        distance, src_idx, dst_idx = get_distance_between_two_contours(
            given_contour, cnt
        )
        if distance < min_distance:
            min_distance = distance
            closest_contour_idx = idx
            nearest_point_idx_on_given_contour = src_idx
            nearest_point_idx_on_closest_contour = dst_idx

    return (
        closest_contour_idx,
        min_distance,
        nearest_point_idx_on_given_contour,
        nearest_point_idx_on_closest_contour,
    )


def get_distance_between_two_contours(contour1, contour2):
    """
    Get the distance between two contours.

    Args:
        contour1: First contour.
        contour2: Second contour.

    Returns:
        Distance between two contours.
        nearest point index of contour1.
        nearest point index of contour2.
    """
    contour1_points = contour1[:, 0, :]
    contour2_points = contour2[:, 0, :]

    min_distance = float("inf")
    contour1_nearest_point_idx = 0
    contour2_nearest_point_idx = 0

    for idx1, point1 in enumerate(contour1_points):
        for idx2, point2 in enumerate(contour2_points):
            distance = np.linalg.norm(point1 - point2)
            if distance < min_distance:
                min_distance = distance
                contour1_nearest_point_idx = idx1
                contour2_nearest_point_idx = idx2

    return min_distance, contour1_nearest_point_idx, contour2_nearest_point_idx


def sift_similarity_on_roi(roi1, roi2):
    """_summary_

    Args:
        roi1 (_type_): Region of interest 1.
        roi2 (_type_): Region of interest 2.

    Returns:
        float: similarity score in %
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for ROIs
    keypoints1, descriptors1 = sift.detectAndCompute(roi1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(roi2, None)

    # Match descriptors between ROIs
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = [[m] for m, n in matches if m.distance < 0.75 * n.distance]
    # Calculate similarity score based on the number of good matches
    similarity = len(good_matches) / max(len(keypoints1), len(keypoints2)) * 100
    print(f"Similarity: {similarity:.2f}%")
    return similarity


def contour_bbox_img(img, contour):
    """
    Get the bounding box img of a contour.

    Args:
        img: Image.
        contour: Contour.

    Returns:
        cropped: Bounding box image, with grayscale and black background.
        target_on_white: Bounding box image, with white background, not grayscale.
    """
    # if not grayscale, convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Create a mask for the contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Extract the target
    extracted = cv2.bitwise_and(gray, gray, mask=mask)

    # get bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the object with some padding, adjust padding as needed
    padding = 1
    cropped = extracted[y - padding : y + h + padding, x - padding : x + w + padding]

    # make it white background
    wbg_cropped = np.full_like(cropped, 255, dtype=np.uint8)
    np.copyto(wbg_cropped, cropped, where=(cropped > 0))

    return cropped, wbg_cropped



