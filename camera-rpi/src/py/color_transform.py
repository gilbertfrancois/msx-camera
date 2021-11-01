import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def rgbi2hsvi(image: np.ndarray) -> np.ndarray:
    """ RGB to HSV in (theta, r, z) coordinates.

    Parameters
    ----------
    image: np.ndarray
        Input image as np.uint8

    Returns
    -------
    np.ndarray
        hue in [0, 180], sat in [0, 255], value in [0, 255]
    """
    if image.ndim != 3:
        raise RuntimeError(f"Wrong input dimension. Expected 3, actual {image.ndim}")
    return cv.cvtColor(image, cv.COLOR_RGB2HSV).astype(np.uint8)


def rgbi2hsvf(image: np.ndarray) -> np.ndarray:
    """ RGB to HSV in (theta, r, z) coordinates.

    Parameters
    ----------
    image: np.ndarray
        Input image as np.uint8

    Returns
    -------
    np.array
        hue in [0, 2Pi], sat in [0,1], value in range [0,1]
    """
    dst = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    dst = dst.astype(np.float64)
    dst = dst * np.array([[2 * np.pi / 180, 1/255, 1/255]])
    return dst


def rgbi2hsvf_xy(image: np.ndarray) -> np.ndarray:
    """ RGB to HSV in (x, y, z) coordinates.

    Parameters
    ----------
    image: np.ndarray
        Input image as np.uint8

    Returns
    -------
    np.array
        [sat * cos(hue), sat * sin(hue), value] as float in [0, 1]
    """
    dst = rgbi2hsvf(image)
    x = dst[:, :, 1] * np.cos(dst[:, :, 0])
    y = dst[:, :, 1] * np.sin(dst[:, :, 0])
    dst[:, :, 0] = x
    dst[:, :, 1] = y
    return dst


def l2_dist(v, A):
    """ Euclidian distance function.

    Parameters
    ----------
    v: np.ndarray
        Input vector
    A: np.ndarray
        Comparison matrix

    Returns
    -------
    (np.ndarray, np.ndarray)
        sorted index, sorted distance
    """
    dist = np.sum(np.square(np.subtract(v, A)), axis=1)
    idx = np.argsort(dist)
    return idx, dist[idx]
    
