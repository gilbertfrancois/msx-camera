import numpy as np
import cv2 as cv

class Adjustment:

    @staticmethod
    def contrast_linear(src:np.ndarray, value:int) -> np.ndarray:
        """ Linear contrast enhancer.

        Parameters
        ----------
        src: np.ndarray
            Input image as type uint8
        value: int
            Amount of contrast applied, value between 0 and 128

        Returns
        -------
        np.ndarray
            Output image as type uint8
        """
        f = 131 * (value + 127) / (127 * (131 - value))
        alpha_c = f
        gamma_c = 127*(1-f)
        dst = cv.addWeighted(src, alpha_c, src, 0, gamma_c)
        dst = np.clip(dst, 0, 255)
        return dst

    @staticmethod
    def contrast_scurve(src:np.ndarray, value:int):
        """ Enhance contrast using an s-curve.

        Parameters
        ----------
        src: np.ndarray
            Input image
        value: int
            Amount of contrast [0, 100]
        
        Returns
        -------
        np.ndarray
            Output image as type uint8
        """
        if not isinstance(value, int):
            raise TypeError(f"Expected: int, actual: {type(value)}.")
        if value < 0 or value > 100:
            raise ValueError(f"Contrast value should be 0 ≤ v ≤ 100. Actual: {value}")
        N = 5
        a = 1 + N*(value / 100)
        dst = src.astype(np.float32)
        if np.max(dst) > 2:
            dst = dst / 255.0
        lidx = dst < 0.5
        uidx = dst >= 0.5
        dst[lidx] = 0.5 * np.power(dst[lidx] / 0.5, a)
        dst[uidx] = 1.0 - (0.5*np.power((1.0 - dst[uidx])/0.5, a))
        dst = np.clip(dst, 0.0, 1.0)
        dst = (dst * 255).astype(np.uint8)
        return dst

    @staticmethod
    def brightness(src:np.ndarray, value:int):
        if not isinstance(value, int):
            raise TypeError(f"Expected: int, actual: {type(value)}.")
        if value < -100 or value > 100:
            raise ValueError(f"Brightness value should be -100 ≤ v ≤ 100. Actual: {value}")
        if value > 0:
            shadow = value
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + value
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        dst = cv.addWeighted(src, alpha_b, src, 0, gamma_b)
        return dst
