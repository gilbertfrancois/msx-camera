import cv2 as cv
import numpy as np
from tms99x8 import TMS99x8
from adjustment import Adjustment
from dither import Dither

class TMS99x8_BW_Dither(TMS99x8):

    def __init__(self, **kwargs):
        """ Constructor

        Parameters
        ----------
        background_color: int (optional)
            Background color, numbered from the MSX screen 2 palette. Default = 14 (gray).

        """
        super().__init__(**kwargs)
        self.background_color = kwargs.get("background_color", 14)
        self.set_background_color(self.background_color)

    def set_background_color(self, background_color):
        self.background_color = background_color
        self.background = np.ones(shape=TMS99x8.SCREEN2_SHAPE, dtype=(np.uint8))
        self.background = self.palette_rgbi[self.background_color, :] * self.background


    def render(self, src, **kwargs):
        """ Render function

        Parameters
        ----------
        src: np.ndarray
            Source image in RGB colour space.
        brightness: int (optional)
            Value between -100 ≤ v ≤ 100 
        contrast: int (optional)
            Value between 0 ≤ v ≤ 100
        dither: int (optional)
            Dither method
            
        Returns
        -------
        np.ndarray
            Rendered image in RGB format.
        """
        dst = src.copy()
        if kwargs.get("brightness") is not None:
            dst = Adjustment.brightness(dst, kwargs.get("brightness"))
        if kwargs.get("contrast") is not None:
            dst = Adjustment.contrast_scurve(dst, kwargs.get("contrast"))
        dst = Dither.dither(src, kwargs.get("dither", Dither.FLOYD_STEINBERG)) 

        if dst.ndim != 3:
            dst = cv.cvtColor(dst, cv.COLOR_GRAY2RGB)
        dst = self._blend(dst, self.background)
        return dst

    def _blend(self, fg, bg):
        fg = fg.astype(np.uint8)
        bg = bg.astype(np.uint8)
        dst = cv.bitwise_and(fg, bg)
        dst = np.clip(dst, 0, 255)
        return dst
