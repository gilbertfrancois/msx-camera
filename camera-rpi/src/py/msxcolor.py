import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import logging
from dither import Dither
from adjustment import Adjustment
import color_transform as ct

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class MSXColor:
    PALETTE_MSX1_SHAPE = (16, 3)

    def __init__(self):
        self.palette_msx1_rgbi = np.zeros(MSXColor.PALETTE_MSX1_SHAPE, dtype=np.uint8)
        self.palette_msx1_hsvi = np.zeros(MSXColor.PALETTE_MSX1_SHAPE, dtype=np.uint8)
        self.palette_msx1_rgbf = np.zeros(MSXColor.PALETTE_MSX1_SHAPE, dtype=np.float64)
        self.palette_msx1_hsvf  = np.zeros(MSXColor.PALETTE_MSX1_SHAPE, dtype=np.float64)
        self.palette_msx1_hsvf_xy  = np.zeros(MSXColor.PALETTE_MSX1_SHAPE, dtype=np.float64)
        self.palette_msx1_labf = np.zeros(MSXColor.PALETTE_MSX1_SHAPE, dtype=np.float64)
        self.hex_color_list = ["#000000", "#010101", "#3eb849", "#74d07d", "#5955e0", "#8076f1", "#b95e51", "#65dbef", 
                               "#db6559", "#ff897d", "#ccc35e", "#ded087", "#3aa241", "#b766b5", "#cccccc", "#ffffff"]
        self.name_color_list = ["transparant", "black", "green2", "green3", "blue1", "blue2", "red1", "cyan",
                                "red2", "red3", "yellow1", "yellow2", "green1", "magenta", "gray", "white"]
        self._init_color_tables()

    def screen2(self, src, fg_style=1, bg_style="hsv", params={}):
        src = cv.resize(src, (256, 192), interpolation=cv.INTER_LINEAR)
        frame = self.style1(src, params)
        # frame = self.style2(src, params, self.palette_msx1_labf, ct.rgbi2labf)
        return frame

    def style1(self, frame_rgbi: np.ndarray, params: Dict) -> np.ndarray:
        fg = Adjustment.contrast_scurve(frame_rgbi, params.get("contrast"))
        fg = Dither.dither(fg, Dither.FLOYD_STEINBERG)
        bg = self.bg_map(frame_rgbi, params, self.palette_msx1_labf, ct.rgbi2labf, True)
        # bg = self.bg_map(frame_rgbi, params, self.palette_msx1_rgbf, ct.rgbi2rgbf, True)
        # bg = self.bg_map(frame_rgbi, params, self.palette_msx1_hsvf_xy, ct.rgbi2hsvf_xy, True)
        frame = self._blend(fg, bg)
        return frame

    def _init_color_tables(self):
        rgb_color_list = []
        for index, hex_color in enumerate(self.hex_color_list):
            hex_color = hex_color.replace("#", "")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            rgb_color_list.append([r, g, b])
        self.palette_msx1_rgbi = np.vstack(rgb_color_list).astype(np.uint8)
        self.palette_msx1_rgbf = ct.rgbi2rgbf(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)
        self.palette_msx1_hsvi = ct.rgbi2hsvi(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)
        self.palette_msx1_hsvf  = ct.rgbi2hsvf(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)
        self.palette_msx1_hsvf_xy = ct.rgbi2hsvf_xy(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)
        self.palette_msx1_labf = ct.rgbi2labf(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)
        # self._plot_vectors_3d(self.palette_msx1_hsvf_xy, self.hex_color_list)
        # self._plot_vectors_3d(self.palette_msx1_labf, self.hex_color_list)

    def style2(self, frame_rgbi: np.ndarray, params: Dict, palette: np.ndarray, cvt_fn) -> np.ndarray:
        scales = (params.get("hue", 1.0), params.get("sat", 1.0), params.get("lum", 1.0))
        scales = np.array([scales]).astype(np.float64)
        if params.get("contrast") is not None and params.get("contrast") > 0:
            frame_rgbi = Adjustment.contrast_scurve(frame_rgbi, params.get("contrast"))
        rows, cols = frame_rgbi.shape[:2]
        frame_cvt = cvt_fn(frame_rgbi) 
        palette_scaled = palette * scales
        frame_rgbi = Dither.simple_colormap(frame_cvt, palette_scaled, self.palette_msx1_rgbi)
        return frame_rgbi

    def fg_map(self, image:np.ndarray, params: Dict, palette: np.ndarray, cvt_fn) -> np.ndarray:
        scales = (params.get("hue", 1.0), params.get("sat", 1.0), params.get("lum", 1.0))
        scales = np.array([scales]).astype(np.float64)
        frame_rgbi = image
        rows, cols = frame_rgbi.shape[:2]
        frame_cvt = cvt_fn(frame_rgbi) 
        palette_scaled = palette * scales
        for i in range(rows):
            for j in range(cols):
                idx, _ = ct.l2_dist(frame_cvt[i, j, :], palette_scaled)
                new_color_idx = idx[0]
                if new_color_idx == 0:
                    new_color_idx = 1
                frame_rgbi[i, j, :] = self.palette_msx1_rgbi[new_color_idx]
        return frame_rgbi.astype(np.uint8)

    def fg_map_cielab(self, image:np.ndarray, params: Dict):
        scales = (params.get("hue", 1.0), params.get("sat", 1.0), params.get("lum", 1.0))
        scales = np.array([scales]).astype(np.float64)
        frame_rgbi = image
        rows, cols = frame_rgbi.shape[:2]
        frame_cvt = ct.rgbi2hsvf_xy(frame_rgbi) 
        table_cvt = self.palette_msx1_hsvf_xy * scales
        for i in range(rows):
            for j in range(cols):
                idx, _ = ct.l2_dist(frame_cvt[i, j, :], table_cvt)
                new_color_idx = idx[0]
                if new_color_idx == 0:
                    new_color_idx = 1
                frame_rgbi[i, j, :] = self.palette_msx1_rgbi[new_color_idx]
        return frame_rgbi.astype(np.uint8)



    def bg_map(self, image:np.ndarray, params: Dict, palette, cvt_fn, black2gray=True):
        scales = (params.get("hue", 1.0), params.get("sat", 1.0), params.get("lum", 1.0))
        scales = np.array([scales]).astype(np.float64)
        bg_colors_rgbi = cv.resize(image, (32, 192), interpolation=cv.INTER_LINEAR)
        rows, cols = bg_colors_rgbi.shape[:2]
        bg_colors_cvt = cvt_fn(bg_colors_rgbi) 
        palette_scaled = palette * scales
        for i in range(rows):
            for j in range(cols):
                idx, _ = ct.l2_dist(bg_colors_cvt[i, j, :], palette_scaled)
                new_color_idx = idx[0]
                if new_color_idx == 0:
                    new_color_idx = 1
                if new_color_idx == 1 and black2gray:
                    new_color_idx = 14
                bg_colors_rgbi[i, j, :] = self.palette_msx1_rgbi[new_color_idx]
        bg_colors_rgbi = cv.resize(bg_colors_rgbi, (256, 192), interpolation=cv.INTER_NEAREST)
        return bg_colors_rgbi.astype(np.uint8)

    def _blend(self, fg, bg):
        if fg.ndim != 3:
            fg = cv.cvtColor(fg, cv.COLOR_GRAY2RGB)
        fg = fg.astype(np.uint8)
        bg = bg.astype(np.uint8)
        dst = cv.bitwise_and(fg, bg)
        dst = np.clip(dst, 0, 255)
        return dst

    def _plot2(self, img1, img2):
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        if img1.ndim != 3:
            axs[0].imshow(img1, cmap="gray")
        else:
            axs[0].imshow(img1)
        axs[1].imshow(img2)
        plt.show()

    def _plot3(self, img1, img2, img3, img4):
        fig, axs = plt.subplots(2, 2)
        axs[0][0].imshow(img1)
        axs[1][0].imshow(img2)
        axs[0][1].imshow(img3)
        axs[1][1].imshow(img4)
        plt.show()

    def _plot_vectors_2d(self, cmap, pixel_hsvr):
        head_width = np.max(cmap)*.05
        plt.figure()
        for index, pixel in enumerate(cmap):
            plt.arrow(0, 0, pixel[0,0], pixel[0,1], color=f"#{self.hex_color_list[index]}", head_width=head_width, head_length=head_width)
        if pixel_hsvr is not None:
            plt.arrow(0, 0, pixel_hsvr[0,0], pixel_hsvr[0,1], color=f"black", head_width=head_width, head_length=head_width)
        plt.show()

    def _plot_vectors_3d(self, vectors: np.ndarray, color_list: List[str]):
        head_width = np.max(vectors)*.05
        plt.figure()
        ax = plt.gca(projection="3d")
        # zero = np.zeros_like(cmap[: , 0])
        zero = 0
        for i in range(vectors.shape[0]):
            ax.quiver(zero, zero, zero, vectors[i, 0], vectors[i, 1], vectors[i, 2], color=color_list[i])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1)
        plt.show()

if __name__ == "__main__":
    msx_color = MSXColor()
    # src = cv.imread("../../../resources/_RGB_24bits_palette_color_test_chart.png")
    src = cv.imread("in/test1.jpg")
    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    src = cv.resize(src, (256,192), interpolation=cv.INTER_LINEAR)
    dst = msx_color.fg_map(src.copy(), {"hue": 1.0, "sat": 1.0, "lum": 1.0}, msx_color.palette_msx1_labf, ct.rgbi2labf)
    dst2 = msx_color.fg_map(src.copy(), {"hue": 1.0, "sat": 1.0, "lum": 1.0}, msx_color.palette_msx1_hsvf_xy, ct.rgbi2hsvf_xy)
    dst3 = msx_color.fg_map(src.copy(), {"hue": 1.0, "sat": 1.0, "lum": 1.0}, msx_color.palette_msx1_rgbf, ct.rgbi2rgbf)
    # dst2 = msx_color.style2(src.copy(), {"hue": 1.0, "sat": 1.0, "lum": 1.0})
    # dst3 = msx_color.style1(src.copy(), {"contrast": 30, "hue": 1.0, "sat": 1.0, "lum": 1.0})
    msx_color._plot3(src, dst, dst2, dst3)


