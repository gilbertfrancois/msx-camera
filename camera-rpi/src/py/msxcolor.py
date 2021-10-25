import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from cartonify import Cartonifier
import logging
import time

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class MSXColor:
    def __init__(self):
        self.rgb_table = np.zeros((16, 1, 3), dtype=np.uint8)
        self.hsv_table = np.zeros((16, 1, 3), dtype=np.uint8)
        self.ycrcb_table = np.zeros((16, 1, 3), dtype=np.uint8)
        self.rgbf_table = np.zeros((16, 1, 3), dtype=np.float32)
        self.hsvf_table = np.zeros((16, 1, 3), dtype=np.float32)
        self.ycrcbf_table = np.zeros((16, 1, 3), dtype=np.float32)
        self.hex_color_list = ["000000", "010101", "3eb849", "74d07d", "5955e0", "8076f1", "b95e51", "65dbef", 
                          "db6559", "ff897d", "ccc35e", "ded087", "3aa241", "b766b5", "cccccc", "ffffff"]
        self.name_color_list = ["transparant", "black", "green2", "green3", "blue1", "blue2", "red1", "cyan",
                           "red2", "red3", "yellow1", "yellow2", "green1", "magenta", "gray", "white"]
        self.mpl_cmap = ["black", "black", "green", "green", "blue", "blue", "red", "cyan",
                    "red", "red", "yellow", "yellow", "green", "magenta", "gray", "white"]
        self._init_color_tables()

    def screen2(self, src, fg_style="simple", bg_style="hsv"):
        src = cv.resize(src, (256, 192), interpolation=cv.INTER_LINEAR)
        fg = Dither.simple(src)
        bg = self.map_hsv(src, s=[1.0, 1.0, 1.0])
        
    def _init_color_tables(self):
        rgb_color_list = []
        for index, hex_color in enumerate(self.hex_color_list):
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            rgb_color_list.append([r, g, b])
        # import pdb; pdb.set_trace()
        self.rgb_table = np.vstack(rgb_color_list).reshape(-1, 1, 3).astype(np.uint8)
        self.hsv_table = cv.cvtColor(self.rgb_table, cv.COLOR_RGB2HSV).astype(np.uint8)
        self.ycrcb_table = cv.cvtColor(self.rgb_table, cv.COLOR_RGB2YCrCb).astype(np.uint8)
        # Normalize the color tables:
        self.rgbf_table = (self.rgb_table / 255.0).astype(np.float32)

        self.hsvf_table = self.to_hsvf(self.rgb_table)
        self.hsvr_table = self.to_hsvr(self.rgb_table)
        self.plot_vectors_3d(self.hsvr_table, None)
        self.ycrcbf_table = (self.ycrcbf_table / 255.0).astype(np.float32)

    def to_hsvf(self, image):
        """ RGB to HSV

        Returns
        -------
        np.array
            hue in [0, 2Pi], sat in [0,1], value in [0,1]
        """
        dst = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        dst = dst.astype(np.float32)
        dst = dst * np.array([[[2 * np.pi / 180, 1/255, 1/255]]])
        return dst

    def to_hsvr(self, image):
        """ RGB to HSV (radians float)

        Returns
        -------
        np.array
            [sat * cos(hue), sat * sin(hue), value]
        """
        s = 1
        dst = self.to_hsvf(image)
        x = s * dst[:, :, 1] * np.cos(dst[:, :, 0])
        y = s * dst[:, :, 1] * np.sin(dst[:, :, 0])
        dst[:, :, 0] = x
        dst[:, :, 1] = y
        return dst

    def colormap():
        for hue in range(360):
            for value in range(255, 0):
                pass

        
    def plot_vectors(self, cmap, pixel_hsvr):
        head_width = np.max(cmap)*.05
        plt.figure()
        for index, pixel in enumerate(cmap):
            plt.arrow(0, 0, pixel[0,0], pixel[0,1], color=f"#{self.hex_color_list[index]}", head_width=head_width, head_length=head_width)
        if pixel_hsvr is not None:
            plt.arrow(0, 0, pixel_hsvr[0,0], pixel_hsvr[0,1], color=f"black", head_width=head_width, head_length=head_width)
        plt.show()

    def plot_vectors_3d(self, cmap, pixel_hsvr):
        head_width = np.max(cmap)*.05
        plt.figure()
        ax = plt.gca(projection="3d")
        zero = np.zeros_like(cmap[:,0,0])
        color_list = [f"#{c}" for c in self.hex_color_list]
        ax.quiver(zero, zero, zero, cmap[:, 0, 0], cmap[:, 0, 1], cmap[:, 0, 2], color=color_list)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1)
        plt.show()


    def map_hsv(self, image:np.ndarray, scales:Tuple[float, float, float], black2gray=True):
        # import pdb; pdb.set_trace()
        _scales = np.array([[scales]]).astype(np.float32)
        assert _scales.ndim == 3
        tiles = cv.resize(image, (32, 192), interpolation=cv.INTER_LINEAR)
        rows, cols = tiles.shape[:2]
        tiles_hsv = self.to_hsvr(tiles) * _scales
        _hsvr_table = self.hsvr_table * _scales
        _hsvr_table = _hsvr_table.reshape(16, 3)
        tiles_new = np.zeros_like(tiles_hsv)
        # import pdb; pdb.set_trace()
        for i in range(rows):
            for j in range(cols):
                tile_ij = tiles_hsv[i, j, :]
                dist = np.sum(np.square(np.subtract(tile_ij, _hsvr_table)), axis=1)
                new_color_idx = np.argmin(dist)
                if new_color_idx == 0:
                    new_color_idx = 1
                if new_color_idx == 1 and black2gray:
                    new_color_idx = 14
                new_color = self.rgb_table[new_color_idx]
                tiles_new[i, j, :] = new_color
        # self.plot(tiles, tiles_new.astype(np.uint8))
        tiles_new = cv.resize(tiles_new, (256, 192), interpolation=cv.INTER_NEAREST)
        return tiles_new.astype(np.uint8)

    def map_ycrcb(self, image, s):
        # import pdb; pdb.set_trace()
        tiles = cv.resize(image, (32, 192), interpolation=cv.INTER_LINEAR)
        tiles = (tiles / 255.0).astype(np.float32)
        rows, cols = tiles.shape[:2]
        tiles_ycrcb = cv.cvtColor(tiles, cv.COLOR_RGB2YCrCb).astype(np.float32)
        ycrcb_color_mat = self.ycrcb_color_mat.copy()
        for i, _s in enumerate(s):
            tiles_ycrcb[:, :, i] = tiles_ycrcb[:, :, i] * _s
            ycrcb_color_mat[:, i] = ycrcb_color_mat[:, i] * _s
        tiles_new = np.zeros_like(tiles_ycrcb)
        for i in range(rows):
            for j in range(cols):
                tile_ij = tiles_ycrcb[i, j, :]
                dist = np.sum(np.square(np.subtract(tile_ij, ycrcb_color_mat)), axis=1)
                new_color_idx = np.argmin(dist)
                if new_color_idx == 0 or new_color_idx == 1:
                    new_color_idx = 14
                new_color = self.rgb_color_mat[new_color_idx]
                tiles_new[i, j, :] = new_color
        # self.plot(tiles, tiles_new.astype(np.uint8))
        return tiles_new.astype(np.uint8)

    def map_rgb(self, image):
        tiles = cv.resize(image, (32, 24), interpolation=cv.INTER_LINEAR).astype(np.float32)
        rows, cols = tiles.shape[:2]
        tiles_new = np.zeros_like(tiles)
        # import pdb; pdb.set_trace()
        for i in range(rows):
            for j in range(cols):
                tile_ij = tiles[i, j].reshape(1, -1)
                dist = np.sum(np.square(np.subtract(tile_ij, self.rgb_color_mat)), axis=1)
                new_color_idx = np.argmin(dist)
                if new_color_idx == 0 or new_color_idx == 1:
                    new_color_idx = 14
                new_color = self.rgb_color_mat[new_color_idx]
                tiles_new[i, j, :] = new_color
        # self.plot(tiles, tiles_new.astype(np.uint8))
        return tiles_new.astype(np.uint8)

    def blend(self, fg, bg):
        if fg.ndim != 3:
            fg = cv.cvtColor(fg, cv.COLOR_GRAY2RGB)
        fg = fg.astype(np.uint8)
        bg = bg.astype(np.uint8)
        dst = cv.bitwise_and(fg, bg)
        # fg = fg.astype(np.float32)
        # bg = bg.astype(np.float32)
        # fg = 255 - fg
        dst = np.clip(dst, 0, 255)
        return dst

    def plot2(self, img1, img2):
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        if img1.ndim != 3:
            axs[0].imshow(img1, cmap="gray")
        else:
            axs[0].imshow(img1)
        axs[1].imshow(img2)
        plt.show()

    def plot3(self, img1, img2, img3, img4):
        fig, axs = plt.subplots(2, 2)
        axs[0][0].imshow(img1)
        axs[1][0].imshow(img2, cmap="gray")
        axs[0][1].imshow(img3)
        axs[1][1].imshow(img4)
        plt.show()

