import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from cartonify import Cartonifier
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class Dither:

    @staticmethod
    def simple(image):
        if image.ndim != 2:
            _image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        _image = cv.copyMakeBorder(_image, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        rows, cols = np.shape(_image)
        out = cv.normalize(_image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # threshold step
                if(out[i][j] > 0.5):
                    err = out[i][j] - 1
                    out[i][j] = 1
                else:
                    err = out[i][j]
                    out[i][j] = 0
                # error diffusion step
                out[i][j + 1] = out[i][j + 1] + (0.5 * err)
                out[i + 1][j] = out[i + 1][j] + (0.5 * err)
        out = (out*255).astype(np.uint8)
        return(out[1:rows-1, 1:cols-1])

    @staticmethod
    def floyd_steinberg(image):
        """
        err_diff_matrix = [[          *    7/16   ],
                           [   3/16  5/16  1/16   ]]


        """
        err_diff = np.array([[0.0, 0.0, 7.0/16], [3.0/16, 5.0/16, 1.0/16]], dtype=np.float32)
        if image.ndim != 2:
            out = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            out = image.copy()
        out = cv.copyMakeBorder(out, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        rows, cols = np.shape(out)
        out = cv.normalize(out.astype(np.float32), None, 0.0, 1.0, cv.NORM_MINMAX)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # threshold step
                if (out[i][j] > 0.5):
                    err = out[i][j] - 1
                    out[i][j] = 1
                else:
                    err = out[i][j]
                    out[i][j] = 0
                # error diffusion step
                out[i:i+1, j-1:j+1] = out[i:i+1, j-1:j+1] + err_diff * err
                # out[i][j + 1] = out[i][j + 1] + ((7/16) * err)
                # out[i + 1][j - 1] = out[i + 1][j - 1] + ((3/16) * err)
                # out[i + 1][j] = out[i + 1][j] + ((5/16) * err)
                # out[i + 1][j + 1] = out[i + 1][j + 1] + ((1/16) * err)
        out = (out*255).astype(np.uint8)
        return (out[1:rows - 1, 1:cols - 1])

    @staticmethod
    def floyd_steinberg

    @staticmethod
    def jarvis_judice_ninke(image):
        if image.ndim != 2:
            _image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        _image = cv.copyMakeBorder(_image, 2, 2, 2, 2, cv.BORDER_REPLICATE)
        rows, cols = np.shape(_image)
        out = cv.normalize(_image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                # threshold step
                if (out[i][j] > 0.5):
                    err = out[i][j] - 1
                    out[i][j] = 1
                else:
                    err = out[i][j]
                    out[i][j] = 0
                # error diffusion step
                out[i][j + 1] = out[i][j + 1] + ((7 / 48) * err)
                out[i][j + 2] = out[i][j + 2] + ((5 / 48) * err)
                out[i + 1][j - 2] = out[i + 1][j - 2] + ((3 / 48) * err)
                out[i + 1][j - 1] = out[i + 1][j - 1] + ((5 / 48) * err)
                out[i + 1][j] = out[i + 1][j] + ((7 / 48) * err)
                out[i + 1][j + 1] = out[i + 1][j + 1] + ((5 / 48) * err)
                out[i + 1][j + 2] = out[i + 1][j + 2] + ((3 / 48) * err)
                out[i + 2][j - 2] = out[i + 2][j - 2] + ((1 / 48) * err)
                out[i + 2][j - 1] = out[i + 2][j - 1] + ((3 / 48) * err)
                out[i + 2][j] = out[i + 2][j] + ((5 / 48) * err)
                out[i + 2][j + 1] = out[i + 2][j + 1] + ((3 / 48) * err)
                out[i + 2][j + 2] = out[i + 2][j + 2] + ((1 / 48) * err)
        out = (out*255).astype(np.uint8)
        return (out[2:rows - 2, 2:cols - 2])


class Adjustment:

    @staticmethod
    def contrast(src:np.ndarray, value:int):
        f = 131 * (value + 127) / (127 * (131 - value))
        alpha_c = f
        gamma_c = 127*(1-f)
        dst = cv.addWeighted(src, alpha_c, src, 0, gamma_c)
        return dst

    @staticmethod
    def brightness(src:np.ndarray, value:int):
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
 

class MSXColor:
    def __init__(self):
        self.rgb_table = np.zeros((16, 1, 3), dtype=np.uint8)
        self.hsv_table = np.zeros((16, 1, 3), dtype=np.uint8)
        self.ycrcb_table = np.zeros((16, 1, 3), dtype=np.uint8)
        self.rgbf_table = np.zeros((16, 1, 3), dtype=np.float32)
        self.hsvf_table = np.zeros((16, 1, 3), dtype=np.float32)
        self.ycrcbf_table = np.zeros((16, 1, 3), dtype=np.float32)
        self._init_color_tables()

    def screen2(self, src, fg_style="simple", bg_style="hsv"):
        src = cv.resize(src, (256, 192), interpolation=cv.INTER_LINEAR)
        fg = Dither.simple(src)
        bg = self.map_hsv(src, s=[1.0, 1.0, 1.0])

        

    def _init_color_tables(self):
        self.hex_color_list = ["000000", "010101", "3eb849", "74d07d", "5955e0", "8076f1", "b95e51", "65dbef", 
                          "db6559", "ff897d", "ccc35e", "ded087", "3aa241", "b766b5", "cccccc", "ffffff"]
        self.name_color_list = ["transparant", "black", "green2", "green3", "blue1", "blue2", "red1", "cyan",
                           "red2", "red3", "yellow1", "yellow2", "green1", "magenta", "gray", "white"]
        self.mpl_cmap = ["black", "black", "green", "green", "blue", "blue", "red", "cyan",
                    "red", "red", "yellow", "yellow", "green", "magenta", "gray", "white"]
        rgb_color_list = []
        for index, hex_color in enumerate(self.hex_color_list):
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            rgb_color_list.append([r, g, b])
            # rgb = np.array([r, g, b], dtype=np.uint8).reshape(1, 1, 3)
            # rgbf = (rgb / 255.0).astype(np.float32)
            # hsv = cv.cvtColor(rgbf, cv.COLOR_RGB2HSV)
            # ycrcb = cv.cvtColor(rgbf, cv.COLOR_RGB2YCrCb)
            # rgb = rgb.reshape(3)
            # hsv = hsv.reshape(3)
            # ycrcb = ycrcb.reshape(3)
            # print(f"color {index:02d}: ({r}, {g}, {b}), ({hsv[0]:0.2f}, {hsv[1]:0.2f}, {hsv[2]:0.2f}), ({ycrcb[0]:0.2f}, {ycrcb[1]:0.2f}, {ycrcb[2]:0.2f})")
            # rgb_color_list.append(rgb)
            # hsv_color_list.append(hsv)
            # ycrcb_color_list.append(ycrcb)
        # import pdb; pdb.set_trace()
        self.rgb_table = np.vstack(rgb_color_list).reshape(-1, 1, 3).astype(np.uint8)
        self.hsv_table = cv.cvtColor(self.rgb_table, cv.COLOR_RGB2HSV).astype(np.uint8)
        self.ycrcb_table = cv.cvtColor(self.rgb_table, cv.COLOR_RGB2YCrCb).astype(np.uint8)
        # Normalize the color tables:
        self.rgbf_table = (self.rgb_table / 255.0).astype(np.float32)

        self.hsvf_table = self.to_hsvf(self.rgb_table)
        self.hsvr_table = self.to_hsvr(self.rgb_table)
        self.plot_vectors(self.hsvr_table, None)
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
        s = 10
        dst = self.to_hsvf(image)
        x = s * dst[:, :, 1] * np.cos(dst[:, :, 0])
        y = s * dst[:, :, 1] * np.sin(dst[:, :, 0])
        dst[:, :, 0] = x
        dst[:, :, 1] = y
        return dst
        
    def plot_vectors(self, cmap, pixel_hsvr):
        head_width = np.max(cmap)*.05
        plt.figure()
        for index, pixel in enumerate(cmap):
            plt.arrow(0, 0, pixel[0,0], pixel[0,1], color=f"#{self.hex_color_list[index]}", head_width=head_width, head_length=head_width)
        if pixel_hsvr is not None:
            plt.arrow(0, 0, pixel_hsvr[0,0], pixel_hsvr[0,1], color=f"black", head_width=head_width, head_length=head_width)
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

def nop(x):
    pass

def cam():
    dither = Dither()
    msx_color = MSXColor()
    cartonifier = Cartonifier()
    # image_bg = msx_color.map_hsv(image_rgb, s=[1.0, 1.0, 1.0], brightness=2.0)
    # # image_bg = msx_color.map_ycrcb(image_rgb, s=[1.0, 1.0, 1.0])
    # image_bg = cv.resize(image_bg, (256, 192), interpolation=cv.INTER_NEAREST)

    # image_ds = dither.simple(image_gray)
    # image_dfs = dither.floyd_steinberg(image_gray)
    # image_djjn = dither.jarvis_judice_ninke(image_gray)

    # image_final = msx_color.blend(image_djjn, image_bg)
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cv.namedWindow("MSX")
    cv.createTrackbar("h", "MSX", 0, 200, nop)
    cv.createTrackbar("s", "MSX", 0, 200, nop)
    cv.createTrackbar("v", "MSX", 0, 200, nop)
    cv.createTrackbar("fg", "MSX", 0, 2, nop)
    cv.createTrackbar("c1", "MSX", 0, 128, nop)
    cv.createTrackbar("c2", "MSX", 0, 128, nop)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    # frame_bgr = cv.imread("pic1.jpg")
    # frame_bgr = cv.resize(frame_bgr, (256, 192))
    while True:
        ret, frame_bgr = cap.read()
        frame_bgr = cv.resize(frame_bgr, (256, 192))
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

        h = float(int(cv.getTrackbarPos("h", "MSX"))/100.0)
        s = float(int(cv.getTrackbarPos("s", "MSX"))/100.0)
        v = float(int(cv.getTrackbarPos("v", "MSX"))/100.0)
        fpstyle = int(cv.getTrackbarPos("fg", "MSX"))
        c1 = int(cv.getTrackbarPos("c1", "MSX"))
        c2 = int(cv.getTrackbarPos("c2", "MSX"))
        scales = (h, s, v)
        # scales = (1.0, 1.0, 1.0)

        src_fg = Adjustment.contrast(frame_rgb, c1)
        bg = None
        fg = None
        black2gray = True
        if fpstyle == 0:
            fg = dither.jarvis_judice_ninke(src_fg)
        elif fpstyle == 1:
            fg = dither.floyd_steinberg(src_fg)
        elif fpstyle == 2:
            fg, src_bg = cartonifier.process_frame(src_fg)
            black2gray = False
        src_bg = Adjustment.contrast(frame_rgb, c2)
        bg = msx_color.map_hsv(src_bg, scales, black2gray)
        frame_sc2 = msx_color.blend(fg, bg)
        # frame_sc2 = cartonifier.process_frame(cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))
        frame_sc2 = cv.resize(frame_sc2, (800, 600), interpolation=cv.INTER_NEAREST)

        cv.imshow("MSX", cv.cvtColor(frame_sc2, cv.COLOR_RGB2BGR))
        if cv.waitKey(100) & 0xFF == ord('q'):
            break
        # else: 
        #     break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    cam()
    # image_bgr = cv.imread("pic2.jpg")
    # image_bgr = cv.resize(image_bgr, (256, 192))
    # image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    # image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

    # msx_color = MSXColor()
    # bg = msx_color.map_hsv(image_rgb, (1.0, 1.0, 1.0))
    # plt.figure()
    # plt.imshow(bg)
    # plt.show()

    # msx_color = MSXColor()
    # image_bg = msx_color.map_hsv(image_rgb, s=[1.0, 1.0, 1.0], brightness=2.0)
    # # image_bg = msx_color.map_ycrcb(image_rgb, s=[1.0, 1.0, 1.0])
    # image_bg = cv.resize(image_bg, (256, 192), interpolation=cv.INTER_NEAREST)

    # dither = Dither()
    # image_ds = dither.simple(image_gray)
    # image_dfs = dither.floyd_steinberg(image_gray)
    # image_djjn = dither.jarvis_judice_ninke(image_gray)

    # image_final = msx_color.blend(image_djjn, image_bg)
    # msx_color.plot3(image_bg, image_djjn, image_rgb, image_final)
    # msx_color.plot2(image_djjn, image_final)


#     cv.imwrite("image_ds.png", image_ds)
#     cv.imwrite("image_dfs.png", image_dfs)
#     cv.imwrite("image_djjn.png", image_djjn)
#     cv.imwrite("d4image.png", image_bg)
