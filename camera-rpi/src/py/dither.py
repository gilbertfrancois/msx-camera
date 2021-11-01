import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import logging
import time
import color_transform as ct

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class Dither:

    SIMPLE = 0
    FLOYD_STEINBERG = 1
    JARVIS_JUDICE_NINKE = 2
    SIMPLE_RGB = 3

    # x/16 lookup table
    _1o16 = 1/16
    _3o16 = 3/16
    _5o16 = 5/16
    _7o16 = 7/16
    # x/48 lookup table
    _1o48 = 1/48
    _3o48 = 3/48
    _5o48 = 5/48
    _7o48 = 7/48

    @staticmethod
    def dither(image, style):
        if style == Dither.SIMPLE:
            return Dither.simple(image)
        elif style == Dither.FLOYD_STEINBERG:
            return Dither.floyd_steinberg(image)
        elif style == Dither.JARVIS_JUDICE_NINKE:
            return Dither.jarvis_judice_ninke(image)
        elif style == Dither.SIMPLE_RGB:
            return Dither.simple_rgb(image)
        else:
            raise ValueError("Requested dither style does not exist.")

    @staticmethod
    def simple_rgb(image, depth=2):
        if image.ndim != 3 and image.shape[2] != 3:
            raise RuntimeError(f"Image does not have the right dimensions.")
        out = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        rows, cols, channels = np.shape(out)
        out = out / 255
        # import pdb; pdb.set_trace()
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                pixel = np.round(depth * out[i][j]) / depth
                err = out[i][j] - pixel
                out[i][j] = pixel
                # error diffusion step
                out[i    ][j + 1] = out[i    ][j + 1] + (0.5 * err)
                out[i + 1][j    ] = out[i + 1][j    ] + (0.5 * err)
        out = np.clip(out, 0, 1)
        out = (out*255).astype(np.uint8)
        return(out[1:rows-1, 1:cols-1])

    @staticmethod
    def simple_colormap(image, src_colormap, dst_colormap):
        if image.ndim != 3 and image.shape[2] != 3:
            raise RuntimeError(f"Image does not have the right dimensions.")
        out = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        if dst_colormap is not None:
            dst = np.zeros_like(out, dtype=np.uint8)
        else:
            dst = out
        rows, cols, _ = out.shape
        # import pdb; pdb.set_trace()
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                idx, _ = ct.l2_dist(out[i, j, :], src_colormap)
                new_color_idx = idx[0]
                if new_color_idx == 0:
                    new_color_idx = 1
                new_color_ij = src_colormap[new_color_idx]
                err = out[i][j] - new_color_ij
                out[i][j] = new_color_ij
                dst[i][j] = dst_colormap[new_color_idx]
                # error diffusion step
                out[i    ][j + 1] = out[i    ][j + 1] + (0.5 * err)
                out[i + 1][j    ] = out[i + 1][j    ] + (0.5 * err)
        return(dst[1:rows-1, 1:cols-1])

    @staticmethod
    def simple(image):
        err_diff = np.array([[0, 0.5], [0.5, 0]])
        if image.ndim != 2:
            _image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        out = cv.copyMakeBorder(_image, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        rows, cols = np.shape(out)
        out = out / 255
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
                # out[i:i+2, j:j+2] = out[i:i+2, j:j+2] + err_diff * err 
                out[i    ][j + 1] = out[i    ][j + 1] + (0.5 * err)
                out[i + 1][j    ] = out[i + 1][j    ] + (0.5 * err)
        out = np.clip(out, 0, 1)
        out = (out*255).astype(np.uint8)
        return(out[1:rows-1, 1:cols-1])

    @staticmethod
    def floyd_steinberg(image):
        """
        err_diff_matrix = [[          *    7/16   ],
                           [   3/16  5/16  1/16   ]]


        """
        err_diff = np.array([[0.0, 0.0, 7.0/16], [3.0/16, 5.0/16, 1.0/16]])
        if image.ndim != 2:
            out = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            out = image.copy()
        out = out / 255
        out = cv.copyMakeBorder(out, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        rows, cols = np.shape(out)
        # out = cv.normalize(out.astype(np.float32), None, 0.0, 1.0, cv.NORM_MINMAX)
        outid1 = id(out)
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
                # out[i:i+2, j-1:j+2] = out[i:i+2, j-1:j+2] + err_diff * err
                out[i    ][j + 1] = out[i    ][j + 1] + (Dither._7o16 * err)
                out[i + 1][j - 1] = out[i + 1][j - 1] + (Dither._3o16 * err)
                out[i + 1][j    ] = out[i + 1][j    ] + (Dither._5o16 * err)
                out[i + 1][j + 1] = out[i + 1][j + 1] + (Dither._1o16 * err)
        out = np.clip(out, 0, 1)
        out = (out*255).astype(np.uint8)
        return (out[1:rows - 1, 1:cols - 1])

    @staticmethod
    def jarvis_judice_ninke(image):
        if image.ndim != 2:
            _image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        out = cv.copyMakeBorder(_image, 2, 2, 2, 2, cv.BORDER_REPLICATE)
        rows, cols = np.shape(out)
        # out = cv.normalize(_image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        out = out / 255
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
                out[i    ][j + 1] = out[i    ][j + 1] + (Dither._7o48 * err)
                out[i    ][j + 2] = out[i    ][j + 2] + (Dither._5o48 * err)
                out[i + 1][j - 2] = out[i + 1][j - 2] + (Dither._3o48 * err)
                out[i + 1][j - 1] = out[i + 1][j - 1] + (Dither._5o48 * err)
                out[i + 1][j    ] = out[i + 1][j    ] + (Dither._7o48 * err)
                out[i + 1][j + 1] = out[i + 1][j + 1] + (Dither._5o48 * err)
                out[i + 1][j + 2] = out[i + 1][j + 2] + (Dither._3o48 * err)
                out[i + 2][j - 2] = out[i + 2][j - 2] + (Dither._1o48 * err)
                out[i + 2][j - 1] = out[i + 2][j - 1] + (Dither._3o48 * err)
                out[i + 2][j    ] = out[i + 2][j    ] + (Dither._5o48 * err)
                out[i + 2][j + 1] = out[i + 2][j + 1] + (Dither._3o48 * err)
                out[i + 2][j + 2] = out[i + 2][j + 2] + (Dither._1o48 * err)
        out = (out*255).astype(np.uint8)
        return (out[2:rows - 2, 2:cols - 2])


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
