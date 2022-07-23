import numpy as np


class TMS99x8:

    SCREEN2_SHAPE = (192, 256, 3)
    PALETTE_SHAPE = (16, 3)
    PALETTE_HEX = ["#000000", "#010101", "#3eb849", "#74d07d", "#5955e0", "#8076f1", "#b95e51", "#65dbef", 
                   "#db6559", "#ff897d", "#ccc35e", "#ded087", "#3aa241", "#b766b5", "#cccccc", "#ffffff"]
    PALETTE_NAME = ["transparant", "black", "green2", "green3", "blue1", "blue2", "red1", "cyan",
                    "red2", "red3", "yellow1", "yellow2", "green1", "magenta", "gray", "white"]

    def __init__(self, **kwargs):
        # Palette in source and destination colorspace, sRGB (uint8)
        self.palette_rgbi = np.zeros(TMS99x8.PALETTE_SHAPE, dtype=np.uint8)
        # Palette in intermediate, computational colorspace, e.g. CieLAB, YCrCb, HSV. (float64)
        self.palette_cmpf = np.zeros(TMS99x8.PALETTE_SHAPE, dtype=np.float64)
        if kwargs.get("colorspace") is not None:
            print(f"Using {kwargs.get('colorspace')} as computational colorspace.")
        self.init_palette()


    def render(self, image):
        raise NotImplementedError(f"Needs to be overridden in a child class.")

    def save_to_sc2(self, filename):
        raise NotImplementedError(f"Not implemented yet.")

    def save_to_bin(self, filename, start_address):
        raise NotImplementedError(f"Not implemented yet.")

    def save_to_png(self, filename):
        raise NotImplementedError(f"Not implemented yet.")

    def save_to_numpy(self, filename):
        raise NotImplementedError(f"Not implemented yet.")

    def init_palette(self):
        rgb_color_list = []
        for hex_color in TMS99x8.PALETTE_HEX:
            hex_color = hex_color.replace("#", "")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            rgb_color_list.append([r, g, b])
        self.palette_rgbi = np.vstack(rgb_color_list).astype(np.uint8)
        # self.palette_msx1_rgbf = ct.rgbi2rgbf(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)
        # self.palette_msx1_hsvi = ct.rgbi2hsvi(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)
        # self.palette_msx1_hsvf  = ct.rgbi2hsvf(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)
        # self.palette_msx1_hsvf_xy = ct.rgbi2hsvf_xy(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)
        # self.palette_msx1_labf = ct.rgbi2labf(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)
        # self.palette_msx1_ycrcbf = ct.rgbi2ycrcbf(self.palette_msx1_rgbi.reshape(-1, 1, 3)).reshape(MSXColor.PALETTE_MSX1_SHAPE)

