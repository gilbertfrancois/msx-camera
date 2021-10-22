import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Dither:

    @staticmethod
    def simple(image):
        image = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        rows, cols = np.shape(image)
        out = cv.normalize(image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
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
        image = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE)
        rows, cols = np.shape(image)
        out = cv.normalize(image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
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
                out[i][j + 1] = out[i][j + 1] + ((7/16) * err)
                out[i + 1][j - 1] = out[i + 1][j - 1] + ((3/16) * err)
                out[i + 1][j] = out[i + 1][j] + ((5/16) * err)
                out[i + 1][j + 1] = out[i + 1][j + 1] + ((1/16) * err)
        out = (out*255).astype(np.uint8)
        return (out[1:rows - 1, 1:cols - 1])

    @staticmethod
    def jarvis_judice_ninke(image):
        image = cv.copyMakeBorder(image, 2, 2, 2, 2, cv.BORDER_REPLICATE)
        rows, cols = np.shape(image)
        out = cv.normalize(image.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
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

class MSXColor:
    def __init__(self):
        self.rgb_color_mat, self.hsv_color_mat, self.ycrcb_color_mat = self._get_color_matrices()

    def _get_color_matrices(self):
        hex_color_list = ["000000", "010101", "3eb849", "74d07d", "5955e0", "8076f1", "b95e51", "65dbef", 
                          "db6559", "ff897d", "ccc35e", "ded087", "3aa241", "b766b5", "cccccc", "ffffff"]
        rgb_color_list = []
        hsv_color_list = []
        ycrcb_color_list = []
        for index, hex_color in enumerate(hex_color_list):
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            rgb = np.array([r, g, b], dtype=np.uint8).reshape(1, 1, 3)
            hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
            ycrcb = cv.cvtColor(rgb, cv.COLOR_RGB2YCrCb)
            rgb = rgb.reshape(3)
            hsv = hsv.reshape(3)
            ycrcb = ycrcb.reshape(3)
            print(f"color {index:02d}: ({r}, {g}, {b}), ({hsv[0]}, {hsv[1]}, {hsv[2]})")
            rgb_color_list.append(rgb)
            hsv_color_list.append(hsv)
            ycrcb_color_list.append(ycrcb)
        rgb_color_mat = np.vstack(rgb_color_list).astype(np.float32)
        hsv_color_mat = np.vstack(hsv_color_list).astype(np.float32)
        ycrcb_color_mat = np.vstack(ycrcb_color_list).astype(np.float32)
        return rgb_color_mat, hsv_color_mat, ycrcb_color_mat

    def predict_background(self, image):
        tiles = cv.resize(image, (32, 24), interpolation=cv.INTER_LINEAR)
        rows, cols = tiles.shape[:2]
        hs_color = self.hsv_color_mat[:, :2]
        hs_color_norm = hs_color / np.linalg.norm(hs_color, axis=1).reshape(-1, 1)
        hs_color_norm = np.nan_to_num(hs_color_norm)
        hs_color_norm[14,:] = np.array([0.5, 0.5])
        tiles_hsv = cv.cvtColor(tiles, cv.COLOR_RGB2HSV)
        tiles_hs = tiles_hsv[:, :, :2]
        tiles_hs_norm = tiles_hs / np.linalg.norm(tiles_hs, axis=2).reshape(24, 32, 1)
        tiles_new = np.zeros_like(tiles_hsv)
        import pdb; pdb.set_trace()
        for i in range(rows):
            for j in range(cols):
                # fig, axs = plt.subplots(3, 1)
                tile_hs_ij = tiles_hsv[i, j]
                dist = np.sum(np.square(np.subtract(tile_hs_ij, self.hsv_color_mat)), axis=1)
                new_color_idx = np.argmin(dist)
                new_color = self.rgb_color_mat[new_color_idx]
                tiles_new[i, j, :] = new_color
        return tiles_new
        
    def map_ycrcb(self, image):
        tiles = cv.resize(image, (32, 24), interpolation=cv.INTER_LINEAR).astype(np.float32)
        rows, cols = tiles.shape[:2]
        tiles_ycrcb = cv.cvtColor(tiles, cv.COLOR_RGB2YCrCb)
        # tiles_ycrcb[:, :, 0] = tiles_ycrcb[:, :, 0] * 0.1
        ycrcb_color_mat = self.ycrcb_color_mat.copy()
        # ycrcb_color_mat[:, 0] = ycrcb_color_mat[:, 0] * 0.1
        tiles_new = np.zeros_like(tiles_ycrcb)
        import pdb; pdb.set_trace()
        for i in range(rows):
            for j in range(cols):
                tile_ij = tiles_ycrcb[i, j]
                dist = np.sum(np.square(np.subtract(tile_ij, ycrcb_color_mat)), axis=1)
                new_color_idx = np.argmin(dist)
                new_color = self.rgb_color_mat[new_color_idx]
                tiles_new[i, j, :] = new_color
        plt.figure(figsize=(16, 12))
        plt.imshow(tiles.astype(np.uint8))
        plt.show()
        plt.figure(figsize=(16, 12))
        plt.imshow(tiles_new.astype(np.uint8))
        plt.show()
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
        plt.figure(figsize=(16, 12))
        plt.imshow(tiles.astype(np.uint8))
        plt.show()
        plt.figure(figsize=(16, 12))
        plt.imshow(tiles_new.astype(np.uint8))
        plt.show()
        return tiles_new.astype(np.uint8)


if __name__ == "__main__":
    image_bgr = cv.imread("myimage.jpg")
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

    imageg = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    imageg = cv.resize(imageg, (256, 192))
    imagec = cv.resize(imageg, (32, 24), interpolation=cv.INTER_LINEAR)
    msx_color = MSXColor()
    imagecn = msx_color.map_ycrcb(image_rgb)
    cv.imwrite("imagecn.png", cv.cvtColor(imagecn, cv.COLOR_RGB2BGR))


    dither = Dither()
    d1image = dither.simple(imageg)
    d2image = dither.floyd_steinberg(imageg)
    d3image = dither.jarvis_judice_ninke(imageg)
    cv.imwrite("d1image.png", d1image)
    cv.imwrite("d2image.png", d2image)
    cv.imwrite("d3image.png", d3image)
