import cv2 as cv
import os


class Cartonifier:

    def __init__(self, n_downsampling_steps=2, n_filtering_steps=7):
        self.num_down = n_downsampling_steps
        self.num_bilateral = n_filtering_steps

    def process_frame(self, img_rgb, blf_d=9, blf_sc=9, blf_ss=7, blur_radius=3, thr_blocksize=9, thr_c=2):
        img_color = img_rgb
        for _ in range(self.num_down):
            img_color = cv.pyrDown(img_color)
        # repeatedly apply small bilateral filter instead of
        # applying one large filter
        for _ in range(self.num_bilateral):
            img_color = cv.bilateralFilter(img_color, d=blf_d, sigmaColor=blf_sc, sigmaSpace=blf_ss)
        # upsample image to original size
        for _ in range(self.num_down):
            img_color = cv.pyrUp(img_color)
        # convert to grayscale and apply median blur
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        img_blur = cv.medianBlur(img_gray, blur_radius)
        # detect and enhance edges
        img_edge = cv.adaptiveThreshold(img_blur, 255,
                                         cv.ADAPTIVE_THRESH_MEAN_C,
                                         cv.THRESH_BINARY,
                                         blockSize=thr_blocksize,
                                         C=thr_c)
        # convert back to color, bit-AND with color image
        img_edge = cv.cvtColor(img_edge, cv.COLOR_GRAY2RGB)
        if img_color.shape[0] != img_edge.shape[0] or img_color.shape[1] != img_edge.shape[1]:
            img_color = cv.resize(img_color, (img_edge.shape[1], img_edge.shape[0]))
        img_cartoon = cv.bitwise_and(img_color, img_edge)
        return img_edge, img_color

    def process(self, file_path, output_folder):
        if not os.path.exists(file_path):
            raise FileNotFoundError('File {} not found'.format(file_path))
        print('[*] Processing {}'.format(file_path))
        img_bgr = cv.imread(file_path)
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        img_cartoon = self.process_frame(img_rgb)
        img_cartoon_bgr = cv.cvtColor(img_cartoon, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(output_folder, os.path.basename(file_path)), img_cartoon_bgr)


