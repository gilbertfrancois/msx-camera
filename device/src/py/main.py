from kivy.config import Config


from kivy.app import App
from kivy.core.window import Window
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.lang import Builder
from kivy.uix.anchorlayout import AnchorLayout

import threading
import numpy as np
import cv2 as cv
from lib.msxcolor import MSXColor
import logging
import time

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

Builder.load_file('mainwindow.kv')
# Builder.load_file('toolbox.kv')


class CamView(AnchorLayout):

    stop = threading.Event()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clock = Clock.schedule_interval(self.update, 2.0 / 1.0)
        self.camera = self.ids["camera"]
        self.processed_frame = self.ids["processed_frame"]
        self.frame_count = 0
        self.msx_color = MSXColor()
        self.tic = 0
        # threading.Thread(target=self.update).start()

    def button_style_1_pressed(self):
        print("pressed 1")

    def button_style_2_pressed(self):
        print("pressed 2")

    def update(self, dt):
        tic = time.time()
        camera = self.ids["camera"]
        params = {
            "contrast": self.ids["contrast"].value,
            "hue": self.ids["hue"].value / 100,
            "sat": self.ids["sat"].value / 100,
            "lum": self.ids["lum"].value / 100,
        }
        texture = camera.texture
        if texture is not None:
            frame = np.frombuffer(self.camera.texture.pixels, np.uint8)
            frame = frame.reshape(texture.height, texture.width, 4)
            frame = frame[:, :, :3]
            frame = np.flipud(frame)
            frame = self.msx_color.screen2(frame, fg_style=1, bg_style=None, params=params)
            assert frame.ndim == 3
            assert frame.shape[2] == 3
            if frame.ndim == 2:
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
            self.update_frame(frame)
        toc = time.time()
        print(f"chrono_update: {toc - tic:0.3f} sec, chrono_total: {toc - self.tic:0.3f} sec")
        self.tic = toc


    @mainthread
    def update_frame(self, frame):    
        if self.processed_frame.texture.height != frame.shape[0] and self.processed_frame.texture.width != frame.shape[1]:
            self.processed_frame.texture = Texture.create(size=(frame.shape[1], frame.shape[0]))
        self.processed_frame.texture.blit_buffer(frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")


class MSXCamApp(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camview = CamView()

    def build(self):
        return self.camview

if __name__ == '__main__':
    Config.set('kivy', 'exit_on_escape', '1')
    Config.set('graphics', 'resizable', 1)
    Window.fullscreen = False
    MSXCamApp().run()
