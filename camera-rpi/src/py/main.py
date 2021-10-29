from kivy.config import Config

Config.set('graphics', 'resizable', 0)

from kivy.app import App
from kivy.core.window import Window
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.lang import Builder
from kivy.uix.anchorlayout import AnchorLayout

import numpy as np
import cv2 as cv
from msxcolor import MSXColor

Builder.load_file('mainwindow.kv')
# Builder.load_file('toolbox.kv')


class CamView(AnchorLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clock = Clock.schedule_interval(self.update, 1.0 / 15.0)
        self.camera = self.ids["camera"]
        self.processed_frame = self.ids["processed_frame"]
        self.frame_count = 0
        self.msx_color = MSXColor()

    def update(self, dt):
        self.frame_count += 1
        camera = self.ids["camera"]
        params = {
            "fg_contrast": self.ids["fg_contrast"].value,
            "bg_scales": (self.ids["hue"].value / 100, self.ids["sat"].value / 100, self.ids["lum"].value / 100)
        }
        # try:
        # if self.frame_count > 20:
        #     import pdb; pdb.set_trace()

        texture = camera.texture
        if texture is not None:
            frame = np.frombuffer(self.camera.texture.pixels, np.uint8)
            frame = frame.reshape(texture.height, texture.width, 4)
            frame = frame[:, :, :3]
            frame = np.flipud(frame)
            frame = self.msx_color.screen2(frame, fg_style=3, bg_style=None, params=params)
            print(frame)
            if frame.ndim == 2:
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
            # frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_NEAREST)
            self.update_frame(frame)


    @mainthread
    def update_frame(self, frame):    
        # if self.processed_frame.texture.height != frame.shape[0] and self.processed_frame.texture.width != frame.shape[1]:
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
    Window.fullscreen = False
    MSXCamApp().run()
