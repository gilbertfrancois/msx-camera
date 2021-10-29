import cv2 as cv
from dither import Dither
from msxcolor import MSXColor
from adjustment import Adjustment

def cam():
    import pdb; pdb.set_trace()
    LIVE_CAM = True
    IMAGE_URL = "in/test1.jpg"
    msx_color = MSXColor()

    fgstyle_map = {
            0: "simple",
            1: "floyd_steinberg",
            2: "jarvis_judice_ninke"
            }

    # Setup UI
    cv.namedWindow("MSX")
    cv.createTrackbar("h", "MSX", 0, 200, nop)
    cv.createTrackbar("s", "MSX", 0, 200, nop)
    cv.createTrackbar("v", "MSX", 0, 200, nop)
    cv.createTrackbar("fg", "MSX", 0, 2, nop)
    cv.createTrackbar("bg", "MSX", 0, 1, nop)
    
    cv.createTrackbar("c1", "MSX", 0, 128, nop)
    cv.createTrackbar("c2", "MSX", 0, 128, nop)
     
    if LIVE_CAM:
        # Setup camera
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
    else:
        # Load photo
        frame_bgr = cv.imread(IMAGE_URL)
        frame_bgr = cv.resize(frame_bgr, (256, 192))
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

    while True:
        if LIVE_CAM:
            ret, frame_bgr = cap.read()
            frame_bgr = cv.resize(frame_bgr, (256, 192))
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        # Read user input
        h = float(int(cv.getTrackbarPos("h", "MSX"))/100.0)
        s = float(int(cv.getTrackbarPos("s", "MSX"))/100.0)
        v = float(int(cv.getTrackbarPos("v", "MSX"))/100.0)
        fgstyle_idx = int(cv.getTrackbarPos("fg", "MSX"))
        c1 = int(cv.getTrackbarPos("c1", "MSX"))
        c2 = int(cv.getTrackbarPos("c2", "MSX"))
        scales = (h, s, v)
        # scales = (1.0, 1.0, 1.0)

        src_fg = Adjustment.contrast_linear(frame_rgb, c1)
        src_bg = Adjustment.contrast_linear(frame_rgb, c2)
        frame_sc2 = msx_color.screen2(src_fg, fg_style=fgstyle_map[fgstyle_idx], bg_style=None)
        frame_sc2 = cv.resize(frame_sc2, (800, 600), interpolation=cv.INTER_NEAREST)

        cv.imshow("MSX", cv.cvtColor(frame_sc2, cv.COLOR_RGB2BGR))
        if cv.waitKey(100) & 0xFF == ord('q'):
            break
        # else: 
        #     break
    if LIVE_CAM:
        cap.release()
    cv.destroyAllWindows()
