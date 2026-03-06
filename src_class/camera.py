import cv2
#from config import Config
from imu import IMU

class Camera(IMU):
    def __init__(self):
        super().__init__()
        self.camera = (None, None)
        self.frames = [None, None]
        self.ret_values = (False, False)
        
    def setup(self):
        super().setup()
        print('Initializing Cameras...', end='', flush=True)
        # Setup cameras
        self.camera = tuple(cv2.VideoCapture(ID, cv2.CAP_V4L2) for ID in (self.CAM_ID_LEFT, self.CAM_ID_RIGHT))
        for cap in self.camera:
            if not cap.isOpened():
                print('\n\x1b[31mError:\x1b[0m Camera is not opened')
                return True
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Dimensions, width {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}, height {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print('\rCameras initialized.       ')

    def loop(self, show=True, resize=False):
        if super().loop(): return True

        # Check if camera is opened
        if not all(self.camera):
            print('\x1b[31mError:\x1b[0m Camera Capture Failed')
            return True

        # Capture frame
        self.ret_values, self.frames = zip(*(cap.read() for cap in self.camera))
        self.frames = list(self.frames)

        # Check failure
        if not all(self.ret_values):
            print('\x1b[31mError:\x1b[0m Failed to read frames.')
            return True

        # Flip images
        for n in range(2):
            self.frames[n] = cv2.rotate(self.frames[n], cv2.ROTATE_180)

        # Resize
        if resize:
            self.frames = [cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA) for frame in self.frames]

        # Generate final window
        if show: self.gui = cv2.hconcat(self.frames)
        self.frames = tuple(self.frames)
        return super().loop()

    def cleanup(self):
        print('Releasing Cameras...', end='', flush=True)
        for camera in self.camera:
            camera.release()
        print('\rCameras released.')
        super().cleanup()

if __name__ == "__main__":
    app = Camera()
    app.run()
