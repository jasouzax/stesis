import cv2
import sys
from config import Config

class Camera(Config):
    cap_l = None
    cap_r = None
    frames = (None, None)
    ret_values = (False, False)

    def setup(self):
        print("Initializing Cameras...")

        # Connect to cameras
        self.cap_l = cv2.VideoCapture(self.CAM_ID_LEFT, cv2.CAP_V4L2)
        self.cap_r = cv2.VideoCapture(self.CAM_ID_RIGHT, cv2.CAP_V4L2)
        
        # Setup cameras
        for cap in [self.cap_l, self.cap_r]:
            if not cap.isOpened():
                print("Error: Camera is not opened")
                self.running = False
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        print("Cameras initialized.")

    def loop(self, show=True):
        if not self.cap_l or not self.cap_r:
            return True

        # Capture frame
        ret_l, frame_l = self.cap_l.read()
        ret_r, frame_r = self.cap_r.read()
        self.ret_values = (ret_l, ret_r)

        # Check failure
        if not ret_l or not ret_r:
            print("Failed to read frames.")
            return True

        # Right camera is upside down, in final design ensure not upside down
        frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)

        # Generate final window
        self.frames = (frame_l, frame_r)
        if show: self.gui = cv2.hconcat([frame_l, frame_r])
        return super().loop()

    def cleanup(self):
        print("Releasing cameras...")
        if self.cap_l: self.cap_l.release()
        if self.cap_r: self.cap_r.release()
        super().cleanup()

if __name__ == "__main__":
    app = Camera()
    app.run()
