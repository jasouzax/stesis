import cv2
import sys
from config import Config

class Camera(Config):
    def __init__(self):
        super().__init__()
        self.cap_l = None
        self.cap_r = None
        self.frames = (None, None) # (frame_l, frame_r)
        self.ret_values = (False, False) # (ret_l, ret_r)

    def setup(self):
        print("Initializing Cameras...")
        self.cap_l = cv2.VideoCapture(self.CAM_ID_LEFT, cv2.CAP_V4L2)
        self.cap_r = cv2.VideoCapture(self.CAM_ID_RIGHT, cv2.CAP_V4L2)
        
        if not self.cap_l.isOpened() or not self.cap_r.isOpened():
            print("[ERROR] Could not open both cameras.")
            # We don't exit here to allow subclasses to handle error or run without cams if needed?
            # But normally we should probably flag error.
            self.running = False
            return

        for cap in [self.cap_l, self.cap_r]:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        print("Cameras initialized.")

    def loop(self):
        if not self.cap_l or not self.cap_r:
            return

        ret_l, frame_l = self.cap_l.read()
        ret_r, frame_r = self.cap_r.read()
        
        self.ret_values = (ret_l, ret_r)

        if not ret_l or not ret_r:
            print("Failed to read frames.")
            self.running = False
            return

        # Rotate right camera (as per original logic)
        frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
        self.frames = (frame_l, frame_r)
        
        super().loop()

    def draw_overlays(self, img, mode='lines'):
        h, w = img.shape[:2]
        cx = w // 2
        # Pitch Lines (Green)
        for y in range(0, h, 40):
            cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)
        # Yaw Center Line (Cyan)
        cv2.line(img, (cx, 0), (cx, h), (255, 255, 0), 2)
        # Roll Grid (Gray) - Only in Grid mode
        if mode == 'grid':
            for x in range(0, w, 40):
                cv2.line(img, (x, 0), (x, h), (100, 100, 100), 1)
        return img

    def cleanup(self):
        print("Releasing cameras...")
        if self.cap_l: self.cap_l.release()
        if self.cap_r: self.cap_r.release()
        super().cleanup()

if __name__ == "__main__":
    class CameraView(Camera):
        def loop(self):
            super().loop()
            if self.frames[0] is not None and self.frames[1] is not None:
                combined = cv2.hconcat([self.frames[0], self.frames[1]])
                cv2.imshow("Camera Class Test", combined)
    
    app = CameraView()
    app.run()
