import cv2
import numpy as np
import json
import sys
from camera import Camera

class Depth(Camera):
    def __init__(self):
        super().__init__()
        self.disparity = None
        self.depth_view = None
        self.depth = None
        self.stereo = None
        self.right_matcher = None
        self.wls_filter = None

        # Calibration Data
        self.M1 = self.M2 = self.d1 = self.d2 = self.R1 = self.R2 = self.P1 = self.P2 = self.Q = None
        self.map1_l = self.map2_l = self.map1_r = self.map2_r = None
        self.focal_length = 0
        
    def setup(self):
        super().setup()
        w, h = (0, 0)
        # Load calibration file
        try:
            with open(f"stereo-{self.BASELINE}.json", 'r') as f:
                data = json.load(f)
            scale = self.SCALE
            self.M1, self.d1 = np.array(data["M1"]), np.array(data["d1"])
            self.M2, self.d2 = np.array(data["M2"]), np.array(data["d2"])
            self.R1, self.P1 = np.array(data["R1"]), np.array(data["P1"])
            self.R2, self.P2 = np.array(data["R2"]), np.array(data["P2"])
            self.Q = np.array(data["Q"])
            w = int(data["width"] * scale)
            h = int(data["height"] * scale)

            for M in [self.M1, self.M2, self.P1, self.P2]:
                M[0, 0] *= scale; M[1, 1] *= scale
                M[0, 2] *= scale; M[1, 2] *= scale
            
            self.Q[0, 3] *= scale; self.Q[1, 3] *= scale; self.Q[2, 3] *= scale 
            self.focal_length = self.M1[0,0]
            
            print(f"Calibration loaded")
        except FileNotFoundError:
            print(f"[ERROR] Calibration file '{filename}' missing. Run calibrate.py first.")
            return True
        
        # Intialize the undistorted recified map
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(self.M1, self.d1, self.R1, self.P1, (w, h), cv2.CV_16SC2)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(self.M2, self.d2, self.R2, self.P2, (w, h), cv2.CV_16SC2)
        
        # Create the stereo matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.MIN_DISPARITY,
            numDisparities=self.NUM_DISPARITIES,
            blockSize=self.BLOCK_SIZE,
            P1=8 * 3 * self.BLOCK_SIZE**2,
            P2=32 * 3 * self.BLOCK_SIZE**2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        if self.USE_WLS:
            try:
                self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo)
                self.wls_filter.setLambda(8000.0)
                self.wls_filter.setSigmaColor(1.5)
                print("WLS Filter Enabled")
            except Exception as e:
                print(f"WLS Filter unavailable: {e}")

    def loop(self, show=True):
        # Get frames
        if super().loop(False, True): return True
        frame_l, frame_r = self.frames

        # Rectify
        rect_l = cv2.remap(frame_l, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        
        # Store rectified frames for subclasses/display
        self.rect_frames = (rect_l, rect_r) # New attribute

        # Compute Depth
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        if self.wls_filter and self.right_matcher:
            disp_l = self.stereo.compute(gray_l, gray_r)
            disp_r = self.right_matcher.compute(gray_r, gray_l)
            disparity_16S = self.wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)
            self.disparity = disparity_16S.astype(np.float32) / 16.0
        else:
            self.disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
            
        # Visual Depth
        disp_vis = (self.disparity - self.MIN_DISPARITY) / self.NUM_DISPARITIES
        disp_vis = np.clip(disp_vis * 255, 0, 255).astype(np.uint8)
        self.depth = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        self.depth[self.disparity <= 0] = [0, 0, 0] 
        self.depth[:, :self.LEFT_OFFSET] = [0, 0, 0]

        # Render
        self.gui = np.hstack(self.rect_frames)
        if show:
            self.depth = cv2.resize(self.depth, (self.width*2, self.height*2))
            self.gui = np.vstack((self.gui, self.depth))

if __name__ == "__main__":
    app = Depth()
    app.run()
