import cv2
import numpy as np
import time
import json
import sys
import os
from camera import Camera

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)): return int(obj)
        elif isinstance(obj, (np.floating, float)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Calibration(Camera):
    def __init__(self):
        super().__init__()
        
        # Calibration State
        self.objpoints = []
        self.imgpoints_l = []
        self.imgpoints_r = []
        
        self.scanning_mode = False
        self.overlay_mode = False
        self.auto_capture_mode = False
        self.image_count = 0
        
        # Auto-capture logic vars
        self.prev_corners_l = None
        self.stability_start_time = None
        self.last_capture_time = 0
        
        # Prepare Object Points
        self.objp = np.zeros((self.CHECKERBOARD_DIMS[0]*self.CHECKERBOARD_DIMS[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.CHECKERBOARD_DIMS[0], 0:self.CHECKERBOARD_DIMS[1]].T.reshape(-1,2)
        self.objp *= self.SQUARE_SIZE_CM

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

    def calculate_movement(self, curr_corners, prev_corners):
        if prev_corners is None or curr_corners is None:
            return 999.9
        diff = np.linalg.norm(curr_corners - prev_corners, axis=2)
        return np.mean(diff)

    def loop(self):
        # 1. Get Frames from Camera Class
        super().loop()
        frame_l, frame_r = self.frames
        
        if frame_l is None or frame_r is None:
            return

        h, w = frame_l.shape[:2]
        vis_l = frame_l.copy()
        vis_r = frame_r.copy()
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # Detect Checkerboards
        ret_c_l, ret_c_r = False, False
        corners_l, corners_r = None, None

        if self.scanning_mode:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            ret_c_l, corners_l = cv2.findChessboardCorners(gray_l, self.CHECKERBOARD_DIMS, flags)
            ret_c_r, corners_r = cv2.findChessboardCorners(gray_r, self.CHECKERBOARD_DIMS, flags)

            if ret_c_l:
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(vis_l, self.CHECKERBOARD_DIMS, corners_l, ret_c_l)
            if ret_c_r:
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(vis_r, self.CHECKERBOARD_DIMS, corners_r, ret_c_r)

        # UI & Visualization
        if self.overlay_mode:
            blended = cv2.addWeighted(vis_l, 0.5, vis_r, 0.5, 0)
            blended = self.draw_overlays(blended, mode='grid')
            combined = np.hstack((blended, blended))
            mode_text = "MODE: OVERLAY"
        else:
            vis_l = self.draw_overlays(vis_l, mode='lines')
            vis_r = self.draw_overlays(vis_r, mode='lines')
            combined = np.hstack((vis_l, vis_r))
            mode_text = "MODE: SIDE-BY-SIDE"

        # Scanning Logic
        capture_now = False
        status_text = "ALIGNMENT MODE (Press ENTER)"
        status_color = (0, 255, 255) 
        progress = 0.0

        if self.scanning_mode:
            current_time = time.time()
            if ret_c_l and ret_c_r:
                if self.auto_capture_mode:
                    if current_time - self.last_capture_time < self.POST_CAPTURE_COOLDOWN:
                        status_text = "Cooldown..."
                        status_color = (100, 100, 100)
                        self.stability_start_time = None 
                    else:
                        movement = self.calculate_movement(corners_l, self.prev_corners_l)
                        if movement < self.MOVEMENT_THRESHOLD:
                            if self.stability_start_time is None:
                                self.stability_start_time = current_time
                            
                            elapsed = current_time - self.stability_start_time
                            progress = min(elapsed / self.AUTO_CAPTURE_DELAY, 1.0)
                            status_text = f"STABLE: {int(progress*100)}%"
                            status_color = (0, 255, 0)
                            
                            if elapsed >= self.AUTO_CAPTURE_DELAY:
                                capture_now = True
                                self.stability_start_time = None
                        else:
                            self.stability_start_time = None
                            status_text = "Movement Detected"
                            status_color = (0, 0, 255)
                        self.prev_corners_l = corners_l
                else:
                    status_text = "READY (Press SPACE)"
                    status_color = (0, 255, 0)
            else:
                self.stability_start_time = None
                status_text = "Looking for board..."
                status_color = (0, 0, 255)

        # Draw Status
        cv2.putText(combined, f"{mode_text} | Captures: {self.image_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(combined, status_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        if self.auto_capture_mode:
            cv2.putText(combined, "[AUTO ACTIVE]", (w*2 - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if self.scanning_mode and progress > 0:
                bar_width = int((w*2) * progress)
                cv2.rectangle(combined, (0, h-20), (bar_width, h), (0, 255, 0), -1)

        cv2.imshow('Stereo Calib Class', combined)

        # Input Handling for Catching 'Space' or 'Enter'
        # Config.run handles 'q', but we need other keys.
        # But Config.run has a waitKey(1). We can't have two waitKeys or we miss events.
        # Ideally, Config.run should allow checking keys.
        # But here checking waitKey again essentially means we pause for another 1ms. It's acceptable.
        
        k = cv2.waitKey(1) & 0xFF
        if k == 32: capture_now = True # Space
        
        if capture_now and self.scanning_mode and ret_c_l and ret_c_r:
            self.imgpoints_l.append(corners_l)
            self.imgpoints_r.append(corners_r)
            self.objpoints.append(self.objp)
            self.image_count += 1
            self.last_capture_time = time.time()
            print(f"Captured Image #{self.image_count}")
            # Flash
            cv2.rectangle(combined, (0,0), (w*2, h), (255,255,255), -1)
            cv2.imshow('Stereo Calib Class', combined)
            cv2.waitKey(50)

        if k == ord('a'):
            self.auto_capture_mode = not self.auto_capture_mode
            self.stability_start_time = None
            print(f"Auto-Capture: {self.auto_capture_mode}")
        elif k == ord('o'): 
            self.overlay_mode = not self.overlay_mode
        elif k == 13: # Enter
            self.scanning_mode = not self.scanning_mode
            if not self.scanning_mode and self.image_count > 0:
                self.running = False # Stop loop to proceed to calibration

    def perform_calibration(self):
        print("Calibrating...")
        # Need width/height. We can get it from self.frames if available, else assume standard
        if self.frames[0] is None: 
            print("No frames available for calibration dimensions.")
            return

        h, w = self.frames[0].shape[:2]
        img_size = (w, h)

        ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_size, None, None)
        ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_size, None, None)

        flags = cv2.CALIB_FIX_INTRINSIC
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        ret_s, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, img_size, criteria=crit, flags=flags
        )

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, d1, M2, d2, img_size, R, T)

        calc_baseline = np.linalg.norm(T)
        fov_x_l = 2 * np.arctan(w / (2 * M1[0, 0])) * 180 / np.pi
        fov_x_r = 2 * np.arctan(w / (2 * M2[0, 0])) * 180 / np.pi

        print(f"RMS Stereo: {ret_s:.4f}")
        print(f"Baseline: {calc_baseline:.4f} cm")

        data = {
            "timestamp": time.time(),
            "baseline": self.BASELINE,
            "width": w, "height": h,
            "M1": M1, "d1": d1, "M2": M2, "d2": d2,
            "R": R, "T": T, "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
            "pixel_error": ret_s,
            "calculated_baseline": calc_baseline,
        }

        fname = f"stereo-{self.BASELINE}.json"
        with open(fname, 'w') as f: 
            json.dump(data, f, cls=NumpyEncoder, indent=4)
        print(f"Saved to {fname}")

    def cleanup(self):
        super().cleanup()
        if self.image_count > 0:
            self.perform_calibration()

if __name__ == "__main__":
    app = Calibration()
    app.run()
