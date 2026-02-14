import cv2
import numpy as np
import sys
import json
import time
import math
import smbus
from config import CAM_ID_LEFT, CAM_ID_RIGHT, BASELINE, MIN_DISPARITY, NUM_DISPARITIES, BLOCK_SIZE

# --- STESIS CONFIG ---
SCALE = 0.5  
global USE_WLS 
USE_WLS = True            

# --- RADAR & SWEEP CONFIG ---
SWEEP_SPEED_SEC = 2.0     
CHANGE_THRESHOLD = 5.0    
RADAR_FADE = 0.93         
MAX_RADAR_DIST_CM = 300   
FOV_H = 60.0              

class MPU6050:
    def __init__(self, address=0x68):
        self.address = address
        self.bus = smbus.SMBus(1)  
        self.bus.write_byte_data(self.address, 0x6B, 0)  
        self.yaw = 0.0
        self.pitch = 0.0
        self.last_time = time.time()

    def read_word(self, reg):
        h = self.bus.read_byte_data(self.address, reg)
        l = self.bus.read_byte_data(self.address, reg+1)
        value = (h << 8) + l
        if value >= 0x8000:
            return -((65535 - value) + 1)
        return value

    def get_accel(self):
        # Accelerometer Registers
        ax = self.read_word(0x3B)
        ay = self.read_word(0x3D)
        az = self.read_word(0x3F)
        # Scale for +/- 2g
        return ax / 16384.0, ay / 16384.0, az / 16384.0

    def update(self):
        # --- Gyroscope for Yaw ---
        gz = self.read_word(0x47)
        gz_deg_s = gz / 131.0  
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if abs(gz_deg_s) > 1.5:
            self.yaw += gz_deg_s * dt

        # --- Accelerometer for Pitch ---
        ax, ay, az = self.get_accel()
        # Calculate Pitch (up/down tilt) using Accel Y
        self.pitch = math.atan2(ay, math.sqrt(ax**2 + az**2)) * (180.0 / math.pi)
        
        return self.yaw, self.pitch

def draw_overlays(img):
    h, w = img.shape[:2]
    cx = w // 2
    for y in range(0, h, max(1, int(40 * SCALE))):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)
    cv2.line(img, (cx, 0), (cx, h), (255, 255, 0), 2)
    return img

def polar_to_cartesian(cx, cy, dist_cm, angle_deg, max_dist, canvas_radius):
    r = (dist_cm / max_dist) * canvas_radius
    r = min(r, canvas_radius)
    angle_rad = math.radians(angle_deg - 90)
    x = int(cx + r * math.cos(angle_rad))
    y = int(cy + r * math.sin(angle_rad))
    return x, y

def load_calibration(baseline_cm, scale=1.0):
    filename = f"stereo-{baseline_cm}.json"
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        M1, d1 = np.array(data["M1"]), np.array(data["d1"])
        M2, d2 = np.array(data["M2"]), np.array(data["d2"])
        R1, P1 = np.array(data["R1"]), np.array(data["P1"])
        R2, P2 = np.array(data["R2"]), np.array(data["P2"])
        Q = np.array(data["Q"])

        w = int(data["width"] * scale)
        h = int(data["height"] * scale)

        for M in [M1, M2, P1, P2]:
            M[0, 0] *= scale; M[1, 1] *= scale
            M[0, 2] *= scale; M[1, 2] *= scale
        
        Q[0, 3] *= scale; Q[1, 3] *= scale; Q[2, 3] *= scale 

        focal_length = M1[0,0] 
        return (M1, d1, M2, d2, R1, P1, R2, P2, Q, w, h, focal_length)
    except FileNotFoundError:
        print(f"Calibration file '{filename}' missing. Run calibrate.py.")
        sys.exit(1)

def main():
    global USE_WLS
    (M1, d1, M2, d2, R1, P1, R2, P2, Q, width, height, focal_length) = load_calibration(BASELINE, SCALE)

    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)
    cap_l.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_r.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (width, height), cv2.CV_16SC2)

    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE**2,
        P2=32 * 3 * BLOCK_SIZE**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    wls_filter = None
    right_matcher = None
    if USE_WLS:
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(stereo)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
            wls_filter.setLambda(8000.0)
            wls_filter.setSigmaColor(1.5)
            print("WLS Filter: ENABLED")
        except Exception as e:
            print(f"WLS Filter unavailable: {e}")
            USE_WLS = False

    try:
        mpu = MPU6050(address=0x68)
        print("MPU6050 initialized successfully on I2C Bus 1.")
    except Exception as e:
        print(f"Failed to connect to MPU6050: {e}.")
        sys.exit(1)

    radar_img = np.zeros((height, width, 3), dtype=np.uint8)
    prev_disparity = None
    last_sweep_x = 0

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r: break

        frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
        img_l = cv2.resize(frame_l, (width, height), interpolation=cv2.INTER_AREA)
        img_r = cv2.resize(frame_r, (width, height), interpolation=cv2.INTER_AREA)

        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        if USE_WLS and wls_filter and right_matcher:
            disp_l = stereo.compute(gray_l, gray_r)
            right_matcher.setNumDisparities(NUM_DISPARITIES)
            disp_r = right_matcher.compute(gray_r, gray_l)
            disparity_16S = wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)
            disparity = disparity_16S.astype(np.float32) / 16.0
        else:
            disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        disparity_clean = np.where(disparity > 0, disparity, 0)

        motion_mask = np.zeros_like(disparity_clean, dtype=bool)
        if prev_disparity is not None:
            diff = cv2.absdiff(disparity_clean, prev_disparity)
            motion_mask = diff > CHANGE_THRESHOLD
        prev_disparity = disparity_clean.copy()

        # --- Read both Yaw and Pitch ---
        current_yaw, current_pitch = mpu.update()

        view_l = draw_overlays(rect_l.copy())
        view_r = draw_overlays(rect_r.copy())
        top_row = np.hstack((view_l, view_r))

        disp_vis = (disparity - MIN_DISPARITY) / NUM_DISPARITIES
        disp_vis = np.clip(disp_vis * 255, 0, 255).astype(np.uint8)
        depth_view = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        depth_view[disparity <= 0] = [0, 0, 0] 

        # Fade persistent radar map
        radar_img = cv2.addWeighted(radar_img, RADAR_FADE, np.zeros_like(radar_img), 0.0, 0)
        radar_cx, radar_cy = width // 2, height // 2
        canvas_radius = min(width, height) // 2 - 10

        # Draw Grid on persistent map
        cv2.circle(radar_img, (radar_cx, radar_cy), canvas_radius, (50, 50, 50), 1)
        cv2.circle(radar_img, (radar_cx, radar_cy), int(canvas_radius * 0.66), (40, 40, 40), 1)
        cv2.circle(radar_img, (radar_cx, radar_cy), int(canvas_radius * 0.33), (30, 30, 30), 1)
        cv2.circle(radar_img, (radar_cx, radar_cy), 4, (255, 255, 0), -1) 

        curr_time = time.time()
        sweep_progress = (curr_time % SWEEP_SPEED_SEC) / SWEEP_SPEED_SEC
        curr_sweep_x = int(sweep_progress * width)

        sweep_range = range(last_sweep_x, curr_sweep_x) if curr_sweep_x >= last_sweep_x else range(0, curr_sweep_x)

        for sx in sweep_range:
            if sx >= width: continue
            
            col_angle_local = ((sx / width) * FOV_H) - (FOV_H / 2)
            global_angle = current_yaw + col_angle_local

            col_disp = disparity_clean[:, sx]
            max_d = np.max(col_disp)
            
            if max_d > 0:
                dist_cm = (focal_length * BASELINE) / max_d
                rx, ry = polar_to_cartesian(radar_cx, radar_cy, dist_cm, global_angle, MAX_RADAR_DIST_CM, canvas_radius)
                cv2.circle(radar_img, (rx, ry), 2, (0, 255, 0), -1)

            col_motion = motion_mask[:, sx]
            if np.any(col_motion):
                motion_disparities = col_disp[col_motion]
                if len(motion_disparities) > 0:
                    motion_d = np.max(motion_disparities)
                    if motion_d > 0:
                        motion_dist = (focal_length * BASELINE) / motion_d
                        rx_m, ry_m = polar_to_cartesian(radar_cx, radar_cy, motion_dist, global_angle, MAX_RADAR_DIST_CM, canvas_radius)
                        cv2.circle(radar_img, (rx_m, ry_m), 4, (0, 0, 255), -1)

        last_sweep_x = curr_sweep_x

        # --- CREATE CLEAN DISPLAY FRAME ---
        radar_display = radar_img.copy()

        # Draw Sweep Lines on clean display
        cv2.line(depth_view, (curr_sweep_x, 0), (curr_sweep_x, height), (255, 255, 255), 2)
        sweep_global_angle = current_yaw + (((curr_sweep_x / width) * FOV_H) - (FOV_H / 2))
        end_x, end_y = polar_to_cartesian(radar_cx, radar_cy, MAX_RADAR_DIST_CM, sweep_global_angle, MAX_RADAR_DIST_CM, canvas_radius)
        cv2.line(radar_display, (radar_cx, radar_cy), (end_x, end_y), (255, 255, 255), 1)

        # Draw Text on clean display (prevents ghosting)
        cv2.putText(radar_display, f"Yaw: {current_yaw:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(radar_display, f"Pitch: {current_pitch:.1f} deg", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        bottom_row = np.hstack((depth_view, radar_display))
        final_ui = np.vstack((top_row, bottom_row))

        cv2.imshow("Stesis Navigation Matrix", final_ui)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
