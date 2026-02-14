import cv2
import numpy as np
import sys
import json
import time
import math
import smbus

# Ensure pyaudio is installed for speaker output
try:
    import pyaudio
except ImportError:
    print("PyAudio not found. Run: pip install pyaudio")
    sys.exit(1)

from config import (CAM_ID_LEFT, CAM_ID_RIGHT, BASELINE, MIN_DISPARITY, 
                    NUM_DISPARITIES, BLOCK_SIZE, LEFT_OFFSET, RING_UNIT, 
                    SPEED_THRESHOLD_CM, BOUNCE_SWEEP, ENABLE_MOTION_DETECTION,
                    SPECTRO_HEIGHT, SPECTRO_TIME_HISTORY_SEC, AUDIO_SMOOTHING_COEFF,
                    AUDIO_BASE_FREQ, AUDIO_MAX_FREQ)

# --- NEW AUDIO CONFIGURATIONS (IN MAIN.PY) ---
PLAY_AUDIO = True                # Toggle to output sound to speakers
AUDIO_SAMPLE_RATE = 44100        # Audio sample rate
AUDIO_CHUNK_SIZE = 1024          # Buffer size for audio chunks

AUDIO_MODE = 'DISTANCE_VERTICAL' # Options: 'DISTANCE_VERTICAL' or 'DISTANCE'
MAX_VOL_DIST_CM = 50.0           # Inner boundary: 100% Volume (Cyan circle)
MIN_VOL_DIST_CM = 250.0          # Outer boundary: 0% Volume (Gray circle)

# --- STESIS CONFIG ---
SCALE = 0.5  
global USE_WLS 
USE_WLS = True            

# --- RADAR & SWEEP CONFIG ---
SWEEP_SPEED_SEC = 2.0     
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
        ax = self.read_word(0x3B)
        ay = self.read_word(0x3D)
        az = self.read_word(0x3F)
        return ax / 16384.0, ay / 16384.0, az / 16384.0

    def update(self):
        gz = self.read_word(0x47)
        gz_deg_s = gz / 131.0  
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if abs(gz_deg_s) > 1.5:
            self.yaw += gz_deg_s * dt

        ax, ay, az = self.get_accel()
        self.pitch = math.atan2(ay, math.sqrt(ax**2 + az**2)) * (180.0 / math.pi)
        
        return self.yaw, self.pitch

# --- AUDIO SYNTHESIS TRACKER ---
class AudioEntity:
    def __init__(self, color_L, color_R):
        self.freq = AUDIO_BASE_FREQ
        self.amp_L = 0.0
        self.amp_R = 0.0
        self.color_L = color_L
        self.color_R = color_R
        self.prev_y = None
        self.phase = 0.0 

    def update(self, target_freq, target_amp_L, target_amp_R):
        if AUDIO_SMOOTHING_COEFF <= 0:
            self.freq = target_freq
            self.amp_L = target_amp_L
            self.amp_R = target_amp_R
        else:
            self.freq = (self.freq * AUDIO_SMOOTHING_COEFF) + (target_freq * (1.0 - AUDIO_SMOOTHING_COEFF))
            self.amp_L = (self.amp_L * AUDIO_SMOOTHING_COEFF) + (target_amp_L * (1.0 - AUDIO_SMOOTHING_COEFF))
            self.amp_R = (self.amp_R * AUDIO_SMOOTHING_COEFF) + (target_amp_R * (1.0 - AUDIO_SMOOTHING_COEFF))

def freq_to_y(freq, height):
    clamped_freq = max(AUDIO_BASE_FREQ, min(AUDIO_MAX_FREQ, freq))
    normalized = (clamped_freq - AUDIO_BASE_FREQ) / (AUDIO_MAX_FREQ - AUDIO_BASE_FREQ)
    return int(height - 1 - (normalized * (height - 1)))

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
        except Exception as e:
            print(f"WLS Filter unavailable: {e}")
            USE_WLS = False

    try:
        mpu = MPU6050(address=0x68)
    except Exception as e:
        print(f"Failed to connect to MPU6050: {e}.")
        sys.exit(1)

    main_audio = AudioEntity(color_L=(255, 255, 0), color_R=(255, 255, 0)) 
    motion_audio = AudioEntity(color_L=(0, 255, 0), color_R=(0, 255, 0))
    active_entities = [main_audio, motion_audio]

    # --- PYAUDIO SETUP ---
    pa = pyaudio.PyAudio()
    def audio_callback(in_data, frame_count, time_info, status):
        out_chunk = np.zeros((frame_count, 2), dtype=np.float32)
        if PLAY_AUDIO:
            t = np.arange(frame_count, dtype=np.float32) / AUDIO_SAMPLE_RATE
            for e in active_entities:
                if e.amp_L < 0.001 and e.amp_R < 0.001:
                    continue
                
                wave = np.sin(2 * np.pi * e.freq * t + e.phase)
                e.phase += 2 * np.pi * e.freq * frame_count / AUDIO_SAMPLE_RATE
                e.phase %= 2 * np.pi
                
                out_chunk[:, 0] += wave * e.amp_L
                out_chunk[:, 1] += wave * e.amp_R
            
        out_chunk = np.clip(out_chunk, -1.0, 1.0)
        return (out_chunk.tobytes(), pyaudio.paContinue)

    stream = pa.open(format=pyaudio.paFloat32,
                     channels=2,
                     rate=AUDIO_SAMPLE_RATE,
                     output=True,
                     frames_per_buffer=AUDIO_CHUNK_SIZE,
                     stream_callback=audio_callback)
    
    stream.start_stream()

    # UI Base Elements
    radar_img = np.zeros((height, width, 3), dtype=np.uint8)
    spectro_L = np.zeros((SPECTRO_HEIGHT, width, 3), dtype=np.uint8)
    spectro_R = np.zeros((SPECTRO_HEIGHT, width, 3), dtype=np.uint8)

    prev_distances = np.zeros(width)
    last_sweep_x = LEFT_OFFSET
    sweep_range_width = width - LEFT_OFFSET - 1 
    last_frame_time = time.time()

    while True:
        curr_time = time.time()
        dt = curr_time - last_frame_time
        last_frame_time = curr_time

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
        disparity_clean[:, :LEFT_OFFSET] = 0 

        current_yaw, current_pitch = mpu.update()

        view_l = draw_overlays(rect_l.copy())
        view_r = draw_overlays(rect_r.copy())

        disp_vis = (disparity - MIN_DISPARITY) / NUM_DISPARITIES
        disp_vis = np.clip(disp_vis * 255, 0, 255).astype(np.uint8)
        depth_view = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        depth_view[disparity <= 0] = [0, 0, 0] 
        depth_view[:, :LEFT_OFFSET] = [0, 0, 0]

        # Fade previous sweep points
        radar_img = cv2.addWeighted(radar_img, RADAR_FADE, np.zeros_like(radar_img), 0.0, 0)
        radar_cx, radar_cy = width // 2, height // 2
        canvas_radius = min(width, height) // 2 - 10

        # --- DISTANCE & MOTION AUDIO EVALUATION ---
        current_distances = np.zeros(width)
        current_y_coords = np.zeros(width, dtype=int)
        
        closest_motion_dist = MAX_RADAR_DIST_CM
        motion_x = -1
        motion_y = -1

        for sx in range(LEFT_OFFSET, width):
            col_disp = disparity_clean[:, sx]
            max_y = np.argmax(col_disp)
            max_d = col_disp[max_y]
            
            if max_d > 0:
                dist_cm = (focal_length * BASELINE) / max_d
                current_distances[sx] = dist_cm
                current_y_coords[sx] = max_y
                
                if ENABLE_MOTION_DETECTION and prev_distances[sx] > 0:
                    speed = abs(dist_cm - prev_distances[sx])
                    if speed > SPEED_THRESHOLD_CM:
                        col_angle_local = ((sx / width) * FOV_H) - (FOV_H / 2)
                        global_angle = current_yaw + col_angle_local
                        rx_m, ry_m = polar_to_cartesian(radar_cx, radar_cy, dist_cm, global_angle, MAX_RADAR_DIST_CM, canvas_radius)
                        cv2.circle(radar_img, (rx_m, ry_m), 4, (0, 255, 0), -1)
                        
                        if dist_cm < closest_motion_dist:
                            closest_motion_dist = dist_cm
                            motion_x = sx
                            motion_y = max_y

        prev_distances = current_distances.copy()

        # Update Motion Entity audio mappings
        if motion_x != -1:
            if AUDIO_MODE == 'DISTANCE_VERTICAL':
                t_freq = np.interp(motion_y, [height - 1, 0], [AUDIO_BASE_FREQ, AUDIO_MAX_FREQ])
                dist_vol = np.interp(closest_motion_dist, [MAX_VOL_DIST_CM, MIN_VOL_DIST_CM], [1.0, 0.0])
            else:
                t_freq = np.interp(closest_motion_dist, [0, MAX_RADAR_DIST_CM], [AUDIO_MAX_FREQ, AUDIO_BASE_FREQ])
                dist_vol = 1.0

            pan = np.interp(motion_x, [LEFT_OFFSET, width - 1], [0.0, 1.0])
            motion_audio.update(t_freq, target_amp_L=dist_vol * (1.0 - pan), target_amp_R=dist_vol * pan)
        else:
            motion_audio.update(AUDIO_BASE_FREQ, 0.0, 0.0) 

        # --- RADAR SWEEP LOGIC ---
        if BOUNCE_SWEEP:
            cycle_time = 2.0 * SWEEP_SPEED_SEC
            sweep_progress = (curr_time % cycle_time) / SWEEP_SPEED_SEC
            if sweep_progress <= 1.0:
                curr_sweep_x = LEFT_OFFSET + int(sweep_progress * sweep_range_width)
            else:
                curr_sweep_x = LEFT_OFFSET + int((2.0 - sweep_progress) * sweep_range_width)
        else:
            sweep_progress = (curr_time % SWEEP_SPEED_SEC) / SWEEP_SPEED_SEC
            curr_sweep_x = LEFT_OFFSET + int(sweep_progress * sweep_range_width)

        if not BOUNCE_SWEEP and curr_sweep_x < last_sweep_x:
            sweep_range = list(range(last_sweep_x, width)) + list(range(LEFT_OFFSET, curr_sweep_x))
        elif BOUNCE_SWEEP and curr_sweep_x < last_sweep_x:
            sweep_range = range(last_sweep_x, curr_sweep_x, -1)
        else:
            sweep_range = range(last_sweep_x, curr_sweep_x)

        closest_sweep_dist = MAX_RADAR_DIST_CM
        closest_sweep_y = height // 2
        valid_sweep = False

        for sx in sweep_range:
            if sx >= width or sx < LEFT_OFFSET: continue
            dist_cm = current_distances[sx]
            
            if dist_cm > 0:
                valid_sweep = True
                y_val = current_y_coords[sx]
                
                if dist_cm < closest_sweep_dist:
                    closest_sweep_dist = dist_cm
                    closest_sweep_y = y_val
                
                # Radar color coding for vertical placement: High (Y=0) -> Yellow, Low (Y=height) -> Red
                g = int(255 * (1.0 - (y_val / (height - 1))))
                color = (0, g, 255) # BGR
                
                col_angle_local = ((sx / width) * FOV_H) - (FOV_H / 2)
                global_angle = current_yaw + col_angle_local
                rx, ry = polar_to_cartesian(radar_cx, radar_cy, dist_cm, global_angle, MAX_RADAR_DIST_CM, canvas_radius)
                cv2.circle(radar_img, (rx, ry), 2, color, -1)

        # Update Main Audio Entity mappings
        if valid_sweep:
            if AUDIO_MODE == 'DISTANCE_VERTICAL':
                t_freq = np.interp(closest_sweep_y, [height - 1, 0], [AUDIO_BASE_FREQ, AUDIO_MAX_FREQ])
                dist_vol = np.interp(closest_sweep_dist, [MAX_VOL_DIST_CM, MIN_VOL_DIST_CM], [1.0, 0.0])
            else:
                t_freq = np.interp(closest_sweep_dist, [0, MAX_RADAR_DIST_CM], [AUDIO_MAX_FREQ, AUDIO_BASE_FREQ])
                dist_vol = 1.0
                
            pan = np.interp(curr_sweep_x, [LEFT_OFFSET, width - 1], [0.0, 1.0])
            main_audio.update(t_freq, target_amp_L=dist_vol * (1.0 - pan), target_amp_R=dist_vol * pan)
        else:
            main_audio.update(AUDIO_BASE_FREQ, 0.0, 0.0)

        last_sweep_x = curr_sweep_x

        # --- SPECTROGRAM RENDERING ---
        pixels_to_shift = max(1, int((dt / SPECTRO_TIME_HISTORY_SEC) * width))
        spectro_L = np.roll(spectro_L, -pixels_to_shift, axis=1)
        spectro_R = np.roll(spectro_R, -pixels_to_shift, axis=1)
        spectro_L[:, -pixels_to_shift:] = 0
        spectro_R[:, -pixels_to_shift:] = 0

        for grid_y in range(0, SPECTRO_HEIGHT, 50):
            cv2.line(spectro_L, (width - pixels_to_shift, grid_y), (width, grid_y), (30, 30, 30), 1)
            cv2.line(spectro_R, (width - pixels_to_shift, grid_y), (width, grid_y), (30, 30, 30), 1)

        def draw_entity_trace(entity, spectro_canvas, channel, width, pixels_to_shift):
            y = freq_to_y(entity.freq, SPECTRO_HEIGHT)
            amp = entity.amp_L if channel == 'L' else entity.amp_R
            
            if amp > 0.01: 
                color = tuple(int(c * amp) for c in (entity.color_L if channel == 'L' else entity.color_R))
                if entity.prev_y is not None:
                    cv2.line(spectro_canvas, (width - pixels_to_shift - 1, entity.prev_y), (width - 1, y), color, 2)
                else:
                    cv2.circle(spectro_canvas, (width - 1, y), 1, color, -1)
            
            return y if amp > 0.01 else None

        prev_yl_main = draw_entity_trace(main_audio, spectro_L, 'L', width, pixels_to_shift)
        prev_yr_main = draw_entity_trace(main_audio, spectro_R, 'R', width, pixels_to_shift)
        main_audio.prev_y = prev_yl_main if prev_yl_main is not None else prev_yr_main 

        prev_yl_mot = draw_entity_trace(motion_audio, spectro_L, 'L', width, pixels_to_shift)
        prev_yr_mot = draw_entity_trace(motion_audio, spectro_R, 'R', width, pixels_to_shift)
        motion_audio.prev_y = prev_yl_mot if prev_yl_mot is not None else prev_yr_mot

        # --- CREATE CLEAN DISPLAY FRAME ---
        radar_display = radar_img.copy()

        # Draw Base Radar UI dynamically over the faded image
        for d in range(RING_UNIT, MAX_RADAR_DIST_CM + 1, RING_UNIT):
            r = int((d / MAX_RADAR_DIST_CM) * canvas_radius)
            cv2.circle(radar_display, (radar_cx, radar_cy), r, (40, 40, 40), 1)
            if d == MAX_RADAR_DIST_CM:
                cv2.putText(radar_display, f"{d}cm", (radar_cx + 5, radar_cy - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Volume Boundaries (Cyan = 100%, Gray = 0%)
        r_max_vol = int((MAX_VOL_DIST_CM / MAX_RADAR_DIST_CM) * canvas_radius)
        r_min_vol = int((MIN_VOL_DIST_CM / MAX_RADAR_DIST_CM) * canvas_radius)
        cv2.circle(radar_display, (radar_cx, radar_cy), r_max_vol, (255, 255, 0), 1)
        cv2.circle(radar_display, (radar_cx, radar_cy), r_min_vol, (100, 100, 100), 1)
        
        # FOV Transparent Range Overlay
        radar_overlay = radar_display.copy()
        fov_min_angle = current_yaw - (FOV_H / 2)
        fov_max_angle = current_yaw + (FOV_H / 2)
        x_min, y_min = polar_to_cartesian(radar_cx, radar_cy, MAX_RADAR_DIST_CM, fov_min_angle, MAX_RADAR_DIST_CM, canvas_radius)
        x_max, y_max = polar_to_cartesian(radar_cx, radar_cy, MAX_RADAR_DIST_CM, fov_max_angle, MAX_RADAR_DIST_CM, canvas_radius)
        cv2.line(radar_overlay, (radar_cx, radar_cy), (x_min, y_min), (255, 255, 255), 2)
        cv2.line(radar_overlay, (radar_cx, radar_cy), (x_max, y_max), (255, 255, 255), 2)
        cv2.addWeighted(radar_overlay, 0.2, radar_display, 0.8, 0, radar_display)

        cv2.circle(radar_display, (radar_cx, radar_cy), 4, (255, 255, 0), -1) 

        # Draw active sweep line
        if curr_sweep_x >= LEFT_OFFSET:
            cv2.line(depth_view, (curr_sweep_x, 0), (curr_sweep_x, height), (255, 255, 255), 2)
            sweep_global_angle = current_yaw + (((curr_sweep_x / width) * FOV_H) - (FOV_H / 2))
            end_x, end_y = polar_to_cartesian(radar_cx, radar_cy, MAX_RADAR_DIST_CM, sweep_global_angle, MAX_RADAR_DIST_CM, canvas_radius)
            cv2.line(radar_display, (radar_cx, radar_cy), (end_x, end_y), (255, 255, 255), 1)

        cv2.putText(radar_display, f"Yaw: {current_yaw:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(radar_display, f"Pitch: {current_pitch:.1f} deg", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw Spectrogram Text on clean overlays
        spectro_L_disp = spectro_L.copy()
        spectro_R_disp = spectro_R.copy()
        cv2.putText(spectro_L_disp, "Left Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(spectro_R_disp, "Right Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Build UI Matrix
        top_row = np.hstack((view_l, view_r))
        mid_row = np.hstack((depth_view, radar_display))
        bot_row = np.hstack((spectro_L_disp, spectro_R_disp))
        final_ui = np.vstack((top_row, mid_row, bot_row))

        cv2.imshow("Stesis Navigation Matrix", final_ui)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Clean up
    stream.stop_stream()
    stream.close()
    pa.terminate()

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
