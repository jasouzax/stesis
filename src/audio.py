import numpy as np
import cv2
import math
import time
from config import (
    AUDIO_MODE, AUDIO_BASE_FREQ, AUDIO_MAX_FREQ, AUDIO_SMOOTHING_COEFF,
    MAX_VOL_DIST_CM, MIN_VOL_DIST_CM, MAX_RADAR_DIST_CM, LEFT_OFFSET,
    SPECTRO_HEIGHT, SPECTRO_TIME_HISTORY_SEC
)

# Import dependencies for NavSystem
import depth
import radar
import calibrate

try:
    import pyaudio
except ImportError:
    pyaudio = None
    print("PyAudio not found. Audio will be disabled.")

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
        # Smoothing
        if AUDIO_SMOOTHING_COEFF <= 0:
            self.freq = target_freq
            self.amp_L = target_amp_L
            self.amp_R = target_amp_R
        else:
            self.freq = (self.freq * AUDIO_SMOOTHING_COEFF) + (target_freq * (1.0 - AUDIO_SMOOTHING_COEFF))
            self.amp_L = (self.amp_L * AUDIO_SMOOTHING_COEFF) + (target_amp_L * (1.0 - AUDIO_SMOOTHING_COEFF))
            self.amp_R = (self.amp_R * AUDIO_SMOOTHING_COEFF) + (target_amp_R * (1.0 - AUDIO_SMOOTHING_COEFF))

class AudioManager:
    def __init__(self, rate=44100, chunk=1024):
        self.RATE = rate
        self.CHUNK = chunk
        self.entities = []
        self.stream = None
        self.p = None
        
        if pyaudio:
            try:
                self.p = pyaudio.PyAudio()
                self.stream = self.p.open(format=pyaudio.paFloat32,
                                          channels=2,
                                          rate=self.RATE,
                                          output=True,
                                          stream_callback=self._callback)
            except Exception as e:
                print(f"Audio Output Error: {e}")

    def add_entity(self, entity):
        self.entities.append(entity)

    def _callback(self, in_data, frame_count, time_info, status):
        out_data = np.zeros((frame_count, 2), dtype=np.float32)
        # t = np.arange(frame_count) / self.RATE # Unused
        
        for entity in self.entities:
            # Generate sine wave with phase continuity
            phase_increment = 2 * np.pi * entity.freq / self.RATE
            phases = entity.phase + np.arange(frame_count) * phase_increment
            entity.phase = (phases[-1] + phase_increment) % (2 * np.pi)
            
            wave = np.sin(phases)
            
            # Add to output
            out_data[:, 0] += wave * entity.amp_L
            out_data[:, 1] += wave * entity.amp_R
            
        # Clip
        out_data = np.clip(out_data, -1.0, 1.0)
        return (out_data.tobytes(), pyaudio.paContinue)

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    def process_radar_data(self, main_audio, motion_audio, sw_x, sw_dist, sw_y, mot_dist, mot_x, mot_y, valid_sweep, width, height):
        # Audio Logic - Motion
        if mot_x != -1:
            if AUDIO_MODE == 'DISTANCE_VERTICAL':
                t_freq = np.interp(mot_y, [height - 1, 0], [AUDIO_BASE_FREQ, AUDIO_MAX_FREQ])
                dist_vol = np.interp(mot_dist, [MAX_VOL_DIST_CM, MIN_VOL_DIST_CM], [1.0, 0.0])
            else:
                t_freq = np.interp(mot_dist, [0, MAX_RADAR_DIST_CM], [AUDIO_MAX_FREQ, AUDIO_BASE_FREQ])
                dist_vol = 1.0

            pan = np.interp(mot_x, [LEFT_OFFSET, width - 1], [0.0, 1.0])
            motion_audio.update(t_freq, target_amp_L=dist_vol * (1.0 - pan), target_amp_R=dist_vol * pan)
        else:
            motion_audio.update(AUDIO_BASE_FREQ, 0.0, 0.0)

        # Audio Logic - Sweep
        if valid_sweep:
            if AUDIO_MODE == 'DISTANCE_VERTICAL':
                t_freq = np.interp(sw_y, [height - 1, 0], [AUDIO_BASE_FREQ, AUDIO_MAX_FREQ])
                dist_vol = np.interp(sw_dist, [MAX_VOL_DIST_CM, MIN_VOL_DIST_CM], [1.0, 0.0])
            else:
                t_freq = np.interp(sw_dist, [0, MAX_RADAR_DIST_CM], [AUDIO_MAX_FREQ, AUDIO_BASE_FREQ])
                dist_vol = 1.0
                
            pan = np.interp(sw_x, [LEFT_OFFSET, width - 1], [0.0, 1.0])
            main_audio.update(t_freq, target_amp_L=dist_vol * (1.0 - pan), target_amp_R=dist_vol * pan)
        else:
            main_audio.update(AUDIO_BASE_FREQ, 0.0, 0.0)

def draw_entity_trace(entity, img, channel, width, pixels_to_shift):
    y = int(np.interp(entity.freq, [AUDIO_BASE_FREQ, AUDIO_MAX_FREQ], [img.shape[0]-1, 0]))
    
    amp = entity.amp_L if channel == 'L' else entity.amp_R
    color = entity.color_L if channel == 'L' else entity.color_R
    
    # Dim color by amplitude
    draw_color = (int(color[0]*amp), int(color[1]*amp), int(color[2]*amp))
    
    if entity.prev_y is not None:
        # Draw line from previous point (at width - pixels_to_shift) to current point (at width)
        # Because we shift the image left by pixels_to_shift before drawing
        cv2.line(img, (width - pixels_to_shift, entity.prev_y), (width, y), draw_color, 2)
    
    return y

class NavSystem:
    def __init__(self, width, height, focal_length, map1_l, map2_l, map1_r, map2_r, stereo, right_matcher, wls_filter):
        self.width = width
        self.height = height
        self.map1_l = map1_l
        self.map2_l = map2_l
        self.map1_r = map1_r
        self.map2_r = map2_r
        self.stereo = stereo
        self.right_matcher = right_matcher
        self.wls_filter = wls_filter
        
        self.radar_sys = radar.Radar(width, height, focal_length)
        
        # Audio Manager & Entities
        self.audio_mgr = AudioManager()
        self.main_audio = AudioEntity(color_L=(255, 255, 0), color_R=(255, 255, 0)) 
        self.motion_audio = AudioEntity(color_L=(0, 255, 0), color_R=(0, 255, 0))
        self.audio_mgr.add_entity(self.main_audio)
        self.audio_mgr.add_entity(self.motion_audio)
        
        # UI Buffers
        self.spectro_L = np.zeros((SPECTRO_HEIGHT, width, 3), dtype=np.uint8)
        self.spectro_R = np.zeros((SPECTRO_HEIGHT, width, 3), dtype=np.uint8)

    def process(self, frame_l, frame_r, current_yaw, current_pitch, pos_x, pos_y, dt, curr_time):
        # Rectify
        rect_l = cv2.remap(frame_l, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        # Depth & Radar
        disparity = depth.compute_depth_map(rect_l, rect_r, self.stereo, self.right_matcher, self.wls_filter)
        radar_view, sw_x, sw_dist, sw_y, mot_dist, mot_x, mot_y, valid_sweep = self.radar_sys.process(
            disparity, current_yaw, current_pitch, curr_time
        )

        # Update Audio based on Radar Data
        self.audio_mgr.process_radar_data(
            self.main_audio, self.motion_audio, 
            sw_x, sw_dist, sw_y, mot_dist, mot_x, mot_y, valid_sweep, 
            self.width, self.height
        )

        # Visualization
        vis_l = calibrate.draw_overlays(rect_l.copy())
        vis_r = calibrate.draw_overlays(rect_r.copy())
        depth_view = depth.get_visual_depth(disparity)
        
        if sw_x >= LEFT_OFFSET:
            cv2.line(depth_view, (sw_x, 0), (sw_x, height), (255, 255, 255), 2)
        
        cv2.putText(radar_view, f"POS X: {pos_x:.1f} Y: {pos_y:.1f}", (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Spectrograms (Audio Vis)
        pixels_to_shift = max(1, int((dt / SPECTRO_TIME_HISTORY_SEC) * self.width))
        self.spectro_L = np.roll(self.spectro_L, -pixels_to_shift, axis=1)
        self.spectro_R = np.roll(self.spectro_R, -pixels_to_shift, axis=1)
        self.spectro_L[:, -pixels_to_shift:] = 0
        self.spectro_R[:, -pixels_to_shift:] = 0

        for grid_y in range(0, SPECTRO_HEIGHT, 50):
            cv2.line(self.spectro_L, (self.width - pixels_to_shift, grid_y), (self.width, grid_y), (30, 30, 30), 1)
            cv2.line(self.spectro_R, (self.width - pixels_to_shift, grid_y), (self.width, grid_y), (30, 30, 30), 1)

        prev_yl_main = draw_entity_trace(self.main_audio, self.spectro_L, 'L', self.width, pixels_to_shift)
        prev_yr_main = draw_entity_trace(self.main_audio, self.spectro_R, 'R', self.width, pixels_to_shift)
        self.main_audio.prev_y = prev_yl_main if prev_yl_main is not None else prev_yr_main

        prev_yl_mot = draw_entity_trace(self.motion_audio, self.spectro_L, 'L', self.width, pixels_to_shift)
        prev_yr_mot = draw_entity_trace(self.motion_audio, self.spectro_R, 'R', self.width, pixels_to_shift)
        self.motion_audio.prev_y = prev_yl_mot if prev_yl_mot is not None else prev_yr_mot
        
        spectro_L_disp = self.spectro_L.copy()
        spectro_R_disp = self.spectro_R.copy()
        cv2.putText(spectro_L_disp, "Left Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(spectro_R_disp, "Right Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        top_row = np.hstack((vis_l, vis_r))
        mid_row = np.hstack((depth_view, radar_view))
        bot_row = np.hstack((spectro_L_disp, spectro_R_disp))
        final_ui = np.vstack((top_row, mid_row, bot_row))
        
        return final_ui

    def mute(self):
        self.main_audio.update(400, 0, 0)
        self.motion_audio.update(400, 0, 0)

    def close(self):
        self.audio_mgr.close()

import sys
import camera
import imu
from config import BASELINE, SCALE

if __name__ == "__main__":
    print("--- Audio/Nav Standalone Mode ---")
    
    # 1. Load Calibration
    try:
        (M1, d1, M2, d2, R1, P1, R2, P2, Q, width, height, focal_length) = depth.load_calibration(BASELINE, SCALE)
    except Exception as e:
        print(f"Calibration Error: {e}")
        sys.exit(1)

    # 2. Init Components
    cap_l, cap_r = camera.init_cameras()
    if not cap_l or not cap_r: sys.exit(1)

    # Stereo Setup
    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (width, height), cv2.CV_16SC2)
    stereo, right_matcher, wls_filter = depth.create_stereo_matcher()

    mpu = imu.MPU6050()
    nav_sys = NavSystem(width, height, focal_length, map1_l, map2_l, map1_r, map2_r, stereo, right_matcher, wls_filter)
    
    last_frame_time = time.time()
    print("System Ready. Press 'q' to exit.")

    try:
        while True:
            curr_time = time.time()
            dt = curr_time - last_frame_time
            last_frame_time = curr_time
            
            # Update IMU
            yaw, pitch, pos_x, pos_y, ax = mpu.update()
            
            # Get Frames
            ret_l, frame_l, ret_r, frame_r = camera.get_frames(cap_l, cap_r)
            if not ret_l or not ret_r: break
            
            frame_l = cv2.resize(frame_l, (width, height), interpolation=cv2.INTER_AREA)
            frame_r = cv2.resize(frame_r, (width, height), interpolation=cv2.INTER_AREA)
            
            # Process Nav System
            final_ui = nav_sys.process(
                frame_l, frame_r, yaw, pitch, pos_x, pos_y, dt, curr_time
            )
            
            cv2.imshow("Audio/Nav Standalone", final_ui)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    except KeyboardInterrupt:
        pass
    finally:
        nav_sys.close()
        camera.close_cameras(cap_l, cap_r)
        cv2.destroyAllWindows()
