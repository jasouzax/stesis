import cv2
import numpy as np
import time
import sys
import threading
import http.server
import socketserver
import os

# Import custom modules
from config import (
    CAM_ID_LEFT, CAM_ID_RIGHT, BASELINE, SCALE,
    SPECTRO_HEIGHT, SPECTRO_TIME_HISTORY_SEC, LEFT_OFFSET
)
import imu
import calibrate
import depth
import radar
import audio
import camera
import hand

# Constants for Mode Switching
# Inactive: pitch >= -10 (and ax >= -2 implied else Hand)
# Hand: accel_x < -2 AND pitch < -10
# Nav/Audio: accel_x >= -2 AND pitch < -10

def start_web_server():
    # Serve files from ../web relative to src/
    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../web')
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=web_dir, **kwargs)
    
    try:
        # Binding to 0.0.0.0 allows external access, port 80 requires admin on some OS
        with socketserver.TCPServer(("0.0.0.0", 80), Handler) as httpd:
            print(f"Serving web content at http://0.0.0.0:80 from {web_dir}")
            httpd.serve_forever()
    except Exception as e:
        print(f"Web Server Error: {e}")

def main():
    print("Initializing Stesis...")
    
    # Start Web Server
    server_thread = threading.Thread(target=start_web_server, daemon=True)
    server_thread.start()

    # 1. Load Calibration
    try:
        (M1, d1, M2, d2, R1, P1, R2, P2, Q, width, height, focal_length) = depth.load_calibration(BASELINE, SCALE)
    except Exception as e:
        print(f"Calibration Error: {e}")
        return

    # 2. Init Components
    cap_l, cap_r = camera.init_cameras()
    if not cap_l or not cap_r: sys.exit(1)

    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (width, height), cv2.CV_16SC2)
    stereo, right_matcher, wls_filter = depth.create_stereo_matcher()

    mpu = imu.MPU6050()
    radar_sys = radar.Radar(width, height, focal_length)
    audio_sys = audio.AudioManager()
    hand_sys = hand.HandGesture()
    
    main_audio = audio.AudioEntity(color_L=(255, 255, 0), color_R=(255, 255, 0)) 
    motion_audio = audio.AudioEntity(color_L=(0, 255, 0), color_R=(0, 255, 0))
    audio_sys.add_entity(main_audio)
    audio_sys.add_entity(motion_audio)

    # UI Buffers
    spectro_L = np.zeros((SPECTRO_HEIGHT, width, 3), dtype=np.uint8)
    spectro_R = np.zeros((SPECTRO_HEIGHT, width, 3), dtype=np.uint8)
    last_frame_time = time.time()

    print("System Ready. Press 'q' to exit.")

    try:
        while True:
            curr_time = time.time()
            dt = curr_time - last_frame_time
            last_frame_time = curr_time

            # 1. Read IMU for State Logic
            current_yaw, current_pitch, pos_x, pos_y, ax = mpu.update()

            # Determine Mode
            # "read the imu first, if the accel x is less than -2 and the pitch is less than -10 degrees then the interface is of 'hand.py', 
            # if accel x is not less than but pitch is still less than -10 then show audio.py, 
            # if neither then show an empty window saying 'inactive'"
            
            # Logic Table:
            # Pitch < -10 ?
            #   No -> INACTIVE
            #   Yes -> Check Ax
            #          Ax < -2 ?
            #            Yes -> HAND
            #            No  -> NAV (Audio/Radar)
            
            MODE = "INACTIVE"
            if current_pitch < -10:
                if ax < -2:
                    MODE = "HAND"
                else:
                    MODE = "NAV"
            
            # 2. Get Frames
            ret_l, frame_l, ret_r, frame_r = camera.get_frames(cap_l, cap_r)
            if not ret_l or not ret_r: break
            
            frame_l = cv2.resize(frame_l, (width, height), interpolation=cv2.INTER_AREA)
            frame_r = cv2.resize(frame_r, (width, height), interpolation=cv2.INTER_AREA)

            # 3. Execution per Mode
            final_ui = None
            
            if MODE == "INACTIVE":
                # Mute Audio
                main_audio.update(400, 0, 0)
                motion_audio.update(400, 0, 0)
                
                # Show black screen with text
                inactive_screen = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(inactive_screen, "INACTIVE", (width//2 - 60, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                
                final_ui = inactive_screen

            elif MODE == "HAND":
                # Mute Audio
                main_audio.update(400, 0, 0)
                motion_audio.update(400, 0, 0)
                
                # Process Right Camera (frame_r is already rotated 180 by camera.get_frames)
                hand_view, msg = hand_sys.process(frame_r.copy())
                if msg:
                    print(f"Hand Event: {msg}")
                
                final_ui = hand_view
                
            elif MODE == "NAV":
                # Rectify
                rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
                rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

                # Depth & Radar
                disparity = depth.compute_depth_map(rect_l, rect_r, stereo, right_matcher, wls_filter)
                radar_view, sw_x, sw_dist, sw_y, mot_dist, mot_x, mot_y, valid_sweep = radar_sys.process(
                    disparity, current_yaw, current_pitch, curr_time
                )

                # Audio (New Logic)
                audio_sys.process_radar_data(
                    main_audio, motion_audio, 
                    sw_x, sw_dist, sw_y, mot_dist, mot_x, mot_y, valid_sweep, 
                    width, height
                )

                # Visualization
                vis_l = calibrate.draw_overlays(rect_l.copy())
                vis_r = calibrate.draw_overlays(rect_r.copy())
                depth_view = depth.get_visual_depth(disparity)
                
                if sw_x >= LEFT_OFFSET:
                    cv2.line(depth_view, (sw_x, 0), (sw_x, height), (255, 255, 255), 2)
                
                cv2.putText(radar_view, f"POS X: {pos_x:.1f} Y: {pos_y:.1f}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Spectrograms (Audio Vis)
                pixels_to_shift = max(1, int((dt / SPECTRO_TIME_HISTORY_SEC) * width))
                spectro_L = np.roll(spectro_L, -pixels_to_shift, axis=1)
                spectro_R = np.roll(spectro_R, -pixels_to_shift, axis=1)
                spectro_L[:, -pixels_to_shift:] = 0
                spectro_R[:, -pixels_to_shift:] = 0

                for grid_y in range(0, SPECTRO_HEIGHT, 50):
                    cv2.line(spectro_L, (width - pixels_to_shift, grid_y), (width, grid_y), (30, 30, 30), 1)
                    cv2.line(spectro_R, (width - pixels_to_shift, grid_y), (width, grid_y), (30, 30, 30), 1)

                prev_yl_main = audio.draw_entity_trace(main_audio, spectro_L, 'L', width, pixels_to_shift)
                prev_yr_main = audio.draw_entity_trace(main_audio, spectro_R, 'R', width, pixels_to_shift)
                main_audio.prev_y = prev_yl_main if prev_yl_main is not None else prev_yr_main

                prev_yl_mot = audio.draw_entity_trace(motion_audio, spectro_L, 'L', width, pixels_to_shift)
                prev_yr_mot = audio.draw_entity_trace(motion_audio, spectro_R, 'R', width, pixels_to_shift)
                motion_audio.prev_y = prev_yl_mot if prev_yl_mot is not None else prev_yr_mot
                
                spectro_L_disp = spectro_L.copy()
                spectro_R_disp = spectro_R.copy()
                cv2.putText(spectro_L_disp, "Left Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                cv2.putText(spectro_R_disp, "Right Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                top_row = np.hstack((vis_l, vis_r))
                mid_row = np.hstack((depth_view, radar_view))
                bot_row = np.hstack((spectro_L_disp, spectro_R_disp))
                final_ui = np.vstack((top_row, mid_row, bot_row))

            cv2.imshow("Stesis", final_ui)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        audio_sys.close()
        camera.close_cameras(cap_l, cap_r)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()