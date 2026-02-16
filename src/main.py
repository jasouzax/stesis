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
    SPECTRO_HEIGHT, SPECTRO_TIME_HISTORY_SEC, LEFT_OFFSET,
    DOMAIN, HOST
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
        with socketserver.TCPServer((HOST, DOMAIN), Handler) as httpd:
            print(f"Serving web content at http://{HOST}:{DOMAIN} from {web_dir}")
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

    # Stereo Setup
    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (width, height), cv2.CV_16SC2)
    stereo, right_matcher, wls_filter = depth.create_stereo_matcher()

    # Systems
    mpu = imu.MPU6050()
    hand_sys = hand.HandGesture()
    nav_sys = audio.NavSystem(width, height, focal_length, map1_l, map2_l, map1_r, map2_r, stereo, right_matcher, wls_filter)
    
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
                nav_sys.mute()
                
                # Show black screen with text
                inactive_screen = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(inactive_screen, "INACTIVE", (width//2 - 60, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                
                final_ui = inactive_screen

            elif MODE == "HAND":
                nav_sys.mute()
                
                # Process Right Camera (frame_r is already rotated 180 by camera.get_frames)
                hand_view, msg = hand_sys.process(frame_r.copy())
                if msg:
                    print(f"Hand Event: {msg}")
                
                final_ui = hand_view
                
            elif MODE == "NAV":
                final_ui = nav_sys.process(
                    frame_l, frame_r, current_yaw, current_pitch, pos_x, pos_y, dt, curr_time
                )

            cv2.imshow("Stesis", final_ui)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Main Loop Error: {e}")
    finally:
        nav_sys.close()
        hand_sys.close()
        camera.close_cameras(cap_l, cap_r)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
