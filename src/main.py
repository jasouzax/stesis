import cv2
import numpy as np
import time
import sys

# Import custom modules
from config import (
    CAM_ID_LEFT, CAM_ID_RIGHT, BASELINE, SCALE,
    AUDIO_MODE, AUDIO_BASE_FREQ, AUDIO_MAX_FREQ,
    MAX_VOL_DIST_CM, MIN_VOL_DIST_CM, MAX_RADAR_DIST_CM,
    LEFT_OFFSET, SPECTRO_HEIGHT, SPECTRO_TIME_HISTORY_SEC
)
import imu
import calibrate
import depth
import radar
import audio

def main():
    print("Initializing Stesis...")
    
    # 1. Load Calibration
    try:
        (M1, d1, M2, d2, R1, P1, R2, P2, Q, width, height, focal_length) = depth.load_calibration(BASELINE, SCALE)
    except Exception as e:
        print(f"Calibration Error: {e}")
        return

    # 2. Init Cameras
    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)
    cap_l.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_r.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 3. Init Maps & Matcher
    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (width, height), cv2.CV_16SC2)
    stereo, right_matcher, wls_filter = depth.create_stereo_matcher()

    # 4. Init Components
    mpu = imu.MPU6050()
    radar_sys = radar.Radar(width, height, focal_length)
    audio_sys = audio.AudioManager()
    
    main_audio = audio.AudioEntity(color_L=(255, 255, 0), color_R=(255, 255, 0)) 
    motion_audio = audio.AudioEntity(color_L=(0, 255, 0), color_R=(0, 255, 0))
    audio_sys.add_entity(main_audio)
    audio_sys.add_entity(motion_audio)

    # 5. UI Buffers
    spectro_L = np.zeros((SPECTRO_HEIGHT, width, 3), dtype=np.uint8)
    spectro_R = np.zeros((SPECTRO_HEIGHT, width, 3), dtype=np.uint8)
    last_frame_time = time.time()

    print("System Ready. Press 'q' to exit.")

    try:
        while True:
            curr_time = time.time()
            dt = curr_time - last_frame_time
            last_frame_time = curr_time

            # Read & Rectify
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r: break

            frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
            img_l = cv2.resize(frame_l, (width, height), interpolation=cv2.INTER_AREA)
            img_r = cv2.resize(frame_r, (width, height), interpolation=cv2.INTER_AREA)

            rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

            # Process
            disparity = depth.compute_depth_map(rect_l, rect_r, stereo, right_matcher, wls_filter)
            current_yaw, current_pitch = mpu.update()
            
            radar_view, sw_x, sw_dist, sw_y, mot_dist, mot_x, mot_y, valid_sweep = radar_sys.process(
                disparity, current_yaw, current_pitch, curr_time
            )

            # Audio Logic - Motion
            if mot_x != -1:
                # Calculate Frequency
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

            # Visualization
            vis_l = calibrate.draw_overlays(rect_l.copy())
            vis_r = calibrate.draw_overlays(rect_r.copy())
            depth_view = depth.get_visual_depth(disparity)
            
            # Show sweep line on depth view
            if sw_x >= LEFT_OFFSET:
                cv2.line(depth_view, (sw_x, 0), (sw_x, height), (255, 255, 255), 2)

            # Spectrograms
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
            
            # Text Overlays
            spectro_L_disp = spectro_L.copy()
            spectro_R_disp = spectro_R.copy()
            cv2.putText(spectro_L_disp, "Left Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(spectro_R_disp, "Right Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # Build UI
            top_row = np.hstack((vis_l, vis_r))
            mid_row = np.hstack((depth_view, radar_view))
            bot_row = np.hstack((spectro_L_disp, spectro_R_disp))
            final_ui = np.vstack((top_row, mid_row, bot_row))

            cv2.imshow("Stesis Navigation Matrix", final_ui)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        audio_sys.close()
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
