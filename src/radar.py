import cv2
import numpy as np
import math
import time
from config import (
    LEFT_OFFSET, MAX_RADAR_DIST_CM, FOV_H, RING_UNIT, RADAR_FADE,
    MAX_VOL_DIST_CM, MIN_VOL_DIST_CM, SCALE,
    SWEEP_SPEED_SEC, BOUNCE_SWEEP, SPEED_THRESHOLD_CM,
    ENABLE_MOTION_DETECTION, BASELINE, MIN_DISPARITY, NUM_DISPARITIES,
    RADAR_RING_COLOR, RADAR_GRID_COLOR, RADAR_TEXT_SCALE
)

def polar_to_cartesian_yaw_corrected(cx, cy, dist_cm, angle_deg, current_yaw, max_dist, canvas_radius):
    # Angle in world coordinates (North/Initial Heading = 0)
    # Global Angle = Yaw + Local Angle
    # If we want North-Up, we plot Global Angle.
    # If we want Head-Up, we plot Local Angle (relative to forward).
    
    # North-Up:
    # angle_deg passed here is usually local angle relative to camera center.
    # We want to plot (Yaw + Local).
    
    global_angle = current_yaw + angle_deg
    
    r = (dist_cm / max_dist) * canvas_radius
    r = min(r, canvas_radius)
    # Standard math: 0 deg is East (Right). we want 0 deg Up (North).
    # So subtract 90.
    angle_rad = math.radians(global_angle - 90)
    x = int(cx + r * math.cos(angle_rad))
    y = int(cy + r * math.sin(angle_rad))
    return x, y

class Radar:
    def __init__(self, width, height, focal_length):
        self.width = width
        self.height = height
        self.focal_length = focal_length
        self.radar_img = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_distances = np.zeros(width)
        self.last_sweep_x = LEFT_OFFSET
        self.sweep_range_width = width - LEFT_OFFSET - 1 
        self.last_frame_time = time.time()
        
    def process(self, disparity, current_yaw, current_pitch, curr_time):
        dt = curr_time - self.last_frame_time
        self.last_frame_time = curr_time
        
        disparity_clean = np.where(disparity > 0, disparity, 0)
        disparity_clean[:, :LEFT_OFFSET] = 0 
        
        # Fade previous sweep points
        self.radar_img = cv2.addWeighted(self.radar_img, RADAR_FADE, np.zeros_like(self.radar_img), 0.0, 0)
        radar_cx, radar_cy = self.width // 2, self.height // 2
        canvas_radius = min(self.width, self.height) // 2 - 10
        
        current_distances = np.zeros(self.width)
        current_y_coords = np.zeros(self.width, dtype=int)
        
        closest_motion_dist = MAX_RADAR_DIST_CM
        motion_x = -1
        motion_y = -1
        
        # --- DISTANCE & MOTION DETECTION ---
        for sx in range(LEFT_OFFSET, self.width):
            col_disp = disparity_clean[:, sx]
            max_y = np.argmax(col_disp)
            max_d = col_disp[max_y]
            
            if max_d > 0:
                dist_cm = (self.focal_length * BASELINE) / max_d
                current_distances[sx] = dist_cm
                current_y_coords[sx] = max_y
                
                if ENABLE_MOTION_DETECTION and self.prev_distances[sx] > 0:
                    speed = abs(dist_cm - self.prev_distances[sx])
                    if speed > SPEED_THRESHOLD_CM:
                        col_angle_local = ((sx / self.width) * FOV_H) - (FOV_H / 2)
                        
                        rx_m, ry_m = polar_to_cartesian_yaw_corrected(
                            radar_cx, radar_cy, dist_cm, col_angle_local, current_yaw, 
                            MAX_RADAR_DIST_CM, canvas_radius
                        )
                        cv2.circle(self.radar_img, (rx_m, ry_m), 4, (0, 255, 0), -1)
                        
                        if dist_cm < closest_motion_dist:
                            closest_motion_dist = dist_cm
                            motion_x = sx
                            motion_y = max_y

        self.prev_distances = current_distances.copy()
        
        # --- SWEEP LOGIC ---
        if BOUNCE_SWEEP:
            cycle_time = 2.0 * SWEEP_SPEED_SEC
            sweep_progress = (curr_time % cycle_time) / SWEEP_SPEED_SEC
            if sweep_progress <= 1.0:
                curr_sweep_x = LEFT_OFFSET + int(sweep_progress * self.sweep_range_width)
            else:
                curr_sweep_x = LEFT_OFFSET + int((2.0 - sweep_progress) * self.sweep_range_width)
        else:
            sweep_progress = (curr_time % SWEEP_SPEED_SEC) / SWEEP_SPEED_SEC
            curr_sweep_x = LEFT_OFFSET + int(sweep_progress * self.sweep_range_width)
            
        if not BOUNCE_SWEEP and curr_sweep_x < self.last_sweep_x:
            sweep_range = list(range(self.last_sweep_x, self.width)) + list(range(LEFT_OFFSET, curr_sweep_x))
        elif BOUNCE_SWEEP and curr_sweep_x < self.last_sweep_x:
            sweep_range = range(self.last_sweep_x, curr_sweep_x, -1)
        else:
            sweep_range = range(self.last_sweep_x, curr_sweep_x)
            
        closest_sweep_dist = MAX_RADAR_DIST_CM
        closest_sweep_y = self.height // 2
        valid_sweep = False

        for sx in sweep_range:
            if sx >= self.width or sx < LEFT_OFFSET: continue
            dist_cm = current_distances[sx]
            
            if dist_cm > 0:
                valid_sweep = True
                y_val = current_y_coords[sx]
                
                if dist_cm < closest_sweep_dist:
                    closest_sweep_dist = dist_cm
                    closest_sweep_y = y_val
                
                g = int(255 * (1.0 - (y_val / (self.height - 1))))
                color = (0, g, 255) # BGR
                
                col_angle_local = ((sx / self.width) * FOV_H) - (FOV_H / 2)
                
                rx, ry = polar_to_cartesian_yaw_corrected(
                    radar_cx, radar_cy, dist_cm, col_angle_local, current_yaw, 
                    MAX_RADAR_DIST_CM, canvas_radius
                )
                cv2.circle(self.radar_img, (rx, ry), 2, color, -1)
                
        self.last_sweep_x = curr_sweep_x
        
        # --- BUILD DISPLAY ---
        radar_display = self.radar_img.copy()
        
        # Radar Rings (Static relative to center)
        for d in range(RING_UNIT, MAX_RADAR_DIST_CM + 1, RING_UNIT):
            r = int((d / MAX_RADAR_DIST_CM) * canvas_radius)
            # Use RADAR_GRID_COLOR (Light Gray) for distance rings
            cv2.circle(radar_display, (radar_cx, radar_cy), r, RADAR_GRID_COLOR, 1)
            
            # Label placement: "next to the rings" (Right side), Scaled text
            if d <= MAX_RADAR_DIST_CM:
                cv2.putText(radar_display, f"{d}cm", (radar_cx + r + 4, radar_cy + 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, RADAR_TEXT_SCALE, (200, 200, 200), 1)
        
        # Volume Boundaries (Cyan)
        r_max_vol = int((MAX_VOL_DIST_CM / MAX_RADAR_DIST_CM) * canvas_radius)
        r_min_vol = int((MIN_VOL_DIST_CM / MAX_RADAR_DIST_CM) * canvas_radius)
        cv2.circle(radar_display, (radar_cx, radar_cy), r_max_vol, RADAR_RING_COLOR, 1)
        cv2.circle(radar_display, (radar_cx, radar_cy), r_min_vol, (100, 100, 100), 1) # Keep outer boundary gray/dim
        
        # FOV Lines - Now Rotated by Yaw!
        radar_overlay = radar_display.copy()
        
        # FOV Logic:
        # The visible area starts after LEFT_OFFSET pixels.
        # We need to calculate the angle that corresponds to the LEFT_OFFSET pixel column.
        angle_per_pixel = FOV_H / self.width
        fov_left_local = -(FOV_H / 2) + (LEFT_OFFSET * angle_per_pixel)
        fov_right_local = (FOV_H / 2)
        
        # Calculate world angles for FOV lines
        x_min, y_min = polar_to_cartesian_yaw_corrected(
            radar_cx, radar_cy, MAX_RADAR_DIST_CM, fov_left_local, current_yaw, MAX_RADAR_DIST_CM, canvas_radius
        )
        x_max, y_max = polar_to_cartesian_yaw_corrected(
            radar_cx, radar_cy, MAX_RADAR_DIST_CM, fov_right_local, current_yaw, MAX_RADAR_DIST_CM, canvas_radius
        )
        
        # Draw FOV lines with RADAR_GRID_COLOR (Light Gray)
        cv2.line(radar_overlay, (radar_cx, radar_cy), (x_min, y_min), RADAR_GRID_COLOR, 2)
        cv2.line(radar_overlay, (radar_cx, radar_cy), (x_max, y_max), RADAR_GRID_COLOR, 2)
        
        # Device Orientation Indicator (Arrow)
        arrow_len = 20
        # If North-Up, and Yaw=0 (North), arrow points Up (-Y). 
        # Standard math 0 is Right (+X). 
        # So we want -90 deg rotation.
        arrow_angle_rad = math.radians(current_yaw - 90)
        arrow_x = int(radar_cx + arrow_len * math.cos(arrow_angle_rad))
        arrow_y = int(radar_cy + arrow_len * math.sin(arrow_angle_rad))
        cv2.arrowedLine(radar_display, (radar_cx, radar_cy), (arrow_x, arrow_y), (0, 0, 255), 2)
        
        cv2.addWeighted(radar_overlay, 0.2, radar_display, 0.8, 0, radar_display)
        
        cv2.circle(radar_display, (radar_cx, radar_cy), 4, (255, 255, 0), -1)
        
        # Sweep Line
        if curr_sweep_x >= LEFT_OFFSET:
            col_angle_local = ((curr_sweep_x / self.width) * FOV_H) - (FOV_H / 2)
            end_x, end_y = polar_to_cartesian_yaw_corrected(
                radar_cx, radar_cy, MAX_RADAR_DIST_CM, col_angle_local, current_yaw, MAX_RADAR_DIST_CM, canvas_radius
            )
            cv2.line(radar_display, (radar_cx, radar_cy), (end_x, end_y), (255, 255, 255), 1)
            
        cv2.putText(radar_display, f"Yaw: {current_yaw:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return radar_display, curr_sweep_x, closest_sweep_dist, closest_sweep_y, closest_motion_dist, motion_x, motion_y, valid_sweep


if __name__ == "__main__":
    import calibrate
    import depth
    import imu
    import camera
    from config import CAM_ID_LEFT, CAM_ID_RIGHT, BASELINE, SCALE
    
    print("--- Radar Debug Mode (North-Up) ---")
    
    (M1, d1, M2, d2, R1, P1, R2, P2, Q, width, height, focal_length) = depth.load_calibration(BASELINE, SCALE)
    
    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (width, height), cv2.CV_16SC2)
    
    stereo, right_matcher, wls_filter = depth.create_stereo_matcher()
    
    radar = Radar(width, height, focal_length)
    mpu = imu.MPU6050()
    
    cap_l, cap_r = camera.init_cameras()
    
    if not cap_l or not cap_r:
        print("Camera init failed")
        exit()
    
    try:
        while True:
            curr_time = time.time()
            ret_l, frame_l, ret_r, frame_r = camera.get_frames(cap_l, cap_r)
            if not ret_l or not ret_r: break
            
            img_l = cv2.resize(frame_l, (width, height), interpolation=cv2.INTER_AREA)
            img_r = cv2.resize(frame_r, (width, height), interpolation=cv2.INTER_AREA)

            rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
            rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)
            
            disparity = depth.compute_depth_map(rect_l, rect_r, stereo, right_matcher, wls_filter)
            depth_view = depth.get_visual_depth(disparity)
            
            # IMU Update
            yaw, pitch, pos_x, pos_y, ax = mpu.update()
            
            # Radar Update (Using Real Yaw)
            radar_view, sw_x, sw_dist, sw_y, mot_dist, mot_x, mot_y, valid = radar.process(disparity, yaw, pitch, curr_time)
            
            # Add sweep line to depth view
            if sw_x >= LEFT_OFFSET:
                cv2.line(depth_view, (sw_x, 0), (sw_x, height), (255, 255, 255), 2)
                
            vis_l = calibrate.draw_overlays(rect_l.copy(), mode='lines')
            vis_r = calibrate.draw_overlays(rect_r.copy(), mode='lines')
            
            top = np.hstack((vis_l, vis_r))
            bot = np.hstack((depth_view, radar_view))
            final = np.vstack((top, bot))
            
            cv2.putText(final, f"POS X: {pos_x:.1f} Y: {pos_y:.1f}", (10, height*2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Radar Debug", final)
            if cv2.waitKey(1) == ord('q'): break
    finally:
        camera.close_cameras(cap_l, cap_r)
        cv2.destroyAllWindows()
