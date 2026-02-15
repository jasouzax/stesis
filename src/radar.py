import cv2
import numpy as np
import math
import time
from config import (
    LEFT_OFFSET, MAX_RADAR_DIST_CM, FOV_H, RING_UNIT, RADAR_FADE,
    MAX_VOL_DIST_CM, MIN_VOL_DIST_CM, SCALE,
    SWEEP_SPEED_SEC, BOUNCE_SWEEP, SPEED_THRESHOLD_CM,
    ENABLE_MOTION_DETECTION, BASELINE, MIN_DISPARITY, NUM_DISPARITIES
)

def polar_to_cartesian(cx, cy, dist_cm, angle_deg, max_dist, canvas_radius):
    r = (dist_cm / max_dist) * canvas_radius
    r = min(r, canvas_radius)
    angle_rad = math.radians(angle_deg - 90)
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
                        global_angle = current_yaw + col_angle_local
                        rx_m, ry_m = polar_to_cartesian(radar_cx, radar_cy, dist_cm, global_angle, MAX_RADAR_DIST_CM, canvas_radius)
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
                
                # Radar color coding for vertical placement: High (Y=0) -> Yellow, Low (Y=height) -> Red
                g = int(255 * (1.0 - (y_val / (self.height - 1))))
                color = (0, g, 255) # BGR
                
                col_angle_local = ((sx / self.width) * FOV_H) - (FOV_H / 2)
                global_angle = current_yaw + col_angle_local
                rx, ry = polar_to_cartesian(radar_cx, radar_cy, dist_cm, global_angle, MAX_RADAR_DIST_CM, canvas_radius)
                cv2.circle(self.radar_img, (rx, ry), 2, color, -1)
                
        self.last_sweep_x = curr_sweep_x
        
        # --- BUILD DISPLAY ---
        radar_display = self.radar_img.copy()
        
        # Radar Rings
        for d in range(RING_UNIT, MAX_RADAR_DIST_CM + 1, RING_UNIT):
            r = int((d / MAX_RADAR_DIST_CM) * canvas_radius)
            cv2.circle(radar_display, (radar_cx, radar_cy), r, (40, 40, 40), 1)
            if d == MAX_RADAR_DIST_CM:
                cv2.putText(radar_display, f"{d}cm", (radar_cx + 5, radar_cy - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Volume Boundaries
        r_max_vol = int((MAX_VOL_DIST_CM / MAX_RADAR_DIST_CM) * canvas_radius)
        r_min_vol = int((MIN_VOL_DIST_CM / MAX_RADAR_DIST_CM) * canvas_radius)
        cv2.circle(radar_display, (radar_cx, radar_cy), r_max_vol, (255, 255, 0), 1)
        cv2.circle(radar_display, (radar_cx, radar_cy), r_min_vol, (100, 100, 100), 1)
        
        # FOV Lines
        radar_overlay = radar_display.copy()
        fov_min_angle = current_yaw - (FOV_H / 2)
        fov_max_angle = current_yaw + (FOV_H / 2)
        x_min, y_min = polar_to_cartesian(radar_cx, radar_cy, MAX_RADAR_DIST_CM, fov_min_angle, MAX_RADAR_DIST_CM, canvas_radius)
        x_max, y_max = polar_to_cartesian(radar_cx, radar_cy, MAX_RADAR_DIST_CM, fov_max_angle, MAX_RADAR_DIST_CM, canvas_radius)
        cv2.line(radar_overlay, (radar_cx, radar_cy), (x_min, y_min), (255, 255, 255), 2)
        cv2.line(radar_overlay, (radar_cx, radar_cy), (x_max, y_max), (255, 255, 255), 2)
        cv2.addWeighted(radar_overlay, 0.2, radar_display, 0.8, 0, radar_display)
        
        cv2.circle(radar_display, (radar_cx, radar_cy), 4, (255, 255, 0), -1)
        
        # Sweep Line
        if curr_sweep_x >= LEFT_OFFSET:
            sweep_global_angle = current_yaw + (((curr_sweep_x / self.width) * FOV_H) - (FOV_H / 2))
            end_x, end_y = polar_to_cartesian(radar_cx, radar_cy, MAX_RADAR_DIST_CM, sweep_global_angle, MAX_RADAR_DIST_CM, canvas_radius)
            cv2.line(radar_display, (radar_cx, radar_cy), (end_x, end_y), (255, 255, 255), 1)
            
        cv2.putText(radar_display, f"Yaw: {current_yaw:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return radar_display, curr_sweep_x, closest_sweep_dist, closest_sweep_y, closest_motion_dist, motion_x, motion_y, valid_sweep


if __name__ == "__main__":
    import calibrate
    import depth
    from config import CAM_ID_LEFT, CAM_ID_RIGHT, BASELINE, SCALE
    
    print("--- Radar Debug Mode ---")
    
    (M1, d1, M2, d2, R1, P1, R2, P2, Q, width, height, focal_length) = depth.load_calibration(BASELINE, SCALE)
    
    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (width, height), cv2.CV_16SC2)
    
    stereo, right_matcher, wls_filter = depth.create_stereo_matcher()
    
    radar = Radar(width, height, focal_length)
    
    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)
    
    while True:
        curr_time = time.time()
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r: break
        
        frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
        img_l = cv2.resize(frame_l, (width, height), interpolation=cv2.INTER_AREA)
        img_r = cv2.resize(frame_r, (width, height), interpolation=cv2.INTER_AREA)

        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)
        
        disparity = depth.compute_depth_map(rect_l, rect_r, stereo, right_matcher, wls_filter)
        depth_view = depth.get_visual_depth(disparity)
        
        # Radar Update (Using 0,0 for yaw/pitch for debug)
        radar_view, sw_x, sw_dist, sw_y, mot_dist, mot_x, mot_y, valid = radar.process(disparity, 0.0, 0.0, curr_time)
        
        # Add sweep line to depth view
        if sw_x >= LEFT_OFFSET:
            cv2.line(depth_view, (sw_x, 0), (sw_x, height), (255, 255, 255), 2)
            
        vis_l = calibrate.draw_overlays(rect_l.copy(), mode='lines')
        vis_r = calibrate.draw_overlays(rect_r.copy(), mode='lines')
        
        # Layout: TopL=vis_l, TopR=vis_r
        # BotL=depth_view, BotR=radar_view
        top = np.hstack((vis_l, vis_r))
        bot = np.hstack((depth_view, radar_view))
        final = np.vstack((top, bot))
        
        cv2.imshow("Radar Debug", final)
        if cv2.waitKey(1) == ord('q'): break
        
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
