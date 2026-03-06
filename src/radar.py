import cv2
import numpy as np
import math
import time
from depth import Depth
from imu import IMU

class Radar(Depth, IMU):
    def __init__(self):
        super().__init__() # Calls Depth.__init__, which calls Camera.__init__ -> IMU.__init__ -> Config.__init__
        
        # Radar State
        self.radar_img = None 
        self.prev_distances = None
        self.last_sweep_x = self.LEFT_OFFSET
        self.sweep_range_width = 0
        self.last_radar_time = time.time()
        
        # Output State
        self.radar_view = None
        
        # Audio/Logic State (Exposing for Audio class)
        self.sw_x = 0
        self.sw_dist = 0
        self.sw_y = 0
        self.mot_dist = 0
        self.mot_x = 0
        self.mot_y = 0
        self.valid_sweep = False

    def setup(self):
        super().setup() # Init IMU, Camera, Depth
        print("Initializing Radar...")
        if self.width == 0 or self.height == 0:
            print("Radar setup failed: Width/Height 0")
            return

        self.radar_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.prev_distances = np.zeros(self.width)
        self.sweep_range_width = self.width - self.LEFT_OFFSET - 1 

    @staticmethod
    def polar_to_cartesian_yaw_corrected(cx, cy, dist_cm, angle_deg, current_yaw, max_dist, canvas_radius):
        global_angle = current_yaw + angle_deg
        r = (dist_cm / max_dist) * canvas_radius
        r = min(r, canvas_radius)
        angle_rad = math.radians(global_angle - 90)
        x = int(cx + r * math.cos(angle_rad))
        y = int(cy + r * math.sin(angle_rad))
        return x, y

    def loop(self):
        super().loop() # Get IMU, Frame, Disparity

        curr_time = time.time()
        dt = curr_time - self.last_radar_time
        self.last_radar_time = curr_time
        
        if self.disparity is None: return

        # Radar Logic
        disparity_clean = np.where(self.disparity > 0, self.disparity, 0)
        disparity_clean[:, :self.LEFT_OFFSET] = 0 
        
        self.radar_img = cv2.addWeighted(self.radar_img, self.RADAR_FADE, np.zeros_like(self.radar_img), 0.0, 0)
        radar_cx, radar_cy = self.width // 2, self.height // 2
        canvas_radius = min(self.width, self.height) // 2 - 10
        
        current_distances = np.zeros(self.width)
        current_y_coords = np.zeros(self.width, dtype=int)
        
        closest_motion_dist = self.MAX_RADAR_DIST_CM
        motion_x = -1
        motion_y = -1
        
        # --- DISTANCE & MOTION DETECTION ---
        for sx in range(self.LEFT_OFFSET, self.width):
            col_disp = disparity_clean[:, sx]
            max_y = np.argmax(col_disp)
            max_d = col_disp[max_y]
            
            if max_d > 0:
                dist_cm = (self.focal_length * self.BASELINE) / max_d
                current_distances[sx] = dist_cm
                current_y_coords[sx] = max_y
                
                if self.ENABLE_MOTION_DETECTION and self.prev_distances[sx] > 0:
                    speed = abs(dist_cm - self.prev_distances[sx])
                    if speed > self.SPEED_THRESHOLD_CM:
                        col_angle_local = ((sx / self.width) * self.FOV_H) - (self.FOV_H / 2)
                        rx_m, ry_m = self.polar_to_cartesian_yaw_corrected(
                            radar_cx, radar_cy, dist_cm, col_angle_local, self.yaw, 
                            self.MAX_RADAR_DIST_CM, canvas_radius
                        )
                        cv2.circle(self.radar_img, (rx_m, ry_m), 4, (0, 255, 0), -1)
                        
                        if dist_cm < closest_motion_dist:
                            closest_motion_dist = dist_cm
                            motion_x = sx
                            motion_y = max_y

        self.prev_distances = current_distances.copy()
        
        # --- SWEEP LOGIC ---
        if self.BOUNCE_SWEEP:
            cycle_time = 2.0 * self.SWEEP_SPEED_SEC
            sweep_progress = (curr_time % cycle_time) / self.SWEEP_SPEED_SEC
            if sweep_progress <= 1.0:
                curr_sweep_x = self.LEFT_OFFSET + int(sweep_progress * self.sweep_range_width)
            else:
                curr_sweep_x = self.LEFT_OFFSET + int((2.0 - sweep_progress) * self.sweep_range_width)
        else:
            sweep_progress = (curr_time % self.SWEEP_SPEED_SEC) / self.SWEEP_SPEED_SEC
            curr_sweep_x = self.LEFT_OFFSET + int(sweep_progress * self.sweep_range_width)
            
        if not self.BOUNCE_SWEEP and curr_sweep_x < self.last_sweep_x:
            sweep_range = list(range(self.last_sweep_x, self.width)) + list(range(self.LEFT_OFFSET, curr_sweep_x))
        elif self.BOUNCE_SWEEP and curr_sweep_x < self.last_sweep_x:
            sweep_range = range(self.last_sweep_x, curr_sweep_x, -1)
        else:
            sweep_range = range(self.last_sweep_x, curr_sweep_x)
            
        closest_sweep_dist = self.MAX_RADAR_DIST_CM
        closest_sweep_y = self.height // 2
        valid_sweep = False

        for sx in sweep_range:
            if sx >= self.width or sx < self.LEFT_OFFSET: continue
            dist_cm = current_distances[sx]
            
            if dist_cm > 0:
                valid_sweep = True
                y_val = current_y_coords[sx]
                
                if dist_cm < closest_sweep_dist:
                    closest_sweep_dist = dist_cm
                    closest_sweep_y = y_val
                
                g = int(255 * (1.0 - (y_val / (self.height - 1))))
                color = (0, g, 255) # BGR
                
                col_angle_local = ((sx / self.width) * self.FOV_H) - (self.FOV_H / 2)
                rx, ry = self.polar_to_cartesian_yaw_corrected(
                    radar_cx, radar_cy, dist_cm, col_angle_local, self.yaw, 
                    self.MAX_RADAR_DIST_CM, canvas_radius
                )
                cv2.circle(self.radar_img, (rx, ry), 2, color, -1)
                
        self.last_sweep_x = curr_sweep_x
        
        # Save state for Audio
        self.sw_x = curr_sweep_x
        self.sw_dist = closest_sweep_dist
        self.sw_y = closest_sweep_y
        self.mot_dist = closest_motion_dist
        self.mot_x = motion_x
        self.mot_y = motion_y
        self.valid_sweep = valid_sweep

        # --- BUILD DISPLAY ---
        radar_display = self.radar_img.copy()
        
        # Radar Rings
        for d in range(self.RING_UNIT, self.MAX_RADAR_DIST_CM + 1, self.RING_UNIT):
            r = int((d / self.MAX_RADAR_DIST_CM) * canvas_radius)
            cv2.circle(radar_display, (radar_cx, radar_cy), r, self.RADAR_GRID_COLOR, 1)
            if d <= self.MAX_RADAR_DIST_CM:
                cv2.putText(radar_display, f"{d}cm", (radar_cx + r + 4, radar_cy + 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, self.RADAR_TEXT_SCALE, (200, 200, 200), 1)
        
        # Volume Boundaries
        r_max_vol = int((self.MAX_VOL_DIST_CM / self.MAX_RADAR_DIST_CM) * canvas_radius)
        r_min_vol = int((self.MIN_VOL_DIST_CM / self.MAX_RADAR_DIST_CM) * canvas_radius)
        cv2.circle(radar_display, (radar_cx, radar_cy), r_max_vol, self.RADAR_RING_COLOR, 1)
        cv2.circle(radar_display, (radar_cx, radar_cy), r_min_vol, (100, 100, 100), 1)
        
        # FOV Lines
        angle_per_pixel = self.FOV_H / self.width
        fov_left_local = -(self.FOV_H / 2) + (self.LEFT_OFFSET * angle_per_pixel)
        fov_right_local = (self.FOV_H / 2)
        
        x_min, y_min = self.polar_to_cartesian_yaw_corrected(
            radar_cx, radar_cy, self.MAX_RADAR_DIST_CM, fov_left_local, self.yaw, self.MAX_RADAR_DIST_CM, canvas_radius
        )
        x_max, y_max = self.polar_to_cartesian_yaw_corrected(
            radar_cx, radar_cy, self.MAX_RADAR_DIST_CM, fov_right_local, self.yaw, self.MAX_RADAR_DIST_CM, canvas_radius
        )
        
        radar_overlay = radar_display.copy()
        cv2.line(radar_overlay, (radar_cx, radar_cy), (x_min, y_min), self.RADAR_GRID_COLOR, 2)
        cv2.line(radar_overlay, (radar_cx, radar_cy), (x_max, y_max), self.RADAR_GRID_COLOR, 2)
        
        # Arrow
        arrow_len = 20
        arrow_angle_rad = math.radians(self.yaw - 90)
        arrow_x = int(radar_cx + arrow_len * math.cos(arrow_angle_rad))
        arrow_y = int(radar_cy + arrow_len * math.sin(arrow_angle_rad))
        cv2.arrowedLine(radar_display, (radar_cx, radar_cy), (arrow_x, arrow_y), (0, 0, 255), 2)
        
        cv2.addWeighted(radar_overlay, 0.2, radar_display, 0.8, 0, radar_display)
        cv2.circle(radar_display, (radar_cx, radar_cy), 4, (255, 255, 0), -1)
        
        # Sweep Line
        if curr_sweep_x >= self.LEFT_OFFSET:
            col_angle_local = ((curr_sweep_x / self.width) * self.FOV_H) - (self.FOV_H / 2)
            end_x, end_y = self.polar_to_cartesian_yaw_corrected(
                radar_cx, radar_cy, self.MAX_RADAR_DIST_CM, col_angle_local, self.yaw, self.MAX_RADAR_DIST_CM, canvas_radius
            )
            cv2.line(radar_display, (radar_cx, radar_cy), (end_x, end_y), (255, 255, 255), 1)
            
        cv2.putText(radar_display, f"Yaw: {self.yaw:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(radar_display, f"POS X: {self.pos_x:.1f} Y: {self.pos_y:.1f}", (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.radar_view = radar_display

if __name__ == "__main__":
    class RadarView(Radar):
        def loop(self):
            super().loop()
            if self.radar_view is not None:
                # Visualization (Depth + Radar)
                # Need frames from Camera/Depth
                rect_l, rect_r = self.rect_frames # From Depth.loop
                vis_l = self.draw_overlays(rect_l.copy(), mode='lines')
                vis_r = self.draw_overlays(rect_r.copy(), mode='lines')
                
                # Add sweep to depth view
                depth_disp = self.depth_view.copy()
                if self.sw_x >= self.LEFT_OFFSET:
                    cv2.line(depth_disp, (self.sw_x, 0), (self.sw_x, self.height), (255, 255, 255), 2)
                
                top = np.hstack((vis_l, vis_r))
                # Resize bottom to match top?
                # bot = np.hstack((depth_disp, self.radar_view))
                # top is 2*width. bot is 2*width.
                bot = np.hstack((depth_disp, self.radar_view))
                final = np.vstack((top, bot))
                
                cv2.imshow("Radar Class Test", final)
    
    app = RadarView()
    app.run()
