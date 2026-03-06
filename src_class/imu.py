import cv2
import smbus
import math
import time
import numpy as np
from config import Config

class IMU(Config):

    def __init__(self):
        super().__init__()
        self.bus = None
        
        self.gui = np.zeros((600, 600, 3), dtype=np.uint8) 
        
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0     
        
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0    
        
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        
        # Mahony Filter variables
        self.q = [1.0, 0.0, 0.0, 0.0] # Quaternion [w, x, y, z]
        self.integralFB = [0.0, 0.0, 0.0]
        self.Kp = 2.0   # Proportional gain
        self.Ki = 0.005 # Integral gain
        
        self.last_imu_time = time.time()

    def setup(self):
        super().setup()
        print("Initializing IMU...")
        try:
            self.bus = smbus.SMBus(1)  
            
            # 1. Check WHO_AM_I register (0x75)
            # MPU6050 returns 0x68, MPU6500 returns 0x70
            who_am_i = self.bus.read_byte_data(self.IMU_ADDR, 0x75)
            if who_am_i != 0x70:
                print(f"Warning: WHO_AM_I returned {hex(who_am_i)}. Expected 0x70 for MPU6500.")
            else:
                print("MPU6500 detected successfully.")

            # 2. Wake up MPU6500
            self.bus.write_byte_data(self.IMU_ADDR, 0x6B, 0)
            
            # 3. Configure Gyro to +/- 250 deg/s
            self.bus.write_byte_data(self.IMU_ADDR, 0x1B, 0x00)
            
            # 4. Configure Accel to +/- 2g
            self.bus.write_byte_data(self.IMU_ADDR, 0x1C, 0x00)
            
            print("MPU6500 Initialized.")
        except Exception as e:
            print(f"Failed to initialize MPU6500: {e}")
            self.bus = None
            return True

    def read_word(self, reg):
        if self.bus is None: return 0
        try:
            h = self.bus.read_byte_data(self.IMU_ADDR, reg)
            l = self.bus.read_byte_data(self.IMU_ADDR, reg+1)
            value = (h << 8) + l
            if value >= 0x8000:
                return -((65535 - value) + 1)
            return value
        except:
            return 0

    def get_accel(self):
        # 16384.0 is the LSB/g for +/- 2g range
        ax = self.read_word(0x3B) / 16384.0
        ay = self.read_word(0x3D) / 16384.0
        az = self.read_word(0x3F) / 16384.0
        return ax, ay, az

    def get_gyro(self):
        # 131.0 is the LSB/(deg/s) for +/- 250 deg/s range
        gx = self.read_word(0x43) / 131.0
        gy = self.read_word(0x45) / 131.0
        gz = self.read_word(0x47) / 131.0
        return gx, gy, gz

    def mahony_update(self, gx, gy, gz, ax, ay, az, dt):
        q = self.q
        
        # Convert gyro to radians per second
        gx *= (math.pi / 180.0)
        gy *= (math.pi / 180.0)
        gz *= (math.pi / 180.0)

        # Normalize accelerometer measurement
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm == 0: return
        ax /= norm; ay /= norm; az /= norm

        # Estimated direction of gravity based on quaternion
        vx = 2.0 * (q[1] * q[3] - q[0] * q[2])
        vy = 2.0 * (q[0] * q[1] + q[2] * q[3])
        vz = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]

        # Error is cross product between estimated and measured direction of gravity
        ex = (ay * vz - az * vy)
        ey = (az * vx - ax * vz)
        ez = (ax * vy - ay * vx)

        # Apply integral feedback
        if self.Ki > 0.0:
            self.integralFB[0] += ex * dt
            self.integralFB[1] += ey * dt
            self.integralFB[2] += ez * dt
            gx += self.Ki * self.integralFB[0]
            gy += self.Ki * self.integralFB[1]
            gz += self.Ki * self.integralFB[2]

        # Apply proportional feedback
        gx += self.Kp * ex
        gy += self.Kp * ey
        gz += self.Kp * ez

        # Integrate rate of change of quaternion
        gx *= (0.5 * dt); gy *= (0.5 * dt); gz *= (0.5 * dt)
        qa = q[0]; qb = q[1]; qc = q[2]
        q[0] += (-qb * gx - qc * gy - q[3] * gz)
        q[1] += (qa * gx + qc * gz - q[3] * gy)
        q[2] += (qa * gy - qb * gz + q[3] * gx)
        q[3] += (qa * gz + qb * gy - qc * gx)

        # Normalize quaternion
        norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        self.q = [q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm]
    def loop(self, show=True):
        super().loop()
        if not self.bus:
            print("IMU Bus failed")
            return True

        current_time = time.time()
        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time
        if dt <= 0: return False

        # 1. Read Raw Sensors
        ax, ay, az = self.get_accel()
        gx, gy, gz = self.get_gyro()

        # Swap X and Y axes depending on rotation direction (uncomment if needed)
        ax, ay = ay, ax
        gx, gy = gy, gx

        # 2. Update Orientation Quaternion
        self.mahony_update(gx, gy, gz, ax, ay, az, dt)
        q0, q1, q2, q3 = self.q

        # 3. Convert Quaternion to Euler Angles for UI (SWAPPED PITCH & ROLL)
        # Pitch now uses the standard Roll calculation
        self.pitch = math.atan2(2.0 * (q0*q1 + q2*q3), 1.0 - 2.0 * (q1**2 + q2**2)) * (180.0 / math.pi)
        
        # Roll now uses the standard Pitch calculation with clamping
        roll_val = max(-1.0, min(1.0, 2.0 * (q0*q2 - q3*q1)))
        self.roll = math.asin(roll_val) * (180.0 / math.pi)
        
        # Yaw remains the same
        self.yaw = math.atan2(2.0 * (q0*q3 + q1*q2), 1.0 - 2.0 * (q2**2 + q3**2)) * (180.0 / math.pi)

        # 4. Calculate Expected Gravity Vector
        grav_x = 2.0 * (q1 * q3 - q0 * q2)
        grav_y = 2.0 * (q0 * q1 + q2 * q3)
        grav_z = q0**2 - q1**2 - q2**2 + q3**2

        # 5. Remove Gravity (Local Linear Acceleration)
        lin_ax = ax - grav_x
        lin_ay = ay - grav_y
        lin_az = az - grav_z

        # 6. Rotate Linear Acceleration into Earth Frame
        # e_a = q * a_local * q*
        e_ax = lin_ax*(1 - 2*(q2**2 + q3**2)) + lin_ay*(2*(q1*q2 - q0*q3))   + lin_az*(2*(q0*q2 + q1*q3))
        e_ay = lin_ax*(2*(q1*q2 + q0*q3))   + lin_ay*(1 - 2*(q1**2 + q3**2)) + lin_az*(2*(q2*q3 - q0*q1))
        e_az = lin_ax*(2*(q1*q3 - q0*q2))   + lin_ay*(2*(q0*q1 + q2*q3))   + lin_az*(1 - 2*(q1**2 + q2**2))

        # 7. Apply Deadband to Earth-Frame Acceleration
        acc_x = e_ax if abs(e_ax) > self.ACCEL_DEADBAND else 0.0
        acc_y = e_ay if abs(e_ay) > self.ACCEL_DEADBAND else 0.0
        acc_z = e_az if abs(e_az) > self.ACCEL_DEADBAND else 0.0
        
        # 8. Integrate into Velocity and Position
        scale = 9.8 * dt * 100 # Convert g to cm/s
        
        self.vel_x += acc_x * scale
        self.vel_y += acc_y * scale
        self.vel_z += acc_z * scale
        
        # Apply decay to prevent runaway mathematical drift
        self.vel_x *= self.VELOCITY_DECAY
        self.vel_y *= self.VELOCITY_DECAY
        self.vel_z *= self.VELOCITY_DECAY
        
        self.pos_x += self.vel_x * dt
        self.pos_y += self.vel_y * dt
        self.pos_z += self.vel_z * dt
        
        # 9. Rendering Pseudo-3D
        if show:
            self.gui.fill(30)
            cx, cy, r = 300, 300, 120 

            def project_3d(x, y, z):
                rad_r = math.radians(self.roll)
                rad_p = math.radians(self.pitch)
                rad_y = math.radians(self.yaw)

                # Roll (X)
                y1 = y * math.cos(rad_r) - z * math.sin(rad_r)
                z1 = y * math.sin(rad_r) + z * math.cos(rad_r)
                x1 = x

                # Pitch (Y)
                x2 = x1 * math.cos(rad_p) + z1 * math.sin(rad_p)
                z2 = -x1 * math.sin(rad_p) + z1 * math.cos(rad_p)
                y2 = y1

                # Yaw (Z)
                x3 = x2 * math.cos(rad_y) - y2 * math.sin(rad_y)
                y3 = x2 * math.sin(rad_y) + y2 * math.cos(rad_y)
                z3 = z2

                return int(cx + x3), int(cy - y3)

            cv2.circle(self.gui, (cx, cy), r, (50, 50, 50), -1)
            cv2.circle(self.gui, (cx, cy), r, (150, 150, 150), 2)

            # Draw axes
            px, py = project_3d(r, 0, 0)
            cv2.line(self.gui, (cx, cy), (px, py), (255, 0, 0), 2) # X / Blue
            px, py = project_3d(0, r, 0)
            cv2.line(self.gui, (cx, cy), (px, py), (0, 255, 0), 2) # Y / Green
            px, py = project_3d(0, 0, r)
            cv2.line(self.gui, (cx, cy), (px, py), (0, 0, 255), 2) # Z / Red

            # Velocity Vector
            v_scale = 3.0
            vx_proj, vy_proj = project_3d(self.vel_x * v_scale, self.vel_y * v_scale, self.vel_z * v_scale)
            if abs(self.vel_x) > 0.1 or abs(self.vel_y) > 0.1 or abs(self.vel_z) > 0.1:
                cv2.arrowedLine(self.gui, (cx, cy), (vx_proj, vy_proj), (0, 255, 255), 3, tipLength=0.2)

            # Telemetry
            cv2.putText(self.gui, f"Yaw:   {self.yaw:6.1f} deg", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(self.gui, f"Pitch: {self.pitch:6.1f} deg", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(self.gui, f"Roll:  {self.roll:6.1f} deg", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.putText(self.gui, f"Vel X: {self.vel_x:6.2f} cm/s", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(self.gui, f"Vel Y: {self.vel_y:6.2f} cm/s", (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(self.gui, f"Vel Z: {self.vel_z:6.2f} cm/s", (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            time.sleep(0.01)

if __name__ == "__main__":
    app = IMU()
    app.run()
