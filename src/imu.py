try:
    import smbus
except ImportError:
    smbus = None
    print("SMBus not found. MPU6050 will not work.")

import math
import time

# Simple deadband filter for accelerometer noise
ACCEL_DEADBAND = 0.05 
# Friction/Decay factor to stop drift when no motion
VELOCITY_DECAY = 0.95

class MPU6050:
    def __init__(self, address=0x68):
        self.address = address
        try:
            self.bus = smbus.SMBus(1)  
            self.bus.write_byte_data(self.address, 0x6B, 0)  
        except Exception as e:
            print(f"Failed to initialize MPU6050: {e}")
            self.bus = None
            
        self.yaw = 0.0
        self.pitch = 0.0
        
        # Position tracking
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        
        self.last_time = time.time()

    def read_word(self, reg):
        if self.bus is None: return 0
        try:
            h = self.bus.read_byte_data(self.address, reg)
            l = self.bus.read_byte_data(self.address, reg+1)
            value = (h << 8) + l
            if value >= 0x8000:
                return -((65535 - value) + 1)
            return value
        except:
            return 0

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
        
        # Yaw integration
        if abs(gz_deg_s) > 1.5:
            self.yaw += gz_deg_s * dt

        # Accelerometer / Pitch
        ax, ay, az = self.get_accel()
        try:
            self.pitch = math.atan2(ay, math.sqrt(ax**2 + az**2)) * (180.0 / math.pi)
        except:
            pass
        
        # Position tracking (Experimental Dead Reckoning)
        # Assuming sensor is mounted such that:
        # X+ is Forward (along camera view)
        # Y+ is Horizontal (Left/Right)
        # This depends heavily on mounting orientation!
        # Standard MPU6050: X/Y are planar. Z is up.
        
        # Check deadband
        acc_x = ax if abs(ax) > ACCEL_DEADBAND else 0.0
        acc_y = ay if abs(ay) > ACCEL_DEADBAND else 0.0
        
        # Integrate acceleration to velocity (m/s assuming 1g = 9.8m/s^2 approx)
        # Scaling factor: 9.8 * dt sets it to m/s
        scale = 9.8 * dt * 100 # cm/s
        
        self.vel_x += acc_x * scale
        self.vel_y += acc_y * scale
        
        # Apply friction/decay to reduce drift
        self.vel_x *= VELOCITY_DECAY
        self.vel_y *= VELOCITY_DECAY
        
        # Integrate velocity to position (cm)
        self.pos_x += self.vel_x * dt
        self.pos_y += self.vel_y * dt
        
        return self.yaw, self.pitch, self.pos_x, self.pos_y, ax

if __name__ == "__main__":
    mpu = MPU6050()
    print("MPU6050 Live Update (Press Ctrl+C to stop)")
    print("Yaw | Pitch | Pos X | Pos Y")
    try:
        while True:
            yaw, pitch, x, y = mpu.update()
            print(f"\r{yaw:6.2f} | {pitch:6.2f} | {x:6.2f} | {y:6.2f}", end="")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
