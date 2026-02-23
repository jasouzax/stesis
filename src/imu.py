try:
    import smbus
except ImportError:
    smbus = None
    print("SMBus not found. MPU6050 will not work.")

import math
import time
from config import Config

class IMU(Config):
    # Simple deadband filter for accelerometer noise
    ACCEL_DEADBAND = 0.05 
    # Friction/Decay factor to stop drift when no motion
    VELOCITY_DECAY = 0.95

    def __init__(self):
        super().__init__()
        self.bus = None
        self.address = 0x68
        
        self.yaw = 0.0
        self.pitch = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.ax = 0.0 # Accel X for state machine
        
        self.last_imu_time = time.time()

    def setup(self):
        super().setup() # Call next setup in MRO
        print("Initializing IMU...")
        if smbus:
            try:
                self.bus = smbus.SMBus(1)  
                self.bus.write_byte_data(self.address, 0x6B, 0)
                print("MPU6050 Initialized.")
            except Exception as e:
                print(f"Failed to initialize MPU6050: {e}")
                self.bus = None
        else:
            print("SMBus module missing (not on Pi?). using dummy IMU.")

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

    def loop(self):
        super().loop()
        
        if not self.bus:
            # Simulation / Fallback
            return

        gz = self.read_word(0x47)
        gz_deg_s = gz / 131.0  
        
        current_time = time.time()
        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time
        
        # Yaw integration
        if abs(gz_deg_s) > 1.5:
            self.yaw += gz_deg_s * dt

        # Accelerometer / Pitch
        ax, ay, az = self.get_accel()
        self.ax = ax 
        try:
            self.pitch = math.atan2(ay, math.sqrt(ax**2 + az**2)) * (180.0 / math.pi)
        except:
            pass
        
        # Position tracking 
        acc_x = ax if abs(ax) > self.ACCEL_DEADBAND else 0.0
        acc_y = ay if abs(ay) > self.ACCEL_DEADBAND else 0.0
        
        scale = 9.8 * dt * 100 # cm/s
        
        self.vel_x += acc_x * scale
        self.vel_y += acc_y * scale
        
        self.vel_x *= self.VELOCITY_DECAY
        self.vel_y *= self.VELOCITY_DECAY
        
        self.pos_x += self.vel_x * dt
        self.pos_y += self.vel_y * dt

if __name__ == "__main__":
    class IMUView(IMU):
        def loop(self):
            super().loop()
            print(f"\rYaw: {self.yaw:6.2f} | Pitch: {self.pitch:6.2f} | X: {self.pos_x:6.2f} | Y: {self.pos_y:6.2f}", end="")
            time.sleep(0.05)
            
    app = IMUView()
    app.run()
