try:
    import smbus
except ImportError:
    smbus = None
    print("SMBus not found. MPU6050 will not work.")

import math
import time

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
        
        if abs(gz_deg_s) > 1.5:
            self.yaw += gz_deg_s * dt

        ax, ay, az = self.get_accel()
        # Protect against divide by zero or math domain errors
        try:
            self.pitch = math.atan2(ay, math.sqrt(ax**2 + az**2)) * (180.0 / math.pi)
        except:
            pass
        
        return self.yaw, self.pitch

if __name__ == "__main__":
    mpu = MPU6050()
    print("MPU6050 Live Update (Press Ctrl+C to stop)")
    print("Yaw | Pitch | Accel X | Accel Y | Accel Z")
    try:
        while True:
            yaw, pitch = mpu.update()
            ax, ay, az = mpu.get_accel()
            print(f"\r{yaw:6.2f} | {pitch:6.2f} | {ax:6.2f} | {ay:6.2f} | {az:6.2f}", end="")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
