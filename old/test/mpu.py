from mpu6050 import mpu6050
import time

sensor = mpu6050(0x68)

while True:
    accel_data = sensor.get_accel_data()
    gyro_data = sensor.get_gyro_data()
    
    print(f"Accel X: {accel_data['x']:.2f}, Y: {accel_data['y']:.2f}")
    print(f"Gyro X: {gyro_data['x']:.2f}, Y: {gyro_data['y']:.2f}")
    time.sleep(0.5)
