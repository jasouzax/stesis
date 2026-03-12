import sys
import os

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Verifying imports...")
try:
    import config
    print("Config imported.")
    import camera
    print("Camera imported.")
    import calibrate
    print("Calibrate imported.")
    import depth
    print("Depth imported.")
    import imu
    print("IMU imported.")
    import radar
    print("Radar imported.")
    import audio
    print("Audio imported.")
    import hand
    print("Hand imported.")
    import main
    print("Main imported.")
    
    print("\nAll modules imported successfully.")
except Exception as e:
    print(f"\nImport Failed: {e}")
    sys.exit(1)
