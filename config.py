# Main configurations (Use "v4l2-ctl --list-devices" to find ID)
CAM_ID_LEFT = 0             # Left camera ID
CAM_ID_RIGHT = 2            # Right camera ID (Is flipped)
BASELINE = 10.0             # Baseline, distance between the two cameras in CM

# For calibration
CHECKERBOARD_DIMS = (9, 6)  # Checkerboard dimension for calibration
SQUARE_SIZE_CM = 2.5        # Square side size in CM of checkerboard
AUTO_CAPTURE_DELAY = 2.0    # Seconds board must be still to trigger capture
MOVEMENT_THRESHOLD = 3.0    # Max pixel movement allowed to count as "still"
POST_CAPTURE_COOLDOWN = 2.0 # Seconds to wait after capture before scanning again

# For depth map generation - StereoSGBM Tuning (Default)
MIN_DISPARITY = 0
NUM_DISPARITIES = 16*3      # Initial default
BLOCK_SIZE = 5              # Initial default (Lower usually captures finer details but noisier)