# Main configurations (Use "v4l2-ctl --list-devices" to find ID)
CAM_ID_LEFT = 0             
CAM_ID_RIGHT = 2            
BASELINE = 13.0             

# For calibration
CHECKERBOARD_DIMS = (9, 6)  
SQUARE_SIZE_CM = 2.5        
AUTO_CAPTURE_DELAY = 2.0    
MOVEMENT_THRESHOLD = 3.0    
POST_CAPTURE_COOLDOWN = 2.0 

# For depth map generation
MIN_DISPARITY = 0
NUM_DISPARITIES = 16*3      
BLOCK_SIZE = 5              

# --- NEW CONFIGURATIONS (Merged from main.py) ---
LEFT_OFFSET = 80             # Pixels to ignore on the left edge of the depth map
RING_UNIT = 100              # Distance in CM between radar rings
SPEED_THRESHOLD_CM = 10.0    # Change in CM per frame to trigger green motion dots
BOUNCE_SWEEP = True             # True = Left-to-Right-to-Left. False = Jump back to start
ENABLE_MOTION_DETECTION = False  # Toggle the green spatial motion dots

# --- SPECTROGRAM & AUDIO CONFIG ---
SPECTRO_HEIGHT = 200             # Pixel height of the bottom graphs
SPECTRO_TIME_HISTORY_SEC = 3.0   # How many seconds the graph width represents
AUDIO_SMOOTHING_COEFF = 0.8      # 0.0 (No smoothing) to 0.99 (Heavy smoothing)
AUDIO_BASE_FREQ = 200.0          # Frequency (Hz) for max distance (300cm)
AUDIO_MAX_FREQ = 1200.0          # Frequency (Hz) for min distance (0cm)

# --- STESIS CONFIG ---
SCALE = 0.5  
USE_WLS = True            

# --- RADAR & SWEEP CONFIG ---
SWEEP_SPEED_SEC = 2.0     
RADAR_FADE = 0.93         
MAX_RADAR_DIST_CM = 300   
FOV_H = 60.0              

# --- AUDIO CONFIGURATIONS ---
PLAY_AUDIO = True                # Toggle to output sound to speakers
AUDIO_SAMPLE_RATE = 44100        # Audio sample rate
AUDIO_CHUNK_SIZE = 1024          # Buffer size for audio chunks

AUDIO_MODE = 'DISTANCE_VERTICAL' # Options: 'DISTANCE_VERTICAL' or 'DISTANCE'
MAX_VOL_DIST_CM = 50.0           # Inner boundary: 100% Volume (Cyan circle)
MIN_VOL_DIST_CM = 250.0          # Outer boundary: 0% Volume (Gray circle)
