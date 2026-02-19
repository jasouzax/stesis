import cv2
import numpy as np

class Config:
    # MAIN
    DOMAIN = 8080               # Port of server
    HOST = '0.0.0.0'            # Host of server

    # CAMERA: camera.py
    CAM_ID_LEFT = 0             # /dev/video{ID} of left camera
    CAM_ID_RIGHT = 2            # /dev/video{ID} of right camera

    # CALIBRATION
    BASELINE = 13.0             # Distance between left and right camera in CM
    CHECKERBOARD_DIMS = (9, 6)  # Calibration board dimensions
    SQUARE_SIZE_CM = 2.5        # Calibration board square size in CM
    AUTO_CAPTURE_DELAY = 2.0    # Automatic calibration capture delay in Seconds
    MOVEMENT_THRESHOLD = 3.0    # Movement treshold to stop/break automatic calibration in Seconds
    POST_CAPTURE_COOLDOWN = 2.0 # Cooldown after capture to continue automatic capture in Seconds

    # DEPTHMAP
    SCALE = 0.5                 # Camera scaled to improve performance
    USE_WLS = True              # Enable WLS for cleaner depth map?
    MIN_DISPARITY = 0           #
    NUM_DISPARITIES = 16*3      #
    BLOCK_SIZE = 5              #

    # RADAR
    LEFT_OFFSET = 80            # Pixels to ignore on the left edge of the depth map
    RING_UNIT = 100             # Distance between radar rings in CM
    SPEED_THRESHOLD_CM = 10.0   # Change per frame to trigger green motion dots in CM
    BOUNCE_SWEEP = True         # Radar bouce after hitting sides instead of restarting?
    ENABLE_MOTION_DETECTION = False # Include motion detection in radar?
    SWEEP_SPEED_SEC = 2.0       # Time taken for a single sweep from both ends in Seconds
    RADAR_FADE = 0.93
    MAX_RADAR_DIST_CM = 300
    FOV_H = 60.0              
    RADAR_RING_COLOR = (255, 255, 0) # Cyan (BGR) - For Max/Min Volume Circles
    RADAR_GRID_COLOR = (150, 150, 150) # Light Gray - For grid rings and FOV lines
    RADAR_TEXT_SCALE = 0.3      # Text size multiplier

    # AUDIO SONIFICATION
    SPECTRO_HEIGHT = 200             # Pixel height of the bottom graphs
    SPECTRO_TIME_HISTORY_SEC = 3.0   # How many seconds the graph width represents
    AUDIO_SMOOTHING_COEFF = 0.8      # 0.0 (No smoothing) to 0.99 (Heavy smoothing)
    AUDIO_BASE_FREQ = 200.0          # Frequency (Hz) for max distance (300cm)
    AUDIO_MAX_FREQ = 1200.0          # Frequency (Hz) for min distance (0cm)
    PLAY_AUDIO = True                # Toggle to output sound to speakers
    AUDIO_SAMPLE_RATE = 44100        # Audio sample rate
    AUDIO_CHUNK_SIZE = 1024          # Buffer size for audio chunks
    AUDIO_MODE = 'DISTANCE_VERTICAL' # Options: 'DISTANCE_VERTICAL' or 'DISTANCE'
    MAX_VOL_DIST_CM = 50.0           # Inner boundary: 100% Volume (Cyan circle)
    MIN_VOL_DIST_CM = 250.0          # Outer boundary: 0% Volume (Gray circle)

    # SYSTEM
    gui = np.zeros((200, 400, 3), dtype=np.uint8)
    key = 0

    def __init__(self):
        pass

    # Intialize resources
    def setup(self):
        pass

    # Repeated every frame
    def loop(self):
        pass

    # On exit
    def cleanup(self):
        cv2.destroyAllWindows()

    # Main function
    def run(self):
        print(f"Starting {self.__class__.__name__}...")
        self.setup()
        try:
            while not self.loop():
                self.key = cv2.waitKey(1) & 0xFF
                cv2.imshow(f"Stesis {self.__class__.__name__}", self.gui)
                if self.key == ord('q'): break
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.cleanup()
            print(f"{self.__class__.__name__} cleanup done.")

if __name__ == "__main__":
    app = Config()
    app.run()
