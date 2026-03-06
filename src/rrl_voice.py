import cv2
import numpy as np
import pyaudio
import time
from camera import Camera

class Voice(Camera):
    def __init__(self):
        super().__init__()
        
        # --- The vOICe Core Settings ---
        self.SCAN_TIME_SEC = 1.0        # 1-second sweep (Standard vOICe default)
        self.FREQ_MIN = 500.0           # Lowest frequency (Bottom of image)
        self.FREQ_MAX = 5000.0          # Highest frequency (Top of image)
        self.VOICE_W = 64               # Width resolution for scanning
        self.VOICE_H = 64               # Height resolution (number of oscillators)
        
        # Audio State
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.scan_start_time = time.time()
        
        # Continuous phase tracking to prevent audio popping between chunks
        self.phases = np.zeros(self.VOICE_H, dtype=np.float32)
        
        # Visual State
        self.current_frame_gray = np.zeros((self.VOICE_H, self.VOICE_W), dtype=np.float32)
        self.scan_line_x = 0

    def setup(self):
        super().setup() # Initializes left/right cameras via Camera.setup()
        print(f"Initializing Original The vOICe Algorithm ({self.VOICE_W}x{self.VOICE_H})...")
        
        # Pre-calculate logarithmic frequency table (Top = High Pitch, Bottom = Low Pitch)
        self.freqs = np.zeros(self.VOICE_H, dtype=np.float32)
        for y in range(self.VOICE_H):
            # y=0 is top (exponent=1 -> FREQ_MAX), y=H-1 is bottom (exponent=0 -> FREQ_MIN)
            norm_y = (self.VOICE_H - 1 - y) / (self.VOICE_H - 1)
            self.freqs[y] = self.FREQ_MIN * (self.FREQ_MAX / self.FREQ_MIN) ** norm_y

        # Initialize PyAudio
        try:
            self.stream = self.p.open(format=pyaudio.paFloat32,
                                      channels=1,
                                      rate=self.AUDIO_SAMPLE_RATE,
                                      output=True,
                                      frames_per_buffer=self.AUDIO_CHUNK_SIZE,
                                      stream_callback=self._audio_callback)
            self.stream.start_stream()
            print("Audio stream started.")
        except Exception as e:
            print(f"Error initializing audio: {e}")
            self.PLAY_AUDIO = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Generates continuous sum-of-sines based on the current grayscale column."""
        if not self.PLAY_AUDIO:
            return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)

        # 1. Determine which column we are currently scanning based on time
        current_time = time.time()
        elapsed = (current_time - self.scan_start_time) % self.SCAN_TIME_SEC
        col_idx = int((elapsed / self.SCAN_TIME_SEC) * self.VOICE_W)
        self.scan_line_x = col_idx # Save for visualization

        # 2. Fetch the brightness values for this column (0.0 to 1.0)
        # We apply a slight exponential curve (** 2) to enhance visual contrast in audio
        amplitudes = (self.current_frame_gray[:, col_idx]) ** 2 

        # 3. Vectorized phase generation for this specific chunk
        phase_inc = 2 * np.pi * self.freqs / self.AUDIO_SAMPLE_RATE
        sample_indices = np.arange(frame_count)
        
        # Matrix of phases: shape (VOICE_H, frame_count)
        phase_matrix = self.phases[:, np.newaxis] + phase_inc[:, np.newaxis] * sample_indices
        
        # Update starting phases for the *next* chunk to maintain continuity
        self.phases = (self.phases + phase_inc * frame_count) % (2 * np.pi)

        # 4. Synthesize sound: Multiply sine waves by pixel brightness
        sines = np.sin(phase_matrix) * amplitudes[:, np.newaxis]
        
        # Sum all rows together and scale down to prevent digital clipping
        mix = np.sum(sines, axis=0) / (self.VOICE_H / 4.0)
        mix = np.clip(mix, -1.0, 1.0)

        return (mix.astype(np.float32).tobytes(), pyaudio.paContinue)

    def loop(self):
        super().loop(show=False) # Get raw frames from Camera
        
        frame_l = self.frames[0]
        if frame_l is None: return

        # Convert left camera to Grayscale (Original vOICe ignores color/depth)
        gray = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        
        # Downsample to the vOICe resolution (e.g., 64x64)
        small_gray = cv2.resize(gray, (self.VOICE_W, self.VOICE_H), interpolation=cv2.INTER_AREA)
        
        # Normalize to 0.0 - 1.0 for the audio thread
        self.current_frame_gray = small_gray / 255.0

        # --- VISUALIZATION ---
        # Display the downsampled feed being read by the algorithm
        display_img = cv2.cvtColor(small_gray, cv2.COLOR_GRAY2BGR)
        
        # Scale it up so it's visible on a monitor
        scale_factor = 8
        display_large = cv2.resize(display_img, (self.VOICE_W * scale_factor, self.VOICE_H * scale_factor), interpolation=cv2.INTER_NEAREST)
        
        # Draw the sweeping scan line
        scan_x_large = self.scan_line_x * scale_factor
        cv2.line(display_large, (scan_x_large, 0), (scan_x_large, self.VOICE_H * scale_factor), (0, 0, 255), 2)
        
        self.gui = display_large

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        super().cleanup()

if __name__ == "__main__":
    app = Voice()
    app.run()