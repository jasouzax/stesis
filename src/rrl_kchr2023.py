import cv2
import numpy as np
import pyaudio
import math
import time
from depth import Depth

class ModernSonar(Depth):
    def __init__(self):
        super().__init__()
        
        # --- Modern Sonar Settings ---
        self.BEEP_DURATION = 0.08       # Fixed length of the beep (80ms)
        self.MIN_SILENCE = 0.05         # Minimum silence between beeps (Close = Fast)
        self.MAX_SILENCE = 1.0          # Maximum silence between beeps (Far = Slow)
        
        # Audio State
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Tracking State
        self.target_dist = float('inf')
        self.target_x_norm = 0.0        # -1.0 (Left) to 1.0 (Right)
        self.target_y_norm = 0.5        # 0.0 (Bottom) to 1.0 (Top)
        self.target_point = None
        
        # Continuous Audio Trackers
        self.beep_timer = 0.0
        self.phase = 0.0

    def setup(self):
        super().setup()
        print("Initializing Modern Beep Repetition Rate Sonar (Katz et al., 2023)...")
        
        try:
            self.stream = self.p.open(format=pyaudio.paFloat32,
                                      channels=2, # Stereo is required for Panning
                                      rate=self.AUDIO_SAMPLE_RATE,
                                      output=True,
                                      frames_per_buffer=self.AUDIO_CHUNK_SIZE,
                                      stream_callback=self._audio_callback)
            self.stream.start_stream()
            print("Stereo audio stream started.")
        except Exception as e:
            print(f"Error initializing audio: {e}")
            self.PLAY_AUDIO = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Vectorized audio generation for stereo panning and variable beep rates."""
        if not self.PLAY_AUDIO or self.target_dist > self.MAX_RADAR_DIST_CM:
            return (np.zeros(frame_count * 2, dtype=np.float32).tobytes(), pyaudio.paContinue)
            
        # 1. Map Depth to Silence Duration (Repetition Rate)
        silence_time = np.interp(self.target_dist, [0, self.MAX_RADAR_DIST_CM], [self.MIN_SILENCE, self.MAX_SILENCE])
        period = self.BEEP_DURATION + silence_time
        
        # 2. Map Elevation (Y) to Pitch (Frequency)
        freq = np.interp(self.target_y_norm, [0, 1], [self.AUDIO_BASE_FREQ, self.AUDIO_MAX_FREQ])
        
        # 3. Map Azimuth (X) to Stereo Pan (Equal Power)
        pan_angle = (self.target_x_norm + 1.0) * (math.pi / 4.0) 
        vol_L = math.cos(pan_angle)
        vol_R = math.sin(pan_angle)
        
        # Generate time array for this chunk
        t = np.arange(frame_count) / self.AUDIO_SAMPLE_RATE
        timers = (self.beep_timer + t) % period
        self.beep_timer = (self.beep_timer + frame_count / self.AUDIO_SAMPLE_RATE) % period
        
        # Create Amplitude Envelope (Avoids digital clicking/popping)
        amp = np.zeros(frame_count, dtype=np.float32)
        on_mask = timers < self.BEEP_DURATION
        fade_in_mask = timers < 0.01
        fade_out_mask = (timers > (self.BEEP_DURATION - 0.01)) & on_mask
        mid_mask = on_mask & ~fade_in_mask & ~fade_out_mask
        
        amp[mid_mask] = 1.0
        amp[fade_in_mask] = timers[fade_in_mask] / 0.01
        amp[fade_out_mask] = (self.BEEP_DURATION - timers[fade_out_mask]) / 0.01
        
        # Generate Sine Wave
        phase_inc = 2 * np.pi * freq / self.AUDIO_SAMPLE_RATE
        phases = self.phase + phase_inc * np.arange(frame_count)
        self.phase = (self.phase + phase_inc * frame_count) % (2 * np.pi)
        
        sines = np.sin(phases) * amp
        
        # Interleave Stereo Channels
        out_data = np.empty((frame_count, 2), dtype=np.float32)
        out_data[:, 0] = sines * vol_L
        out_data[:, 1] = sines * vol_R
        
        return (out_data.flatten().tobytes(), pyaudio.paContinue)

    def loop(self):
        super().loop() 
        
        if self.disparity is None: return

        # Isolate the active field of view based on your config
        valid_disp = self.disparity.copy()
        valid_disp[:, :self.LEFT_OFFSET] = 0

        # Find the absolute closest point in the depth map
        max_disp_val = np.max(valid_disp)

        if max_disp_val > 0:
            # Extract 2D coordinates of the closest obstacle
            y_idx, x_idx = np.unravel_index(np.argmax(valid_disp), valid_disp.shape)
            
            # Calculate physical distance
            dist_cm = (self.focal_length * self.BASELINE) / max_disp_val
            
            # Normalize X for Panning (-1.0 to 1.0)
            active_width = self.width - self.LEFT_OFFSET
            active_x = x_idx - self.LEFT_OFFSET
            self.target_x_norm = (active_x / active_width) * 2.0 - 1.0 
            
            # Normalize Y for Pitch (0.0 to 1.0, inverted so top is 1.0)
            self.target_y_norm = 1.0 - (y_idx / self.height)
            
            # Update state for Audio Thread
            self.target_dist = dist_cm
            self.target_point = (x_idx, y_idx)
        else:
            self.target_dist = float('inf')
            self.target_point = None

        # --- VISUALIZATION ---
        display = self.depth_view.copy()
        
        # Draw a target reticle over the tracked obstacle
        if self.target_point and self.target_dist <= self.MAX_RADAR_DIST_CM:
            x, y = self.target_point
            cv2.circle(display, (x, y), 20, (0, 255, 0), 2)
            cv2.line(display, (x - 30, y), (x + 30, y), (0, 255, 0), 1)
            cv2.line(display, (x, y - 30), (x, y + 30), (0, 255, 0), 1)
            
            # Text layout
            text = f"Dist: {self.target_dist:.1f}cm | Pan: {self.target_x_norm:.1f}"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Combine and display
        rect_l, _ = self.rect_frames
        if rect_l is not None:
             top = np.hstack((rect_l, display))
             cv2.imshow(f"Stesis {self.__class__.__name__}", top)
        else:
             cv2.imshow(f"Stesis {self.__class__.__name__}", display)

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        super().cleanup()

if __name__ == "__main__":
    app = ModernSonar()
    app.run()