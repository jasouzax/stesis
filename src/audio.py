import numpy as np
import cv2
import math
import time
from config import (
    AUDIO_MODE, AUDIO_BASE_FREQ, AUDIO_MAX_FREQ, AUDIO_SMOOTHING_COEFF,
    MAX_VOL_DIST_CM, MIN_VOL_DIST_CM, MAX_RADAR_DIST_CM, LEFT_OFFSET
)

try:
    import pyaudio
except ImportError:
    pyaudio = None
    print("PyAudio not found. Audio will be disabled.")

class AudioEntity:
    def __init__(self, color_L, color_R):
        self.freq = AUDIO_BASE_FREQ
        self.amp_L = 0.0
        self.amp_R = 0.0
        self.color_L = color_L
        self.color_R = color_R
        self.prev_y = None
        self.phase = 0.0 

    def update(self, target_freq, target_amp_L, target_amp_R):
        # Smoothing
        if AUDIO_SMOOTHING_COEFF <= 0:
            self.freq = target_freq
            self.amp_L = target_amp_L
            self.amp_R = target_amp_R
        else:
            self.freq = (self.freq * AUDIO_SMOOTHING_COEFF) + (target_freq * (1.0 - AUDIO_SMOOTHING_COEFF))
            self.amp_L = (self.amp_L * AUDIO_SMOOTHING_COEFF) + (target_amp_L * (1.0 - AUDIO_SMOOTHING_COEFF))
            self.amp_R = (self.amp_R * AUDIO_SMOOTHING_COEFF) + (target_amp_R * (1.0 - AUDIO_SMOOTHING_COEFF))

class AudioManager:
    def __init__(self, rate=44100, chunk=1024):
        self.RATE = rate
        self.CHUNK = chunk
        self.entities = []
        self.stream = None
        self.p = None
        
        if pyaudio:
            try:
                self.p = pyaudio.PyAudio()
                self.stream = self.p.open(format=pyaudio.paFloat32,
                                          channels=2,
                                          rate=self.RATE,
                                          output=True,
                                          stream_callback=self._callback)
            except Exception as e:
                print(f"Audio Output Error: {e}")

    def add_entity(self, entity):
        self.entities.append(entity)

    def _callback(self, in_data, frame_count, time_info, status):
        out_data = np.zeros((frame_count, 2), dtype=np.float32)
        t = np.arange(frame_count) / self.RATE
        
        for entity in self.entities:
            # Generate sine wave
            # Note: Phase continuity is important for smooth audio.
            # Simple approach: standard sine with phase accumulation
            # For this simple demo, we just use freq*t but that pops on freq change.
            # Better: phase += freq * dt
            
            # Vectorized phase accumulation is tricky in callback without state per sample.
            # But since frame_count is small, we can just do:
            phase_increment = 2 * np.pi * entity.freq / self.RATE
            phases = entity.phase + np.arange(frame_count) * phase_increment
            entity.phase = (phases[-1] + phase_increment) % (2 * np.pi)
            
            wave = np.sin(phases)
            
            # Add to output
            out_data[:, 0] += wave * entity.amp_L
            out_data[:, 1] += wave * entity.amp_R
            
        # Clip
        out_data = np.clip(out_data, -1.0, 1.0)
        return (out_data.tobytes(), pyaudio.paContinue)

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    def process_radar_data(self, main_audio, motion_audio, sw_x, sw_dist, sw_y, mot_dist, mot_x, mot_y, valid_sweep, width, height):
        # Audio Logic - Motion
        if mot_x != -1:
            if AUDIO_MODE == 'DISTANCE_VERTICAL':
                t_freq = np.interp(mot_y, [height - 1, 0], [AUDIO_BASE_FREQ, AUDIO_MAX_FREQ])
                dist_vol = np.interp(mot_dist, [MAX_VOL_DIST_CM, MIN_VOL_DIST_CM], [1.0, 0.0])
            else:
                t_freq = np.interp(mot_dist, [0, MAX_RADAR_DIST_CM], [AUDIO_MAX_FREQ, AUDIO_BASE_FREQ])
                dist_vol = 1.0

            pan = np.interp(mot_x, [LEFT_OFFSET, width - 1], [0.0, 1.0])
            motion_audio.update(t_freq, target_amp_L=dist_vol * (1.0 - pan), target_amp_R=dist_vol * pan)
        else:
            motion_audio.update(AUDIO_BASE_FREQ, 0.0, 0.0)

        # Audio Logic - Sweep
        if valid_sweep:
            if AUDIO_MODE == 'DISTANCE_VERTICAL':
                t_freq = np.interp(sw_y, [height - 1, 0], [AUDIO_BASE_FREQ, AUDIO_MAX_FREQ])
                dist_vol = np.interp(sw_dist, [MAX_VOL_DIST_CM, MIN_VOL_DIST_CM], [1.0, 0.0])
            else:
                t_freq = np.interp(sw_dist, [0, MAX_RADAR_DIST_CM], [AUDIO_MAX_FREQ, AUDIO_BASE_FREQ])
                dist_vol = 1.0
                
            pan = np.interp(sw_x, [LEFT_OFFSET, width - 1], [0.0, 1.0])
            main_audio.update(t_freq, target_amp_L=dist_vol * (1.0 - pan), target_amp_R=dist_vol * pan)
        else:
            main_audio.update(AUDIO_BASE_FREQ, 0.0, 0.0)

def draw_entity_trace(entity, img, channel, width, pixels_to_shift):
    # Determine current Y based on amp
    # For visualization, let's map freq to Y? Or Amp to Y?
    # Original code mapped freq to Y implicitly via spectrogram logic which isn't fully captured here.
    # The original main.py just drew lines on a 'spectro' buffer.
    # Let's reproduce the simple visualization:
    # Map Frequency to Y position (Low freq = Bottom, High freq = Top)
    
    # In main.py it was: 
    # cv2.line(spectro, ...)
    # But wait, main.py didn't actually compute FFT. It just drew a line based on current Freq.
    
    y = int(np.interp(entity.freq, [AUDIO_BASE_FREQ, AUDIO_MAX_FREQ], [img.shape[0]-1, 0]))
    
    amp = entity.amp_L if channel == 'L' else entity.amp_R
    color = entity.color_L if channel == 'L' else entity.color_R
    
    # Dim color by amplitude
    draw_color = (int(color[0]*amp), int(color[1]*amp), int(color[2]*amp))
    
    if entity.prev_y is not None:
        cv2.line(img, (width - pixels_to_shift, entity.prev_y), (width, y), draw_color, 2)
    
    return y

if __name__ == "__main__":
    print("Audio Test... Playing sweep")
    mgr = AudioManager()
    entity = AudioEntity((255,255,255), (255,255,255))
    mgr.add_entity(entity)
    
    try:
        import time
        t0 = time.time()
        while True:
            t = time.time() - t0
            freq = 400 + 200 * math.sin(t * 2)
            pan = 0.5 + 0.5 * math.sin(t)
            entity.update(freq, 0.5 * (1-pan), 0.5 * pan)
            time.sleep(0.01)
    except KeyboardInterrupt:
        mgr.close()
