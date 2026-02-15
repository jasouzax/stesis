import numpy as np
import time
import sys
import cv2

try:
    import pyaudio
except ImportError:
    print("PyAudio not found. Run: pip install pyaudio")
    # sys.exit(1) # Don't exit on import, only when main is run or class is used

from config import (
    AUDIO_BASE_FREQ, AUDIO_MAX_FREQ, AUDIO_SMOOTHING_COEFF,
    SPECTRO_HEIGHT, SPECTRO_TIME_HISTORY_SEC,
    PLAY_AUDIO, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE
)

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
        if AUDIO_SMOOTHING_COEFF <= 0:
            self.freq = target_freq
            self.amp_L = target_amp_L
            self.amp_R = target_amp_R
        else:
            self.freq = (self.freq * AUDIO_SMOOTHING_COEFF) + (target_freq * (1.0 - AUDIO_SMOOTHING_COEFF))
            self.amp_L = (self.amp_L * AUDIO_SMOOTHING_COEFF) + (target_amp_L * (1.0 - AUDIO_SMOOTHING_COEFF))
            self.amp_R = (self.amp_R * AUDIO_SMOOTHING_COEFF) + (target_amp_R * (1.0 - AUDIO_SMOOTHING_COEFF))

def freq_to_y(freq, height):
    clamped_freq = max(AUDIO_BASE_FREQ, min(AUDIO_MAX_FREQ, freq))
    normalized = (clamped_freq - AUDIO_BASE_FREQ) / (AUDIO_MAX_FREQ - AUDIO_BASE_FREQ)
    return int(height - 1 - (normalized * (height - 1)))

def draw_entity_trace(entity, spectro_canvas, channel, width, pixels_to_shift):
    y = freq_to_y(entity.freq, SPECTRO_HEIGHT)
    amp = entity.amp_L if channel == 'L' else entity.amp_R
    
    if amp > 0.01: 
        color = tuple(int(c * amp) for c in (entity.color_L if channel == 'L' else entity.color_R))
        if entity.prev_y is not None:
            cv2.line(spectro_canvas, (width - pixels_to_shift - 1, entity.prev_y), (width - 1, y), color, 2)
        else:
            cv2.circle(spectro_canvas, (width - 1, y), 1, color, -1)
    
    return y if amp > 0.01 else None

class AudioManager:
    def __init__(self):
        self.active_entities = []
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=AUDIO_SAMPLE_RATE,
            output=True,
            frames_per_buffer=AUDIO_CHUNK_SIZE,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()

    def add_entity(self, entity):
        self.active_entities.append(entity)
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        out_chunk = np.zeros((frame_count, 2), dtype=np.float32)
        if PLAY_AUDIO:
            t = np.arange(frame_count, dtype=np.float32) / AUDIO_SAMPLE_RATE
            for e in self.active_entities:
                if e.amp_L < 0.001 and e.amp_R < 0.001:
                    continue
                
                wave = np.sin(2 * np.pi * e.freq * t + e.phase)
                e.phase += 2 * np.pi * e.freq * frame_count / AUDIO_SAMPLE_RATE
                e.phase %= 2 * np.pi
                
                out_chunk[:, 0] += wave * e.amp_L
                out_chunk[:, 1] += wave * e.amp_R
            
        out_chunk = np.clip(out_chunk, -1.0, 1.0)
        return (out_chunk.tobytes(), pyaudio.paContinue)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

if __name__ == "__main__":
    print("--- Audio Debug Mode ---")
    print("Playing constant frequency panning Left <-> Right...")
    
    if 'pyaudio' not in sys.modules:
        print("Pyaudio missing, exiting.")
        sys.exit(1)
        
    manager = AudioManager()
    
    # Test Entity
    test_entity = AudioEntity(color_L=(255, 0, 0), color_R=(0, 0, 255))
    manager.add_entity(test_entity)
    
    # Visualization setup
    width = 800
    spectro_L = np.zeros((SPECTRO_HEIGHT, width, 3), dtype=np.uint8)
    spectro_R = np.zeros((SPECTRO_HEIGHT, width, 3), dtype=np.uint8)
    
    last_time = time.time()
    
    try:
        while True:
            curr_time = time.time()
            dt = curr_time - last_time
            last_time = curr_time
            
            # Logic: Constant freq, Pan L->R->L
            freq = 600.0
            pan_cycle = (curr_time % 4.0) / 2.0 # 0 to 2
            if pan_cycle > 1.0:
                 pan = 2.0 - pan_cycle # 1 -> 0
            else:
                 pan = pan_cycle       # 0 -> 1
            
            vol = 0.8
            test_entity.update(freq, target_amp_L=vol*(1.0-pan), target_amp_R=vol*pan)
            
            # --- SPECTROGRAM RENDERING ---
            pixels_to_shift = max(1, int((dt / SPECTRO_TIME_HISTORY_SEC) * width))
            spectro_L = np.roll(spectro_L, -pixels_to_shift, axis=1)
            spectro_R = np.roll(spectro_R, -pixels_to_shift, axis=1)
            spectro_L[:, -pixels_to_shift:] = 0
            spectro_R[:, -pixels_to_shift:] = 0

            for grid_y in range(0, SPECTRO_HEIGHT, 50):
                cv2.line(spectro_L, (width - pixels_to_shift, grid_y), (width, grid_y), (30, 30, 30), 1)
                cv2.line(spectro_R, (width - pixels_to_shift, grid_y), (width, grid_y), (30, 30, 30), 1)
            
            prev_yl = draw_entity_trace(test_entity, spectro_L, 'L', width, pixels_to_shift)
            prev_yr = draw_entity_trace(test_entity, spectro_R, 'R', width, pixels_to_shift)
            # Update entity prev_y based on which channel was drawn? 
            # In main.py it was combined. Here we track simple.
            test_entity.prev_y = prev_yl if prev_yl is not None else prev_yr
            
            cv2.putText(spectro_L, f"Left Channel (Vol: {test_entity.amp_L:.2f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(spectro_R, f"Right Channel (Vol: {test_entity.amp_R:.2f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            combined = np.hstack((spectro_L, spectro_R))
            cv2.imshow("Audio Debug", combined)
            
            if cv2.waitKey(1) == ord('q'): break
            
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
        cv2.destroyAllWindows()
