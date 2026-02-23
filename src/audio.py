import numpy as np
import cv2
import math
import time
from radar import Radar

try:
    import pyaudio
except ImportError:
    pyaudio = None
    print("PyAudio not found. Audio will be disabled.")

class AudioEntity:
    def __init__(self, color_L, color_R, base_freq):
        self.freq = base_freq
        self.amp_L = 0.0
        self.amp_R = 0.0
        self.color_L = color_L
        self.color_R = color_R
        self.prev_y = None
        self.phase = 0.0 

    def update(self, target_freq, target_amp_L, target_amp_R, smoothing_coeff):
        # Smoothing
        if smoothing_coeff <= 0:
            self.freq = target_freq
            self.amp_L = target_amp_L
            self.amp_R = target_amp_R
        else:
            self.freq = (self.freq * smoothing_coeff) + (target_freq * (1.0 - smoothing_coeff))
            self.amp_L = (self.amp_L * smoothing_coeff) + (target_amp_L * (1.0 - smoothing_coeff))
            self.amp_R = (self.amp_R * smoothing_coeff) + (target_amp_R * (1.0 - smoothing_coeff))

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
        
        for entity in self.entities:
            phase_increment = 2 * np.pi * entity.freq / self.RATE
            phases = entity.phase + np.arange(frame_count) * phase_increment
            entity.phase = (phases[-1] + phase_increment) % (2 * np.pi)
            
            wave = np.sin(phases)
            
            out_data[:, 0] += wave * entity.amp_L
            out_data[:, 1] += wave * entity.amp_R
            
        out_data = np.clip(out_data, -1.0, 1.0)
        return (out_data.tobytes(), pyaudio.paContinue)

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

class Audio(Radar):
    def __init__(self):
        super().__init__()
        
        self.audio_mgr = None
        self.main_audio = None
        self.motion_audio = None
        
        self.spectro_L = None
        self.spectro_R = None
        self.audio_ui = None 

    def setup(self):
        super().setup()
        print("Initializing Audio...")
        
        self.audio_mgr = AudioManager(rate=self.AUDIO_SAMPLE_RATE, chunk=self.AUDIO_CHUNK_SIZE)
        self.main_audio = AudioEntity(color_L=(255, 255, 0), color_R=(255, 255, 0), base_freq=self.AUDIO_BASE_FREQ) 
        self.motion_audio = AudioEntity(color_L=(0, 255, 0), color_R=(0, 255, 0), base_freq=self.AUDIO_BASE_FREQ)
        
        if self.PLAY_AUDIO:
            self.audio_mgr.add_entity(self.main_audio)
            self.audio_mgr.add_entity(self.motion_audio)
            
        self.spectro_L = np.zeros((self.SPECTRO_HEIGHT, self.width, 3), dtype=np.uint8)
        self.spectro_R = np.zeros((self.SPECTRO_HEIGHT, self.width, 3), dtype=np.uint8)

    def draw_entity_trace(self, entity, img, channel, width, pixels_to_shift):
        y = int(np.interp(entity.freq, [self.AUDIO_BASE_FREQ, self.AUDIO_MAX_FREQ], [img.shape[0]-1, 0]))
        
        amp = entity.amp_L if channel == 'L' else entity.amp_R
        color = entity.color_L if channel == 'L' else entity.color_R
        
        draw_color = (int(color[0]*amp), int(color[1]*amp), int(color[2]*amp))
        
        if entity.prev_y is not None:
            cv2.line(img, (width - pixels_to_shift, entity.prev_y), (width, y), draw_color, 2)
        
        return y

    def mute(self):
        if self.main_audio: self.main_audio.update(400, 0, 0, 0)
        if self.motion_audio: self.motion_audio.update(400, 0, 0, 0)

    def loop(self):
        super().loop() # Up to Radar loop
        
        # Audio Logic
        # Calculate dt again or store in Config?
        # Radar loop updated last_radar_time. We can assume small drift or use same time.
        # But we need consistent time delta for scrolling spectrogram.
        
        # We can implement a shared time delta in Config or just recalc here.
        # Recalc is safer if called sequentially in same thread.
        # Actually Config.loop assumes one big loop.
        # But here we are in same iteration.
        
        # Motion Audio
        if self.mot_x != -1:
            if self.AUDIO_MODE == 'DISTANCE_VERTICAL':
                t_freq = np.interp(self.mot_y, [self.height - 1, 0], [self.AUDIO_BASE_FREQ, self.AUDIO_MAX_FREQ])
                dist_vol = np.interp(self.mot_dist, [self.MAX_VOL_DIST_CM, self.MIN_VOL_DIST_CM], [1.0, 0.0])
            else:
                t_freq = np.interp(self.mot_dist, [0, self.MAX_RADAR_DIST_CM], [self.AUDIO_MAX_FREQ, self.AUDIO_BASE_FREQ])
                dist_vol = 1.0

            pan = np.interp(self.mot_x, [self.LEFT_OFFSET, self.width - 1], [0.0, 1.0])
            self.motion_audio.update(t_freq, dist_vol * (1.0 - pan), dist_vol * pan, self.AUDIO_SMOOTHING_COEFF)
        else:
            self.motion_audio.update(self.AUDIO_BASE_FREQ, 0.0, 0.0, self.AUDIO_SMOOTHING_COEFF)

        # Main/Sweep Audio
        if self.valid_sweep:
            if self.AUDIO_MODE == 'DISTANCE_VERTICAL':
                t_freq = np.interp(self.sw_y, [self.height - 1, 0], [self.AUDIO_BASE_FREQ, self.AUDIO_MAX_FREQ])
                dist_vol = np.interp(self.sw_dist, [self.MAX_VOL_DIST_CM, self.MIN_VOL_DIST_CM], [1.0, 0.0])
            else:
                t_freq = np.interp(self.sw_dist, [0, self.MAX_RADAR_DIST_CM], [self.AUDIO_MAX_FREQ, self.AUDIO_BASE_FREQ])
                dist_vol = 1.0
                
            pan = np.interp(self.sw_x, [self.LEFT_OFFSET, self.width - 1], [0.0, 1.0])
            self.main_audio.update(t_freq, dist_vol * (1.0 - pan), dist_vol * pan, self.AUDIO_SMOOTHING_COEFF)
        else:
            self.main_audio.update(self.AUDIO_BASE_FREQ, 0.0, 0.0, self.AUDIO_SMOOTHING_COEFF)

        # Spectrograms
        # Need true dt logic? We can assume ~30fps -> 0.033s
        # Or measure time since last Audio loop.
        # Let's verify if 'dt' used in original was simply curr-prev.
        # Yes.
        # Using 0.05 as approx if needed, or track time.
        dt = 0.05 # Approximation since we don't track self.last_audio_time yet.
        
        pixels_to_shift = max(1, int((dt / self.SPECTRO_TIME_HISTORY_SEC) * self.width))
        self.spectro_L = np.roll(self.spectro_L, -pixels_to_shift, axis=1)
        self.spectro_R = np.roll(self.spectro_R, -pixels_to_shift, axis=1)
        self.spectro_L[:, -pixels_to_shift:] = 0
        self.spectro_R[:, -pixels_to_shift:] = 0

        for grid_y in range(0, self.SPECTRO_HEIGHT, 50):
            cv2.line(self.spectro_L, (self.width - pixels_to_shift, grid_y), (self.width, grid_y), (30, 30, 30), 1)
            cv2.line(self.spectro_R, (self.width - pixels_to_shift, grid_y), (self.width, grid_y), (30, 30, 30), 1)

        prev_yl_main = self.draw_entity_trace(self.main_audio, self.spectro_L, 'L', self.width, pixels_to_shift)
        prev_yr_main = self.draw_entity_trace(self.main_audio, self.spectro_R, 'R', self.width, pixels_to_shift)
        self.main_audio.prev_y = prev_yl_main if prev_yl_main is not None else prev_yr_main

        prev_yl_mot = self.draw_entity_trace(self.motion_audio, self.spectro_L, 'L', self.width, pixels_to_shift)
        prev_yr_mot = self.draw_entity_trace(self.motion_audio, self.spectro_R, 'R', self.width, pixels_to_shift)
        self.motion_audio.prev_y = prev_yl_mot if prev_yl_mot is not None else prev_yr_mot
        
        spectro_L_disp = self.spectro_L.copy()
        spectro_R_disp = self.spectro_R.copy()
        cv2.putText(spectro_L_disp, "Left Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(spectro_R_disp, "Right Channel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Combine Final UI
        # Need vis_l, vis_r from Camera (rect_frames from Depth)
        # depth_view from Depth
        # radar_view from Radar
        
        if self.rect_frames[0] is not None and self.depth_view is not None and self.radar_view is not None:
            vis_l = self.draw_overlays(self.rect_frames[0].copy())
            vis_r = self.draw_overlays(self.rect_frames[1].copy())
            
            # Add sweep line to depth view here again (as per original logic logic duplication)
            depth_disp = self.depth_view.copy()
            if self.sw_x >= self.LEFT_OFFSET:
                cv2.line(depth_disp, (self.sw_x, 0), (self.sw_x, self.height), (255, 255, 255), 2)
            
            top_row = np.hstack((vis_l, vis_r))
            mid_row = np.hstack((depth_disp, self.radar_view))
            bot_row = np.hstack((spectro_L_disp, spectro_R_disp))
            
            self.audio_ui = np.vstack((top_row, mid_row, bot_row))
        
    def cleanup(self):
        super().cleanup()
        if self.audio_mgr: self.audio_mgr.close()

if __name__ == "__main__":
    app = Audio()
    app.run()
