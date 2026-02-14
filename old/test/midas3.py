import cv2
import numpy as np
import pyaudio
import time
import os
import urllib.request
import onnxruntime as ort
import threading

class DepthRadar:
    def __init__(self, camera_id=0, width=320, height=240):
        # --- Configuration ---
        self.width = width
        self.height = height
        self.running = False
        
        # Threading Locks
        # We need locks to prevent reading the image while it's being written to
        self.data_lock = threading.Lock()

        # --- 1. Setup Camera ---
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # --- 2. Load ONNX Model ---
        self.model_path = "midas_v2_1_small.onnx"
        self._ensure_model_downloaded()

        print(f"Loading ONNX Model: {self.model_path}...")
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.net_h, self.net_w = self.input_shape[2], self.input_shape[3]

        # --- 3. Audio Config ---
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=2,
                                  rate=self.sample_rate,
                                  output=True)

        # --- Shared State (Accessed by both threads) ---
        # Initialize depth map with zeros to avoid errors before first frame
        self.current_depth_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        self.scan_x = 0.0
        self.scan_direction = 1
        # Scan speed: pixels per audio chunk
        # Adjust this to change how fast the radar sweeps
        self.scan_speed = 3.0 

        # Frequency bins for IFFT
        self.num_freq_bins = 64
        self.freq_indices = np.logspace(np.log10(1), np.log10(self.chunk_size // 2), num=self.num_freq_bins).astype(int)

        # Spectrogram buffers (Shared for display)
        self.spec_height = 200
        self.spec_l = np.zeros((self.spec_height, self.width), dtype=np.uint8)
        self.spec_r = np.zeros((self.spec_height, self.width), dtype=np.uint8)

    def _ensure_model_downloaded(self):
        if not os.path.exists(self.model_path):
            print("Downloading MiDaS ONNX model (first run only)...")
            url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def run_inference(self, frame):
        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.net_w, self.net_h))
        img = img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # Inference
        outputs = self.session.run(None, {self.input_name: img})
        depth_map = outputs[0][0]

        # Post-process
        depth_map = cv2.resize(depth_map, (self.width, self.height))
        d_min, d_max = depth_map.min(), depth_map.max()
        depth_norm = (depth_map - d_min) / (d_max - d_min + 1e-6)
        
        return depth_norm

    def generate_audio_chunk(self, depth_slice, pan):
        # 1. Downsample slice to freq bins
        depth_binned = cv2.resize(depth_slice.reshape(-1, 1), (1, self.num_freq_bins), interpolation=cv2.INTER_AREA).flatten()
        # Flip (Top of image = High Freq)
        depth_binned = depth_binned[::-1]

        # 2. Build Spectrum
        spectrum = np.zeros(self.chunk_size // 2 + 1, dtype=np.complex64)
        for i, idx in enumerate(self.freq_indices):
            if idx < len(spectrum):
                spectrum[idx] = depth_binned[i] * 30

        # Random phase
        phase = np.random.uniform(0, 2*np.pi, len(spectrum))
        spectrum = spectrum * np.exp(1j * phase)

        # 3. IFFT
        audio = np.fft.irfft(spectrum)

        # Normalize
        m = np.max(np.abs(audio))
        if m > 0: audio = audio / m * 0.5

        # 4. Stereo Pan
        vol_l = 1.0 - pan
        vol_r = pan
        stereo = np.zeros((len(audio), 2), dtype=np.float32)
        stereo[:, 0] = audio * vol_l
        stereo[:, 1] = audio * vol_r
        
        return stereo.flatten().tobytes(), depth_binned * vol_l, depth_binned * vol_r

    def update_spec(self, buffer, new_col):
        # Shift left
        buffer[:, :-1] = buffer[:, 1:]
        # Resize col
        col = cv2.resize(new_col.reshape(-1, 1), (1, self.spec_height))
        col = cv2.normalize(col, None, 0, 255, cv2.NORM_MINMAX)
        # Insert at right
        buffer[:, -1] = col.flatten()
        return buffer

    # --- THE BACKGROUND THREAD ---
    def audio_radar_loop(self):
        """
        This runs in a separate thread. 
        It moves the radar line and generates audio continuously, 
        reading whatever the 'latest' depth map happens to be.
        """
        print("Audio Thread Started.")
        while self.running:
            # 1. Update Radar Position
            self.scan_x += (self.scan_speed * self.scan_direction)
            
            # Bounce logic
            if self.scan_x >= self.width:
                self.scan_x = self.width - 1
                self.scan_direction = -1
            elif self.scan_x < 0:
                self.scan_x = 0
                self.scan_direction = 1
                
            current_x_int = int(self.scan_x)

            # 2. Get Data safely
            with self.data_lock:
                # Copy the slice so we don't hold the lock during audio generation
                depth_slice = self.current_depth_map[:, current_x_int].copy()

            # 3. Generate Audio
            pan = current_x_int / self.width
            audio_data, mag_l, mag_r = self.generate_audio_chunk(depth_slice, pan)

            # 4. Write to Stream (This blocks for ~chunk_duration, acting as the thread timer)
            self.stream.write(audio_data)

            # 5. Update Spectrograms (Protected by lock because Main thread reads them)
            with self.data_lock:
                self.spec_l = self.update_spec(self.spec_l, mag_l)
                self.spec_r = self.update_spec(self.spec_r, mag_r)

    # --- THE MAIN THREAD ---
    def run(self):
        print("Starting DepthRadar... Press 'q' to quit.")
        self.running = True
        
        # Start the Audio/Radar thread
        t = threading.Thread(target=self.audio_radar_loop)
        t.daemon = True # Thread dies if main program dies
        t.start()

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break

                # --- 1. Heavy Inference ---
                depth_norm = self.run_inference(frame)

                # --- 2. Update Shared State ---
                with self.data_lock:
                    self.current_depth_map = depth_norm
                    # We also read scan_x and specs here for visualization to ensure consistency
                    viz_scan_x = int(self.scan_x)
                    viz_spec_l = self.spec_l.copy()
                    viz_spec_r = self.spec_r.copy()

                # --- 3. Visualization (GUI) ---
                
                # Colorize Depth
                depth_viz = (depth_norm * 255).astype(np.uint8)
                depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)
                
                # Draw the Radar Line (Green)
                cv2.line(depth_viz, (viz_scan_x, 0), (viz_scan_x, self.height), (0, 255, 0), 2)

                # Colorize Spectrograms
                sl_viz = cv2.applyColorMap(viz_spec_l, cv2.COLORMAP_VIRIDIS)
                sr_viz = cv2.applyColorMap(viz_spec_r, cv2.COLORMAP_PLASMA)
                
                # Resize for layout
                sl_viz = cv2.resize(sl_viz, (self.width, self.height))
                sr_viz = cv2.resize(sr_viz, (self.width, self.height))

                # Combine images
                top_row = np.hstack((frame, depth_viz))
                bot_row = np.hstack((sl_viz, sr_viz))
                final_img = np.vstack((top_row, bot_row))
                
                # Display
                if final_img.shape[1] > 1000:
                    scale = 0.8
                    final_img = cv2.resize(final_img, (0,0), fx=scale, fy=scale)
                    
                cv2.imshow("MiDaS Depth Radar (Multi-threaded)", final_img)
                if cv2.waitKey(1) == ord('q'):
                    self.running = False
                    break

        finally:
            self.running = False
            # Wait a moment for thread to exit
            time.sleep(0.5) 
            self.cap.release()
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DepthRadar()
    app.run()
