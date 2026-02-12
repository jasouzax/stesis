import cv2
import numpy as np
import pyaudio
import time
import os
import urllib.request
import onnxruntime as ort

class DepthRadar:
    def __init__(self, camera_id=0, width=320, height=240):
        # 1. Setup Camera (320x240 is optimal for Pi 4 speed)
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 2. Load ONNX Model
        self.model_path = "midas_v2_1_small.onnx"
        self._ensure_model_downloaded()

        print(f"Loading ONNX Model: {self.model_path}...")
        # Use CPU provider (Pi 4 GPU is not supported by standard ONNX)
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        # Shape is usually [1, 3, 256, 256] for MiDaS ONNX
        self.net_h, self.net_w = self.input_shape[2], self.input_shape[3]

        # 3. Audio Config
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=2,
                                  rate=self.sample_rate,
                                  output=True)

        # Radar/Audio logic
        self.scan_x = 0
        self.scan_direction = 1
        self.scan_speed = 6

        # Frequency bins for IFFT
        self.num_freq_bins = 64
        self.freq_indices = np.logspace(np.log10(1), np.log10(self.chunk_size // 2), num=self.num_freq_bins).astype(int)

        # Spectrogram buffers
        self.spec_height = 200
        self.spec_l = np.zeros((self.spec_height, width), dtype=np.uint8)
        self.spec_r = np.zeros((self.spec_height, width), dtype=np.uint8)

    def _ensure_model_downloaded(self):
        if not os.path.exists(self.model_path):
            print("Downloading MiDaS ONNX model (first run only)...")
            # Official ONNX model link for MiDaS v2.1 Small
            url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def run_inference(self, frame):
        img_h, img_w, _ = frame.shape

        # Preprocess: Resize, Normalize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.net_w, self.net_h))

        # Ensure image is float32
        img = img.astype(np.float32) / 255.0

        # FIX: Define mean and std explicitly as float32 to prevent upcasting to double
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std

        img = img.transpose(2, 0, 1) # HWC -> CHW
        img = np.expand_dims(img, axis=0) # Add batch dimension [1, 3, 256, 256]

        # Run Inference
        outputs = self.session.run(None, {self.input_name: img})
        depth_map = outputs[0][0] # Remove batch dim -> [256, 256]

        # Post-process: Resize back to original frame size
        depth_map = cv2.resize(depth_map, (img_w, img_h))

        # Normalize 0-1 for visualization/audio
        d_min, d_max = depth_map.min(), depth_map.max()
        depth_norm = (depth_map - d_min) / (d_max - d_min + 1e-6)

        return depth_norm

    def generate_audio(self, depth_slice, pan):
        # 1. Downsample slice to freq bins (using simpler resizing)
        # depth_slice shape is (H,). We resize to (num_freq_bins, 1)
        depth_binned = cv2.resize(depth_slice.reshape(-1, 1), (1, self.num_freq_bins), interpolation=cv2.INTER_AREA).flatten()

        # Flip because Index 0 in image is TOP (High Freq), but FFT index 0 is Low Freq.
        # We want Top -> High Freq. So we reverse it.
        # But wait! FFT array layout is [DC, Low...High].
        # So high indices = high freq.
        # If we map directly: Img[0] (Top) -> FFT[0] (Low). Incorrect.
        # We want Img[0] (Top) -> FFT[MAX] (High).
        # So we MUST reverse the array.
        depth_binned = depth_binned[::-1]

        # 2. Build Spectrum
        spectrum = np.zeros(self.chunk_size // 2 + 1, dtype=np.complex64)

        for i, idx in enumerate(self.freq_indices):
            if idx < len(spectrum):
                # Amplitude = Depth
                spectrum[idx] = depth_binned[i] * 30

        # Random phase for "drone" sound
        phase = np.random.uniform(0, 2*np.pi, len(spectrum))
        spectrum = spectrum * np.exp(1j * phase)

        # 3. IFFT
        audio = np.fft.irfft(spectrum)

        # Normalize volume
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
        # Resize col to fit buffer height
        col = cv2.resize(new_col.reshape(-1, 1), (1, self.spec_height))
        # Normalize for display
        col = cv2.normalize(col, None, 0, 255, cv2.NORM_MINMAX)
        # Insert at right
        buffer[:, -1] = col.flatten()
        return buffer

    def run(self):
        print("Starting Radar (ONNX)... Press 'q' to quit.")
        try:
            while self.cap.isOpened():
                start_t = time.time()
                ret, frame = self.cap.read()
                if not ret: break

                img_h, img_w, _ = frame.shape

                # --- 1. Depth Inference ---
                depth_norm = self.run_inference(frame)

                # --- 2. Radar Scan Logic ---
                self.scan_x += (self.scan_speed * self.scan_direction)
                # Bounce
                if self.scan_x >= img_w:
                    self.scan_x = img_w - 1
                    self.scan_direction = -1
                elif self.scan_x < 0:
                    self.scan_x = 0
                    self.scan_direction = 1

                current_x = int(self.scan_x)

                # --- 3. Audio Generation ---
                # Get the vertical slice
                slc = depth_norm[:, current_x]

                # Pan: 0.0 (Left) -> 1.0 (Right)
                pan = current_x / img_w

                # Generate
                audio_data, mag_l, mag_r = self.generate_audio(slc, pan)
                self.stream.write(audio_data)

                # --- 4. Visualization ---

                # A. Depth Map (Colorized)
                depth_viz = (depth_norm * 255).astype(np.uint8)
                depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)
                # Draw Scan Line
                cv2.line(depth_viz, (current_x, 0), (current_x, img_h), (0, 255, 0), 2)

                # B. Spectrograms
                self.spec_l = self.update_spec(self.spec_l, mag_l)
                self.spec_r = self.update_spec(self.spec_r, mag_r)

                sl_viz = cv2.applyColorMap(self.spec_l, cv2.COLORMAP_VIRIDIS)
                sr_viz = cv2.applyColorMap(self.spec_r, cv2.COLORMAP_PLASMA)

                # Resize Spectrograms to match width/height for grid
                sl_viz = cv2.resize(sl_viz, (img_w, img_h))
                sr_viz = cv2.resize(sr_viz, (img_w, img_h))

                # Layout Grid
                # [ Camera ] [ Depth ]
                # [ Spec L ] [ Spec R ]
                top_row = np.hstack((frame, depth_viz))
                bot_row = np.hstack((sl_viz, sr_viz))
                final_img = np.vstack((top_row, bot_row))

                # Resize if too big for screen
                if final_img.shape[1] > 1000:
                    scale = 0.8
                    final_img = cv2.resize(final_img, (0,0), fx=scale, fy=scale)

                cv2.imshow("MiDaS Depth Radar (ONNX)", final_img)

                if cv2.waitKey(1) == ord('q'): break

                # Optional: Print FPS
                # print(f"FPS: {1.0 / (time.time() - start_t):.2f}")

        finally:
            self.cap.release()
            self.stream.close()
            self.p.terminate()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DepthRadar()
    app.run()