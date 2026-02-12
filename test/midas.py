from mpu6050 import mpu6050
import time

sensor = mpu6050(0x68)

while True:
    accel_data = sensor.get_accel_data()
    gyro_data = sensor.get_gyro_data()

    print(f"Accel X: {accel_data['x']:.2f}, Y: {accel_data['y']:.2f}")
    print(f"Gyro X: {gyro_data['x']:.2f}, Y: {gyro_data['y']:.2f}")
    time.sleep(0.5)
(venv) dev@stesis:~/stesis $ ls
calibrate.py  config.py  depth.py  __pycache__  requirements.txt  stereo-10.0.json  test  venv
(venv) dev@stesis:~/stesis $ cat test/midas.py
import cv2
import torch
import numpy as np
import pyaudio
import time

class DepthRadar:
    def __init__(self, camera_id=0, width=640, height=480):
        # 1. Setup Camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 2. Setup MiDaS (Lightest Model: MiDaS_small)
        print("Loading MiDaS Small model...")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device)
        self.midas.eval()

        # MiDaS transforms
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        # 3. Audio Config
        self.sample_rate = 44100
        self.chunk_size = 1024  # Buffer size
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=2,
                                  rate=self.sample_rate,
                                  output=True)

        # Radar/Audio logic variables
        self.scan_x = 0
        self.scan_direction = 1  # 1 = Right, -1 = Left
        self.scan_speed = 4      # Pixels per frame
        self.img_width = 0       # Will be set dynamically
        self.img_height = 0

        # Frequency Mapping (Vertical axis -> Frequency bins)
        self.num_freq_bins = 64  # Reduce vertical resolution for audio synthesis to save CPU
        # Create a mapping of Y positions to frequencies (Logarithmic scale sounds better)
        self.freq_indices = np.logspace(np.log10(1), np.log10(self.chunk_size // 2), num=self.num_freq_bins).astype(int)

        # Spectrogram buffers (Images)
        self.spec_height = 200
        self.spec_width = width
        self.spec_l = np.zeros((self.spec_height, self.spec_width), dtype=np.uint8)
        self.spec_r = np.zeros((self.spec_height, self.spec_width), dtype=np.uint8)

    def generate_audio_from_slice(self, depth_col, pan):
        """
        Converts a column of depth values into a stereo audio chunk.
        Using IFFT for efficient synthesis of multiple frequencies.
        """
        # 1. Downsample the column to our number of frequency bins
        # We resize the column to num_freq_bins.
        # Top of image (index 0) = High Frequency. Bottom (index -1) = Low Frequency.
        # We usually want low pixels (ground) to be low freq, and high pixels (sky) to be high freq.
        # depth_col is ordered Top -> Bottom.

        depth_binned = cv2.resize(depth_col, (1, self.num_freq_bins), interpolation=cv2.INTER_AREA).flatten()

        # Flip so index 0 is bottom of image (Low Freq) if we map linearly,
        # BUT prompt says: "more on top = higher frequency".
        # Array index 0 is TOP. So Index 0 should map to High Freq.
        # The FFT array usually goes [DC, Low Freq -> High Freq].
        # So we need to REVERSE the depth_binned array so the Top (High depth indices) go to High Freq slots.
        depth_binned = depth_binned[::-1]

        # 2. Construct Frequency Domain (Spectrogram Slice)
        # Create an empty complex spectrum
        spectrum = np.zeros(self.chunk_size // 2 + 1, dtype=np.complex64)

        # Map depth magnitude to frequency bins
        # We sparsely populate the spectrum based on our selected frequency indices
        for i, idx in enumerate(self.freq_indices):
            if idx < len(spectrum):
                # Amplitude is based on depth (closer = louder)
                # Normalize depth usually 0-1000 or similar, scaling to 0.0-1.0
                amp = depth_binned[i]
                spectrum[idx] = amp * 50 # Gain multiplier

        # Randomize phase to prevent weird impulse artifacts (making it sound like noise/drone rather than a click)
        phase = np.random.uniform(0, 2*np.pi, len(spectrum))
        spectrum = spectrum * np.exp(1j * phase)

        # 3. IFFT to Time Domain
        audio_signal = np.fft.irfft(spectrum)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_signal))
        if max_val > 0:
            audio_signal = audio_signal / max_val * 0.5 # Master volume 0.5

        # 4. Apply Panning (Stereo)
        # pan: 0.0 (Left) to 1.0 (Right)
        vol_l = 1.0 - pan
        vol_r = pan

        stereo_signal = np.zeros((len(audio_signal), 2), dtype=np.float32)
        stereo_signal[:, 0] = audio_signal * vol_l
        stereo_signal[:, 1] = audio_signal * vol_r

        # Return raw bytes for pyaudio, and the magnitudes for visualization
        return stereo_signal.flatten().tobytes(), (depth_binned * vol_l), (depth_binned * vol_r)

    def update_spectrogram_img(self, img_buffer, new_col_data):
        """
        Scrolls the spectrogram image to the left and adds new data on the right.
        """
        # Shift image left
        img_buffer[:, :-1] = img_buffer[:, 1:]

        # Create visualization col (Map 0-1 float data to 0-255 uint8)
        # Resize new_col_data (which is num_freq_bins) to spec_height
        col_viz = cv2.resize(new_col_data.reshape(-1, 1), (1, self.spec_height))

        # Normalize for display
        col_viz = cv2.normalize(col_viz, None, 0, 255, cv2.NORM_MINMAX)

        # Color map it (optional, but grayscale is faster for raw insert)
        img_buffer[:, -1] = col_viz.flatten()
        return img_buffer

    def run(self):
        print("Starting Radar... Press 'q' to exit.")
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # --- 1. Process Depth (MiDaS) ---
                img_h, img_w, _ = frame.shape
                self.img_width, self.img_height = img_w, img_h

                # Preprocess for MiDaS
                input_batch = self.midas_transforms(frame).to(self.device)

                # Inference
                with torch.no_grad():
                    prediction = self.midas(input_batch)

                    # Resize to original resolution
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img_h,
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                    depth_map = prediction.cpu().numpy()

                # Normalize depth map for visualization (0-255)
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                # Normalize 0.0 to 1.0 for audio math
                depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

                # --- 2. Radar Scan Logic ---
                self.scan_x += (self.scan_speed * self.scan_direction)

                # Ping-pong bounds check
                if self.scan_x >= img_w:
                    self.scan_x = img_w - 1
                    self.scan_direction = -1
                elif self.scan_x < 0:
                    self.scan_x = 0
                    self.scan_direction = 1

                current_scan_x = int(self.scan_x)

                # --- 3. Audio Generation ---
                # Get the vertical slice at current_scan_x
                depth_slice = depth_norm[:, current_scan_x]

                # Calculate Pan (0.0 = Left, 1.0 = Right)
                pan = current_scan_x / img_w

                # Generate audio
                audio_bytes, mag_l, mag_r = self.generate_audio_from_slice(depth_slice, pan)

                # Play audio (Non-blocking write to stream usually causes sync issues in simple loops,
                # but blocking write might slow down FPS. Here we assume fast processing.)
                # We write a small chunk. To match video FPS, chunk size logic is tricky.
                # Here we just pump the calculated audio.
                self.stream.write(audio_bytes)

                # --- 4. Visualization ---

                # A. Depth Map Image (Colorized)
                depth_viz = (depth_norm * 255).astype(np.uint8)
                depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)

                # Draw the scan line
                cv2.line(depth_viz, (current_scan_x, 0), (current_scan_x, img_h), (0, 255, 0), 2)

                # B. Spectrograms
                self.spec_l = self.update_spectrogram_img(self.spec_l, mag_l)
                self.spec_r = self.update_spectrogram_img(self.spec_r, mag_r)

                # Colorize Spectrograms for "Matrix" look
                spec_l_viz = cv2.applyColorMap(self.spec_l, cv2.COLORMAP_VIRIDIS)
                spec_r_viz = cv2.applyColorMap(self.spec_r, cv2.COLORMAP_PLASMA)

                # Resize Spectrograms to match half-width of the screen for layout
                spec_h = img_h # Match height of top row roughly or fix it
                spec_l_viz = cv2.resize(spec_l_viz, (img_w, img_h))
                spec_r_viz = cv2.resize(spec_r_viz, (img_w, img_h))

                # --- 5. GUI Layout ---
                # Layout:
                # [ Camera ] [ Depth  ]
                # [ Spec L ] [ Spec R ]

                top_row = np.hstack((frame, depth_viz))
                bottom_row = np.hstack((spec_l_viz, spec_r_viz))
                combined = np.vstack((top_row, bottom_row))

                # Resize for display if too big
                disp_h, disp_w, _ = combined.shape
                if disp_w > 1280:
                    scale = 1280 / disp_w
                    combined = cv2.resize(combined, (int(disp_w*scale), int(disp_h*scale)))

                cv2.imshow("MiDaS Depth Radar", combined)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure you are not running other audio apps that block the device
    app = DepthRadar(camera_id=0)
    app.run()