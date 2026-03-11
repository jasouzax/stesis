import cv2
import numpy as np
import torch
from camera import Camera

class Midas(Camera):
    def __init__(self):
        super().__init__()
        self.depth = None
        self.gui = None
        
        # MiDaS specific variables
        self.device = None
        self.midas = None
        self.transform = None

    def setup(self):
        super().setup()
        
        print("Loading MiDaS Small model...")
        
        # Use GPU if available for better framerates, otherwise fallback to CPU
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")

        # Load MiDaS Small from PyTorch Hub
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.to(self.device)
        self.midas.eval() # Set model to evaluation mode

        # Load transforms to resize and normalize the image for the model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform
        
        print("MiDaS loaded successfully.")

    def loop(self, show=True):
        # Get frames from your Camera superclass
        if super().loop(False, True): return True
        
        # MiDaS is monocular, so we only need one frame.
        # Assuming self.frames contains (frame_l, frame_r), we'll grab the left one.
        frame = self.frames[0] 

        # OpenCV uses BGR, but MiDaS expects RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply the transform and move data to the correct device (CPU/GPU)
        input_batch = self.transform(img).to(self.device)

        # Compute Depth
        with torch.no_grad():
            prediction = self.midas(input_batch)

            # Resize the prediction to match the original image resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Move prediction back to CPU and convert to a numpy array
        depth_map = prediction.cpu().numpy()

        # Visual Depth: Normalize the raw output to 0-255 for visual rendering
        depth_map_normalized = cv2.normalize(
            depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        # Apply the JET colormap (Red is closer, Blue is further in standard MiDaS visualization)
        self.depth = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

        # Render setup for your GUI/display
        self.gui = frame
        if show:
            # Stack the original frame and the depth map vertically
            self.gui = np.vstack((self.gui, self.depth))

if __name__ == "__main__":
    app = Midas()
    app.run()
