import cv2
import numpy as np
import threading
import http.server
import socketserver
import os
import sys

from hand import Hand
from audio import Audio

class Main(Hand, Audio):
    def __init__(self):
        super().__init__()
        self.mode = "INACTIVE"

    def start_web_server(self):
        # Serve files from ../web relative to src_class/
        web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../web')
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=web_dir, **kwargs)
        
        try:
            # Binding to 0.0.0.0 allowing external access
            # We use reuse_address to avoid "Address already in use" on restarts
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer((self.HOST, self.DOMAIN), Handler) as httpd:
                print(f"Serving web content at http://{self.HOST}:{self.DOMAIN} from {web_dir}")
                httpd.serve_forever()
        except Exception as e:
            print(f"Web Server Error: {e}")

    def setup(self):
        print("Initializing Stesis Main...")
        super().setup() # Init all chains
        
        # Start Web Server
        server_thread = threading.Thread(target=self.start_web_server, daemon=True)
        server_thread.start()
        
        print("System Ready. Press 'q' to exit.")

    def loop(self):
        super().loop() # Run Hand, Audio, Radar, Depth, Camera, IMU logic
        
        # Decide Mode based on IMU (self.pitch, self.ax from IMU class)
        # Pitch < -10 ?
        #   No -> INACTIVE
        #   Yes -> Check Ax
        #          Ax < -2 ?
        #            Yes -> HAND
        #            No  -> NAV (Audio/Radar)
        
        if self.pitch < -10:
            if self.ax < -2:
                self.mode = "HAND"
            else:
                self.mode = "NAV"
        else:
            self.mode = "INACTIVE"
            
        final_ui = None
        
        if self.mode == "INACTIVE":
            self.mute() # From Audio class
            
            # Show black screen with text
            inactive_screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(inactive_screen, "INACTIVE", (self.width//2 - 60, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            final_ui = inactive_screen
            
        elif self.mode == "HAND":
            self.mute()
            if self.hand_message:
                print(f"Hand Event: {self.hand_message}")
            final_ui = self.hand_view
            
        elif self.mode == "NAV":
            final_ui = self.audio_ui

        if final_ui is not None:
            cv2.imshow("Stesis", final_ui)

if __name__ == "__main__":
    app = Main()
    app.run()
