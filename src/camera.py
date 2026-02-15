import cv2
import sys
from config import CAM_ID_LEFT, CAM_ID_RIGHT

def init_cameras():
    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)
    
    # Optional: Set buffer size to 1 to reduce lag
    cap_l.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_r.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap_l.isOpened() or not cap_r.isOpened():
        print("[ERROR] Could not open both cameras.")
        return None, None
        
    return cap_l, cap_r

def get_frames(cap_l, cap_r, rotate_right=True):
    ret_l, frame_l = cap_l.read()
    ret_r, frame_r = cap_r.read()
    
    if not ret_l or not ret_r:
        return False, None, False, None
        
    if rotate_right:
        frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
        
    return True, frame_l, True, frame_r

def close_cameras(cap_l, cap_r):
    if cap_l: cap_l.release()
    if cap_r: cap_r.release()

if __name__ == "__main__":
    print("--- Camera Module Test ---")
    cap_l, cap_r = init_cameras()
    if cap_l and cap_r:
        try:
            print("Cameras opened. Press 'q' to exit.")
            while True:
                ret_l, frame_l, ret_r, frame_r = get_frames(cap_l, cap_r)
                if not ret_l or not ret_r: break
                
                # Check if frames are valid before stacking
                if frame_l is None or frame_r is None: continue
                
                # Resize for display if needed? Assuming they are same size from config
                # Just horizontal stack
                combined = cv2.hconcat([frame_l, frame_r])
                cv2.imshow("Camera Test", combined)
                
                if cv2.waitKey(1) == ord('q'): break
        finally:
            close_cameras(cap_l, cap_r)
            cv2.destroyAllWindows()
