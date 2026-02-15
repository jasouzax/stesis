import cv2
import mediapipe as mp
import time

class HandGesture:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_call_time = 0

    def process(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        message = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Check for "Call" gesture (Thumb and Pinky extended, others folded)
                # Fingers: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
                # Tips vs PIP/MCP joints
                
                # Thumb: Tip(4) to right of IP(3) (assuming right hand facing camera?) 
                # Actually media pipe hands are relative.
                # Simple logic: Tip distance from palm base (0) vs PIP distance?
                # Better: Check if fingers are extended or folded.
                
                lm = hand_landmarks.landmark
                
                # 1. Check if Index, Middle, Ring are folded
                # Tip (8,12,16) should be lower (higher y value) than PIP (6,10,14) for a vertically held hand?
                # Robust way: Tip distance to wrist (0) < PIP distance to wrist (0)
                
                def dist(i, j):
                    return ((lm[i].x - lm[j].x)**2 + (lm[i].y - lm[j].y)**2)**0.5
                
                is_index_folded = dist(8, 0) < dist(6, 0)
                is_middle_folded = dist(12, 0) < dist(10, 0)
                is_ring_folded = dist(16, 0) < dist(14, 0)
                
                # 2. Check if Thumb and Pinky are extended
                is_thumb_extended = dist(4, 0) > dist(3, 0) # Rough check
                is_pinky_extended = dist(20, 0) > dist(18, 0)
                
                if is_index_folded and is_middle_folded and is_ring_folded and is_thumb_extended and is_pinky_extended:
                    current_time = time.time()
                    if current_time - self.last_call_time > 2.0: # Debounce
                        print("Calling Mockup")
                        self.last_call_time = current_time
                        message = "Calling Mockup"
                    
                    cv2.putText(frame, "CALL DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        return frame, message

if __name__ == "__main__":
    from config import CAM_ID_RIGHT
    cap = cv2.VideoCapture(CAM_ID_RIGHT)
    detector = HandGesture()
    
    print("Hand Gesture Test. Show 'Call' gesture (Thumb+Pinky).")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Mirror for self-view comfort? Or keep raw? User said "right hand gesture".
        # Let's keep it raw but maybe rotate?
        # Main.py rotates right camera 180.
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        frame, msg = detector.process(frame)
        
        cv2.imshow("Hand", frame)
        if cv2.waitKey(1) == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
