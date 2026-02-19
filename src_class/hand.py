import cv2
import mediapipe as mp
import math
from camera import Camera

class Hand(Camera):
    def __init__(self):
        super().__init__()
        self.mp_hands = None
        self.hands = None
        self.mp_drawing = None
        
        self.hand_view = None
        self.hand_message = None

    def setup(self):
        super().setup()
        print("Initializing Hand Tracking...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def is_call_gesture(self, landmarks):
        wrist = landmarks.landmark[0]
        
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        
        ring_tip = landmarks.landmark[16]
        ring_pip = landmarks.landmark[14]
        
        pinky_tip = landmarks.landmark[20]
        pinky_pip = landmarks.landmark[18]
        
        index_curled = self.dist(wrist, index_tip) < self.dist(wrist, index_pip)
        middle_curled = self.dist(wrist, middle_tip) < self.dist(wrist, middle_pip)
        ring_curled = self.dist(wrist, ring_tip) < self.dist(wrist, ring_pip)
        
        pinky_extended = self.dist(wrist, pinky_tip) > self.dist(wrist, pinky_pip)
        thumb_extended = self.dist(wrist, thumb_tip) > self.dist(wrist, thumb_ip)
        
        if index_curled and middle_curled and ring_curled and pinky_extended and thumb_extended:
            return True
        return False

    def loop(self):
        super().loop()
        
        # Hand works on Right Camera (frames[1])
        if self.frames[1] is None: return

        frame = self.frames[1].copy()
        
        # MediaPipe needs RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        self.hand_message = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                if self.is_call_gesture(hand_landmarks):
                    self.hand_message = "Calling Mockup"
                    cv2.putText(frame, self.hand_message, (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        self.hand_view = frame

    def cleanup(self):
        super().cleanup()
        if self.hands: self.hands.close()

if __name__ == "__main__":
    class HandView(Hand):
        def loop(self):
            super().loop()
            if self.hand_view is not None:
                cv2.imshow("Hand Class Test", self.hand_view)
                
    app = HandView()
    app.run()
