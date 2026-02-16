import cv2
import mediapipe as mp
import math

class HandGesture:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process(self, frame):
        """
        Process the frame to detect hands and gestures.
        Returns:
            annotated_frame: Frame with landmarks drawn.
            message: String message if a specific gesture is detected, else None.
        """
        # MediaPipe works with RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        annotated_frame = frame.copy()
        message = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Check for "Call" Gesture
                # Thumb and Pinky Extended, Index/Middle/Ring Curled.
                if self.is_call_gesture(hand_landmarks):
                    message = "Calling Mockup"
                    cv2.putText(annotated_frame, message, (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        return annotated_frame, message

    def is_call_gesture(self, landmarks):
        # Tips: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
        # PIPs (Knuckles/Joints): Index=6, Middle=10, Ring=14
        
        # Simple Logic:
        # Extended: Tip is further from wrist (0) than PIP/MCP.
        # Curled: Tip is closer to wrist than PIP/MCP.
        
        # Wrist
        wrist = landmarks.landmark[0]
        
        # Thumb (4) - Check relative to MCP (2) or IP (3)
        # Thumb is tricky, usually check x-distance relative to wrist/MCP for extension?
        # Or just check if Tip(4) is far from Index MCP(5).
        # Let's use simple distance from wrist for now, though orientation matters.
        # Better: Check specific landmark positions.
        
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_mcp = landmarks.landmark[2]
        
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        
        ring_tip = landmarks.landmark[16]
        ring_pip = landmarks.landmark[14]
        
        pinky_tip = landmarks.landmark[20]
        pinky_pip = landmarks.landmark[18]
        
        # Vector geometry is more robust, but straight y-coord check works for upright hand.
        # Assuming hand is roughly upright (Wrist at bottom).
        
        # Curled Fingers (Tip below PIP)
        # Note: Y increases downwards in image coords.
        # So "Above" mean y_tip < y_pip. "Below" means y_tip > y_pip.
        
        # NOTE: This assumes an upright hand. If hand is sideways, logic needs vectors.
        # User requirement is simple "Call" gesture.
        
        # "Call" / "Shaka": Thumb & Pinky Extended. Index, Middle, Ring Curled.
        
        # Heuristic:
        # Piny and Thumb tips should be 'far' from wrist/center.
        # Index, Middle, Ring tips should be 'close' to palm center/wrist.
        
        def dist(p1, p2):
            return math.hypot(p1.x - p2.x, p1.y - p2.y)
            
        d_wrist_index = dist(wrist, index_tip)
        d_wrist_middle = dist(wrist, middle_tip)
        d_wrist_ring = dist(wrist, ring_tip)
        d_wrist_pinky = dist(wrist, pinky_tip)
        d_wrist_thumb = dist(wrist, thumb_tip)
        
        # Checking if extended fingers are further than curled ones
        # and checking curl condition (Tip close to palm base/wrist)
        
        # Condition 1: Pinky and Thumb extended
        # Comparing against their own PIPs is better for 'extended' status regardless of rotation (mostly).
        # But 'curled' usually implies tip is closer to wrist than PIP is.
        
        # Let's use the MCP-Tip vs Wrist-MCP distance ratios or similar?
        # Simplest consistent check:
        # Is finger extended?
        # Index: dist(wrist, tip) < dist(wrist, pip) ? (Curled)
        # Middle: dist(wrist, tip) < dist(wrist, pip) ? (Curled)
        # Ring: dist(wrist, tip) < dist(wrist, pip) ? (Curled)
        
        # Thumb & Pinky Extended:
        # dist(wrist, tip) > dist(wrist, ip/pip)
        
        # This works for "curled into palm".
        
        index_curled = dist(wrist, index_tip) < dist(wrist, index_pip)
        middle_curled = dist(wrist, middle_tip) < dist(wrist, middle_pip)
        ring_curled = dist(wrist, ring_tip) < dist(wrist, ring_pip)
        
        pinky_extended = dist(wrist, pinky_tip) > dist(wrist, pinky_pip)
        thumb_extended = dist(wrist, thumb_tip) > dist(wrist, thumb_ip) # IP for thumb
        
        if index_curled and middle_curled and ring_curled and pinky_extended and thumb_extended:
            return True
            
        return False
        
    def close(self):
        self.hands.close()
