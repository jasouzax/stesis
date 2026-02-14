import cv2
import numpy as np
import sys
import json
import time
from config import CAM_ID_LEFT, CAM_ID_RIGHT, BASELINE, MIN_DISPARITY, NUM_DISPARITIES, BLOCK_SIZE

# --- PERFORMANCE CONFIG ---
SCALE = 0.25  
USE_WLS = True 

def update_params(val):
    pass

def load_calibration(baseline_cm, scale=1.0):
    filename = f"stereo-{baseline_cm}.json"
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        M1, d1 = np.array(data["M1"]), np.array(data["d1"])
        M2, d2 = np.array(data["M2"]), np.array(data["d2"])
        R1, P1 = np.array(data["R1"]), np.array(data["P1"])
        R2, P2 = np.array(data["R2"]), np.array(data["P2"])
        Q = np.array(data["Q"])

        w = int(data["width"] * scale)
        h = int(data["height"] * scale)

        for M in [M1, M2, P1, P2]:
            M[0, 0] *= scale  # fx
            M[1, 1] *= scale  # fy
            M[0, 2] *= scale  # cx
            M[1, 2] *= scale  # cy
        
        Q[0, 3] *= scale 
        Q[1, 3] *= scale 
        Q[2, 3] *= scale 

        return (M1, d1, M2, d2, R1, P1, R2, P2, Q, w, h), data.get("pixel_error", "N/A")
    except FileNotFoundError:
        print(f"Error: Calibration file '{filename}' not found.")
        sys.exit(1)

def main():
    (M1, d1, M2, d2, R1, P1, R2, P2, Q, width, height), pixel_error = load_calibration(BASELINE, SCALE)

    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)
    cap_l.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_r.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (width, height), cv2.CV_16SC2)

    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE**2,
        P2=32 * 3 * BLOCK_SIZE**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    wls_filter = None
    if USE_WLS:
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(stereo)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
            wls_filter.setLambda(8000.0)
            wls_filter.setSigmaColor(1.5)
        except:
            print("WLS Filter unavailable.")

    cv2.namedWindow("Stereo Depth")
    cv2.createTrackbar("Num Disp (16x)", "Stereo Depth", NUM_DISPARITIES//16, 16, update_params)

    while True:
        start_time = time.time()
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r: break

        frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
        img_l = cv2.resize(frame_l, (width, height), interpolation=cv2.INTER_AREA)
        img_r = cv2.resize(frame_r, (width, height), interpolation=cv2.INTER_AREA)

        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        num_disp = cv2.getTrackbarPos("Num Disp (16x)", "Stereo Depth") * 16
        num_disp = max(16, num_disp)
        stereo.setNumDisparities(num_disp)

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        if wls_filter:
            disp_l = stereo.compute(gray_l, gray_r)
            right_matcher.setNumDisparities(num_disp)
            disp_r = right_matcher.compute(gray_r, gray_l)
            disparity = wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r).astype(np.float32) / 16.0
        else:
            disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # --- RESTORED STABLE VISUALIZATION ---
        # 1. Use your original static math
        disp_vis = (disparity - MIN_DISPARITY) / num_disp
        disp_vis = np.clip(disp_vis * 255, 0, 255).astype(np.uint8)
        
        # 2. Apply Colormap
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        
        # 3. MASK: Force invalid/noisy pixels (disparity <= 0) to Black
        disp_color[disparity <= 0] = [0, 0, 0]

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(disp_color, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Stereo Depth", np.hstack((rect_l, disp_color)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
