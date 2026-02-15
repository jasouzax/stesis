import cv2
import numpy as np
import json
import sys
from config import (CAM_ID_LEFT, CAM_ID_RIGHT, BASELINE, MIN_DISPARITY, 
                    NUM_DISPARITIES, BLOCK_SIZE, LEFT_OFFSET, SCALE, USE_WLS)

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
            M[0, 0] *= scale; M[1, 1] *= scale
            M[0, 2] *= scale; M[1, 2] *= scale
        
        Q[0, 3] *= scale; Q[1, 3] *= scale; Q[2, 3] *= scale 
        
        # Q matrix structure for reprojectImageTo3D:
        # [[1, 0, 0, -cx]
        #  [0, 1, 0, -cy]
        #  [0, 0, 0,  f]
        #  [0, 0, 1/Tx, (cx-cx')/Tx]]
        # Focal length is usually at Q[2,3], but for raw calculation we use M1[0,0]
        focal_length = M1[0,0] 
        return (M1, d1, M2, d2, R1, P1, R2, P2, Q, w, h, focal_length)
    except FileNotFoundError:
        print(f"[ERROR] Calibration file '{filename}' missing. Run calibrate.py first.")
        sys.exit(1)

def create_stereo_matcher():
    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE**2,
        P2=32 * 3 * BLOCK_SIZE**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    wls_filter = None
    right_matcher = None
    
    if USE_WLS:
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(stereo)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
            wls_filter.setLambda(8000.0)
            wls_filter.setSigmaColor(1.5)
            print("WLS Filter Enabled")
        except Exception as e:
            print(f"WLS Filter unavailable: {e}")
            return stereo, None, None
            
    return stereo, right_matcher, wls_filter

def compute_depth_map(img_l, img_r, stereo, right_matcher=None, wls_filter=None):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if wls_filter and right_matcher:
        disp_l = stereo.compute(gray_l, gray_r)
        # For WLS we need the right disparity map too
        # Note: right_matcher.compute requires (rect_r, rect_l)
        disp_r = right_matcher.compute(gray_r, gray_l)
        disparity_16S = wls_filter.filter(disp_l, gray_l, disparity_map_right=disp_r)
        disparity = disparity_16S.astype(np.float32) / 16.0
    else:
        disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        
    return disparity

def get_visual_depth(disparity):
    disp_vis = (disparity - MIN_DISPARITY) / NUM_DISPARITIES
    disp_vis = np.clip(disp_vis * 255, 0, 255).astype(np.uint8)
    depth_view = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    depth_view[disparity <= 0] = [0, 0, 0] 
    depth_view[:, :LEFT_OFFSET] = [0, 0, 0]
    return depth_view

if __name__ == "__main__":
    import calibrate # Import here to avoid circular dependencies if any, and for visualization usage
    print("--- Depth Map Debug Mode ---")
    
    (M1, d1, M2, d2, R1, P1, R2, P2, Q, width, height, focal_length) = load_calibration(BASELINE, SCALE)
    
    map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (width, height), cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (width, height), cv2.CV_16SC2)
    
    stereo, right_matcher, wls_filter = create_stereo_matcher()
    
    cap_l = cv2.VideoCapture(CAM_ID_LEFT)
    cap_r = cv2.VideoCapture(CAM_ID_RIGHT)
    
    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r: break
        
        frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
        img_l = cv2.resize(frame_l, (width, height), interpolation=cv2.INTER_AREA)
        img_r = cv2.resize(frame_r, (width, height), interpolation=cv2.INTER_AREA)

        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)
        
        # Compute Depth
        disparity = compute_depth_map(rect_l, rect_r, stereo, right_matcher, wls_filter)
        depth_view = get_visual_depth(disparity)
        
        # Visualize
        vis_l = calibrate.draw_overlays(rect_l.copy(), mode='lines')
        vis_r = calibrate.draw_overlays(rect_r.copy(), mode='lines')
        
        # Create Layout
        # Top: Stereo Feeds
        top_row = np.hstack((vis_l, vis_r))
        # Bottom: Depth Map (Stretched to match width)
        # Depth map is width x height. Top row is 2*width x height.
        # Let's resize depth map to match top row width? Or just center it?
        # User said "bottom is the generate depth map". 
        # Let's double the width of depth map to fit?
        depth_doubled = cv2.resize(depth_view, (width*2, height*2), interpolation=cv2.INTER_NEAREST)
        
        # Actually user said "top left and right... bottom is the generate depth map"
        # Since top row has 2 images, bottom being 1 image might look weird if not same width.
        # Let's display depth view centered or stretched.
        # Simplest: Two separate windows or `np.vstack` with resizing.
        # Let's try to fit it nicely.
        
        combined_img = np.vstack((top_row, depth_doubled))
        # Wait, if depth_doubled is 2*h, it's too tall.
        # Let's just scale width by 2, keep height same?
        depth_wide = cv2.resize(depth_view, (width*2, height))
        final_ui = np.vstack((top_row, depth_wide))
                   
        cv2.imshow("Depth Debug", final_ui)
        if cv2.waitKey(1) == ord('q'): break
        
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
