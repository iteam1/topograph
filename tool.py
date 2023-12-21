import cv2
import sys
import numpy as np

def empty(i):
    pass

def on_trackbar(val):
    global img
    
    # get values from trackbar
    BLUR_KERNEL = cv2.getTrackbarPos("BLUR_KERNEL", "TrackedBars")
    MIN_THRESH = cv2.getTrackbarPos("MIN_THRESH", "TrackedBars")
    MAX_THRESH = cv2.getTrackbarPos("MAX_THRESH", "TrackedBars")
    BOX_SIZE = cv2.getTrackbarPos("BOX_SIZE", "TrackedBars")
    CUT_OFF = cv2.getTrackbarPos("CUT_OFF", "TrackedBars")
    SCALE = cv2.getTrackbarPos("SCALE", "TrackedBars")
    
    # preprocess
    if BLUR_KERNEL %2 == 0:
        BLUR_KERNEL = BLUR_KERNEL + 1
        
    CUT_OFF = CUT_OFF /100
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to graycsale
    img_blur = cv2.GaussianBlur(img_gray, (BLUR_KERNEL,BLUR_KERNEL), 0) # Blur the image for better edge detection
    edges = cv2.Canny(image=img_blur, threshold1=MIN_THRESH, threshold2=MAX_THRESH) # Canny Edge Detection
    
    # Create empty
    ground = np.zeros_like(edges,dtype=np.uint16)
    
    # reshape the image
    H,W,c = img.shape
    
    # topograph
    print('processing...')
    for c in range(H):
        for r in range(W):
            # Create empty layer
            layer = np.zeros_like(edges)
            
            # Get current value pixel
            current_pixel = edges[c,r]
            if current_pixel == 255:
                
                # Set region in current layer
                layer[c-BOX_SIZE:c+BOX_SIZE,r-BOX_SIZE:r+BOX_SIZE] = 255
                
                # Accumulate current layer
                ground += layer

    # Inspect result
    MEAN_VAL = ground.mean()
    MIN_VAL = ground.min()
    MAX_VAL = ground.max()
    D = MAX_VAL - MIN_VAL

    norm_ground = ground / D
    norm_ground[norm_ground < CUT_OFF] = 0 # cut off
    
    img_norm = norm_ground * 255
    img_norm = img_norm.astype(np.uint8)

    # Heatmap
    img_heatmap = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)
    
    # Find contour
    # Export
    out = cv2.addWeighted(img,0.5,img_heatmap,0.5,0.0)

    # display result
    resized = cv2.resize(out,(0,0),fx=SCALE/100,fy=SCALE/100)
    cv2.imshow("res",out)

if __name__ == "__main__":
    
    # read input the images
    img_path = sys.argv[1]
        
    # stack the images
    img = cv2.imread(img_path)
    
    # create window
    cv2.namedWindow("TrackedBars")
    cv2.resizeWindow("TrackedBars", 640, 240)
    
    # create trackbars
    cv2.createTrackbar("BLUR_KERNEL", "TrackedBars", 3, 20, on_trackbar)
    cv2.createTrackbar("MIN_THRESH", "TrackedBars", 100, 150, on_trackbar)
    cv2.createTrackbar("MAX_THRESH", "TrackedBars", 151, 200, on_trackbar)
    cv2.createTrackbar("BOX_SIZE", "TrackedBars", 5, 200, on_trackbar)
    cv2.createTrackbar("CUT_OFF", "TrackedBars", 50, 100, on_trackbar)
    cv2.createTrackbar("SCALE", "TrackedBars",90, 100, on_trackbar)

    # show some stuff
    on_trackbar(0)

    # wait until user press any key
    k = cv2.waitKey()