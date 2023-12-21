'''
python3 main.py images/22112023203029_cam_box_side_1_box.jpg
'''
import sys
import cv2
import time
import numpy as np

start_time = time.time()

# Init
img_path = sys.argv[1]
BLUR_KERNEL = 3
MIN_THRESH = 100
MAX_THRESH = 200
BOX_SIZE = 20
CUT_OFF = 0.5 # threshold = CUT_OFF * MAX_VAL

# Read image
img = cv2.imread(img_path)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (BLUR_KERNEL,BLUR_KERNEL), 0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=MIN_THRESH, threshold2=MAX_THRESH) # Canny Edge Detection
print(edges.shape, edges.min(), edges.max())
H,W = edges.shape

# Create empty
ground = np.zeros_like(edges,dtype=np.uint16)

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

#cv2.imwrite('heatmap_8.jpg', cv2.applyColorMap(ground.astype(np.uint8), cv2.COLORMAP_JET))

# Inspect result
MEAN_VAL = ground.mean()
MIN_VAL = ground.min()
MAX_VAL = ground.max()
D = MAX_VAL - MIN_VAL
THRESHOLD_VAL = int(CUT_OFF * D)
print(ground.shape, MIN_VAL, MAX_VAL, D, MEAN_VAL, THRESHOLD_VAL)

norm_ground = ground / D
print(norm_ground.shape, norm_ground.min(), norm_ground.max())
img_norm = norm_ground * 255
img_norm = img_norm.astype(np.uint8)

# Heatmap
img_heatmap = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)

# Export
out = cv2.addWeighted(img,0.5,img_heatmap,0.5,0.0)
cv2.imwrite('out.jpg',out)

end_time = time.time() - start_time
print('elapsed time:',end_time)