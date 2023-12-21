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
BOX_SIZE = 30
CUT_OFF = 0.7 # threshold = CUT_OFF * MAX_VAL

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

# Inspect result
MEAN_VAL = ground.mean()
MIN_VAL = ground.min()
MAX_VAL = ground.max()
D = MAX_VAL - MIN_VAL

mask = np.zeros_like(edges)

norm_ground = ground / D
mask[norm_ground > CUT_OFF] = (255)
img_norm = norm_ground * 255
img_norm = img_norm.astype(np.uint8)
img_heatmap = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET) # Heatmap

# find contours for overlap
blank = np.zeros_like(edges)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(blank,(x,y),(x+w,y+h),(255),-1)
    
contours, hierarchy = cv2.findContours(blank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# find the biggest countour (c) by the area
c = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(img,(x,y),(x+w,y+h),(255),5)

# Export
out = cv2.addWeighted(img,0.8,img_heatmap,0.2,0.0)
cv2.imwrite('out.jpg',out)

end_time = time.time() - start_time
print('elapsed time:',end_time)