import cv2
import numpy as np
img = "img/test.png"

image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
image = np.astype(np.uint8, dtype=np.uint8)
cv2.imshow("image", image)
cv2.imwrite("new.png", image)