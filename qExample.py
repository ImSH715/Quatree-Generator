import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

img_path = "img/test.png"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

def pad_to_power_of_2(img):
    h, w = img.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[:h, :w] = img
    return padded

padded = pad_to_power_of_2(img)
size = padded.shape[0]

def is_homogeneous(region, eps=2):
    return np.std(region) < eps or np.all(region == region.flat[0])

def quadtree(ax, img, x, y, size, orig_w, orig_h, eps=2):
    region = img[y:y+size, x:x+size]

    if is_homogeneous(region, eps):
        x2, y2 = x + size, y + size
        if x < orig_w and y < orig_h:
            ax.plot([x, x2], [y, y], color='white', lw=0.5)
            ax.plot([x, x2], [y2, y2], color='white', lw=0.5)
            ax.plot([x, x], [y, y2], color='white', lw=0.5)
            ax.plot([x2, x2], [y, y2], color='white', lw=0.5)
        return

    half = size // 2
    quadtree(ax, img, x, y, half, orig_w, orig_h, eps)
    quadtree(ax, img, x+half, y, half, orig_w, orig_h, eps)
    quadtree(ax, img, x, y+half, half, orig_w, orig_h, eps)
    quadtree(ax, img, x+half, y+half, half, orig_w, orig_h, eps)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(padded, cmap='gray')
quadtree(ax, padded, 0, 0, size, w, h, eps=2)
ax.axis('off')
plt.tight_layout()
plt.show()
