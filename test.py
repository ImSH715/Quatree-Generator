import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches


"""

Quad Tree Massk Creater


"""


imgPath = "../Vent_Int/MultiInflation_150083_mr_sVent_medfilt_3.nii.gz"
maskPath = "../seg_complete/150083_RV_mask.nii.gz"

img = sitk.ReadImage(imgPath)
imgArray = sitk.GetArrayFromImage(img)
slicedImg = imgArray[32, :, :]

mask = sitk.ReadImage(maskPath)
maskArray = sitk.GetArrayFromImage(mask)
slicedMask = maskArray[32, :, :]
segment = np.where(slicedMask > 0, slicedImg, 0)

def padding(img):
    h, w = img.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[:h, :w] = img
    return padded

padSegment = padding(segment)
orig_h, orig_w = slicedImg.shape
segSize = padSegment.shape[0]

valid_pixels = padSegment[padSegment > np.min(padSegment)]
fullRange = np.max(valid_pixels) - np.min(valid_pixels)
percentage = 10
globalStddev = fullRange * (percentage / 100)

max_level = [0]
def quadtree_decompose(ax, img, x, y, size, orig_w, orig_h, globalStddev, level=0, max_level=[0]):
    if x >= orig_w or y >= orig_h or size <= 1:
        return

    region = img[y:y+size, x:x+size]
    regionMax = np.max(region)
    regionMin = np.min(region)
    regionRange = regionMax - regionMin
    x1, y1 = x, y
    x2, y2 = min(x + size, orig_w), min(y + size, orig_h)

    if regionRange <= globalStddev:
        max_level[0] = max(max_level[0], level)
    else:
        half = size // 2
        quadtree_decompose(ax, img, x, y, half, orig_w, orig_h, globalStddev, level + 1, max_level)
        quadtree_decompose(ax, img, x + half, y, half, orig_w, orig_h, globalStddev, level + 1, max_level)
        quadtree_decompose(ax, img, x, y + half, half, orig_w, orig_h, globalStddev, level + 1, max_level)
        quadtree_decompose(ax, img, x + half, y + half, half, orig_w, orig_h, globalStddev, level + 1, max_level)

def quadtree_draw(ax, img, x, y, size, orig_w, orig_h, globalStddev, level=0):
    if x >= orig_w or y >= orig_h or size <= 1:
        return

    region = img[y:y+size, x:x+size]
    regionMax = np.max(region)
    regionMin = np.min(region)
    regionRange = regionMax - regionMin
    x1, y1 = x, y
    x2, y2 = min(x + size, orig_w), min(y + size, orig_h)

    if regionRange <= globalStddev:
        norm_level = level / max_level[0] if max_level[0] != 0 else 0
        grayValue = norm_level 
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=0, edgecolor=None,
                                 facecolor=str(grayValue), alpha=1.0)
        ax.add_patch(rect)
    else:
        half = size // 2
        quadtree_draw(ax, img, x, y, half, orig_w, orig_h, globalStddev, level + 1)
        quadtree_draw(ax, img, x + half, y, half, orig_w, orig_h, globalStddev, level + 1)
        quadtree_draw(ax, img, x, y + half, half, orig_w, orig_h, globalStddev, level + 1)
        quadtree_draw(ax, img, x + half, y + half, half, orig_w, orig_h, globalStddev, level + 1)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(segment, cmap='gray', origin='lower')

quadtree_decompose(ax, padSegment, 0, 0, segSize, orig_w, orig_h, globalStddev, level=0, max_level=max_level)

ax.clear()
ax.imshow(segment, cmap='gray', origin='lower')
quadtree_draw(ax, padSegment, 0, 0, segSize, orig_w, orig_h, globalStddev, level=0)

ax.axis('off')
plt.tight_layout()
plt.show()
