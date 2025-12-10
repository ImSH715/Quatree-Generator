import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

imgPath = "../Images/P-T4LJ6_SPGR_FRC+bag.nii.gz"
img = sitk.ReadImage(imgPath)
imgArray = sitk.GetArrayFromImage(img)
maskPath = "../Masks/PT4LJ6_SPGR_FRC+bag_seg.nii.gz"
mask = sitk.ReadImage(maskPath)
maskArray = sitk.GetArrayFromImage(mask)

slicedImg = imgArray[:, 40, :]
slicedMask = maskArray[:, 40, :]
masked = np.where(slicedMask > 0, slicedImg, 0)

plt.imshow(masked)
plt.show()
orig_h, orig_w = slicedImg.shape
imgSize = slicedImg.shape[0]

def padding(img):
    h, w = img.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[:h, :w] = img
    return padded

paddedImg = padding(slicedImg)
paddedSize = paddedImg.shape[0]
globalStddev = np.std(paddedImg)
print(globalStddev)

def quadtree_decompose(ax, img, x, y, size, orig_w, orig_h, globalStddev):
    if x >= orig_w or y >= orig_h or size <= 1:
        return

    region = img[y:y+size, x:x+size]
    regionStddev = np.std(region)
    if globalStddev == 0:
        normalizedStddev = 0
    else:
        normalizedStddev = regionStddev / globalStddev

    x1, y1 = x, y
    x2, y2 = min(x + size, orig_w), min(y + size, orig_h)

    if normalizedStddev < 1.0:
        ax.plot([x1, x2], [y1, y1], color='white', linewidth=1.5)
        ax.plot([x1, x2], [y2, y2], color='white', linewidth=1.5)
        ax.plot([x1, x1], [y1, y2], color='white', linewidth=1.5)
        ax.plot([x2, x2], [y1, y2], color='white', linewidth=1.5)
    else:
        half = size // 2
        quadtree_decompose(ax, img, x, y, half, orig_w, orig_h, globalStddev)
        quadtree_decompose(ax, img, x + half, y, half, orig_w, orig_h, globalStddev)
        quadtree_decompose(ax, img, x, y + half, half, orig_w, orig_h, globalStddev)
        quadtree_decompose(ax, img, x + half, y + half, half, orig_w, orig_h, globalStddev)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(slicedImg, cmap='gray', origin = "lower")
quadtree_decompose(ax, slicedImg, 0, 0, paddedSize, orig_w=orig_w, orig_h=orig_h, globalStddev=50)
ax.axis('off')
plt.tight_layout()
plt.show()
