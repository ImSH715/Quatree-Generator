import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

imgPath = "Z:/polaris2/students/Will_Students_2025/acb20si/Quadtree/Dataset/P-T4LJ6/mrA/Images/P-T4LJ6_SPGR_FRC+bag.nii.gz"
img = sitk.ReadImage(imgPath)
imgArray = sitk.GetArrayFromImage(img)

slicedImg = imgArray[219, :, :]
stretchedImg = np.repeat(slicedImg, 2, axis=0)
orig_h, orig_w = stretchedImg.shape

def padding(img):
    h, w = img.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[:h, :w] = img
    return padded



paddedImg = padding(stretchedImg)
paddedSize = paddedImg.shape[0]


globalStddev = np.std(paddedImg)

valid_pixels = paddedImg[paddedImg > np.min(paddedImg)]
fullRange = np.max(valid_pixels) - np.min(valid_pixels)
print(valid_pixels)
print("Full intensity range:", fullRange)

def quadtree_decompose(ax, img, x, y, size, orig_w, orig_h, fullRange, globalStddev):
    if x >= orig_w or y >= orig_h or size <= 1:
        return

    region = img[y:y+size, x:x+size]
    regionRange = np.max(region) - np.min(region)
    normalizedRange = regionRange / fullRange if fullRange > 0 else 0

    x1, y1 = x, y
    x2, y2 = min(x + size, orig_w), min(y + size, orig_h)

    print()

    if normalizedRange <= globalStddev:
        ax.plot([x1, x2], [y1, y1], color='white', linewidth=0.5)
        ax.plot([x1, x2], [y2, y2], color='white', linewidth=0.5)
        ax.plot([x1, x1], [y1, y2], color='white', linewidth=0.5)
        ax.plot([x2, x2], [y1, y2], color='white', linewidth=0.5)
    else:
        half = size // 2
        quadtree_decompose(ax, img, x, y, half, orig_w, orig_h, globalStddev)
        quadtree_decompose(ax, img, x + half, y, half, orig_w, orig_h, globalStddev)
        quadtree_decompose(ax, img, x, y + half, half, orig_w, orig_h, globalStddev)
        quadtree_decompose(ax, img, x + half, y + half, half, orig_w, orig_h, globalStddev)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(stretchedImg, cmap='gray', origin="lower")
quadtree_decompose(ax, stretchedImg, 0, 0, paddedSize, orig_w, orig_h, fullRange, globalStddev=globalStddev)
ax.axis('off')
plt.tight_layout()
plt.show()
