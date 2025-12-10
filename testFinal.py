import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

imgPath = "../Vent_Int/MultiInflation_150083_mr_sVent_medfilt_3.nii.gz"
maskPath = "../seg_complete/150083_RV_mask.nii.gz"

img = sitk.ReadImage(imgPath)
imgArray = sitk.GetArrayFromImage(img)
mask = sitk.ReadImage(maskPath)
maskArray = sitk.GetArrayFromImage(mask)

slice_idx = 32
slicedImg = imgArray[slice_idx, :, :]
#slicedMask = maskArray[slice_idx, :, :]
"""
orig_h, orig_w = imgArray.shape
orig_h, orig_w = maskArray.shape"""

depth, height, width = imgArray.shape


orig_h, orig_w = slicedImg.shape

#segment = np.where(slicedMask > 0, slicedImg, 0)

segment = np.where(maskArray, imgArray, np.nan)

def padding(imgArray):
    depth, height, width = imgArray.shape
    size = 2 ** int(np.ceil(np.log2(max(orig_h, orig_w))))
    padded = np.zeros((size, size), dtype=imgArray.dtype)
    padded[:depth, :height, :width] = imgArray
    return padded, depth, height, width, size

paddedSegment, orig_h, orig_w, segSize = padding(segment)

valid_pixels = paddedSegment[paddedSegment > np.min(paddedSegment)]
fullRange = np.max(valid_pixels) - np.min(valid_pixels)
percentage = 10
globalStddev = fullRange * (percentage / 100)

result = np.zeros_like(paddedSegment, dtype=np.float32)

def quadtree_weighted_fill(img, output, x, y, size, orig_w, orig_h, globalStddev, segSize):
    if x >= orig_w or y >= orig_h or size <= 1:
        return

    region = img[y:y+size, x:x+size]
    regionMax = np.max(region)
    regionMin = np.min(region)
    regionRange = regionMax - regionMin

    x2 = min(x + size, orig_w)
    y2 = min(y + size, orig_h)
    h, w = y2 - y, x2 - x
    regionCrop = region[0:h, 0:w]

    if regionRange <= globalStddev or np.all(regionCrop == 0) or regionRange < 1e-5:
        weight = 1.0 - (size / segSize)
        weight = np.clip(weight, 0.4, 1.0)

        if np.all(regionCrop == 0):
            fill_value = 0
        else:
            fill_value = np.mean(regionCrop) * weight

        output[y:y2, x:x2] = fill_value
    else:
        half = size // 2
        quadtree_weighted_fill(img, output, x, y, half, orig_w, orig_h, globalStddev, segSize)
        quadtree_weighted_fill(img, output, x + half, y, half, orig_w, orig_h, globalStddev, segSize)
        quadtree_weighted_fill(img, output, x, y + half, half, orig_w, orig_h, globalStddev, segSize)
        quadtree_weighted_fill(img, output, x + half, y + half, half, orig_w, orig_h, globalStddev, segSize)

quadtree_weighted_fill(paddedSegment, result, 0, 0, segSize, orig_w, orig_h, globalStddev, segSize)

final = result[:orig_h, :orig_w]
final[final < 1e-4] = 0
final = final - final.min()
final = final / (final.max() + 1e-5)

# 시각화
plt.figure(figsize=(6, 6))
plt.imshow(final, cmap='gray', origin='lower')
plt.axis('off')
plt.tight_layout()
plt.show()
