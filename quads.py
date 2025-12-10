import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# MRI Img
# Raw data
imgPath = "../mrA/Reg_TLC_2_RV/Vent_Int/MultiInflation_150083_mr_sVent_medfilt_3.nii.gz"
maskPath = "../150083/mrA/seg_complete/150083_RV_mask.nii.gz"
outputPath = "../P-T4LJ6/mrA/PT4LJ6_SPGR_FRC+bag_seg.nii.gz"

img = sitk.ReadImage(imgPath)

imgArray = sitk.GetArrayFromImage(img)

slicedImg = imgArray[32, :, :]

orig_h, orig_w = slicedImg.shape
imgSize = slicedImg.shape[0]

# Mask
mask = sitk.ReadImage(maskPath)

maskArray = sitk.GetArrayFromImage(mask)

slicedMask = maskArray[32, :, :]

#Excluded
segment = np.where(slicedMask > 0, slicedImg, 0)


def padding(img):
    h, w = img.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[:h, :w] = img
    return padded

padSegment = padding(segment)
segSize = padSegment.shape[0]
#Global_standard deviation
globalStddev = np.std(padSegment)

#percentage
valid_pixels = padSegment[padSegment > np.min(padSegment)]
fullRange = np.max(valid_pixels) - np.min(valid_pixels)

percentage = 10

globalStddev = fullRange*(percentage/100)
print(globalStddev)
def quadtree_decompose(ax, img, x, y, size, orig_w, orig_h, globalStddev):
    if x >= orig_w or y >= orig_h or size <= 1:
        return
    
    region = img[y:y+size, x:x+size]
    regionMax = np.max(region)
    regionMin = np.min(region)
    regionRange = regionMax - regionMin
    x1, y1 = x, y
    x2, y2 = min(x + size, orig_w), min(y + size, orig_h)

    if regionRange <= globalStddev:
        meanIntensity = np.mean(region)
        grayValue = 1.0 - (meanIntensity - np.min(valid_pixels)) / regionRange
        grayValue = np.clip(grayValue,0,1)
        """
        Border lines
        ax.plot([x1, x2], [y1, y1], color='white', linewidth=0.5)
        ax.plot([x1, x2], [y2, y2], color='white', linewidth=0.5)
        ax.plot([x1, x1], [y1, y2], color='white', linewidth=0.5)
        ax.plot([x2, x2], [y1, y2], color='white', linewidth=0.5)
        """
    else: 
        half = size // 2
        quadtree_decompose(ax, img, x, y, half, orig_w, orig_h, globalStddev)
        quadtree_decompose(ax, img, x + half, y, half, orig_w, orig_h, globalStddev)
        quadtree_decompose(ax, img, x, y + half, half, orig_w, orig_h, globalStddev)
        quadtree_decompose(ax, img, x + half, y + half, half, orig_w, orig_h, globalStddev)


fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(segment, cmap='gray', origin = "lower")
quadtree_decompose(ax, segment, 0, 0, segSize, orig_w=orig_w, orig_h=orig_h, globalStddev=globalStddev)

ax.axis('off')
plt.tight_layout()
plt.show()