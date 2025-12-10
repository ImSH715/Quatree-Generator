import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

imgPath = "../MultiInflation_150083_mr_sVent_medfilt_3.nii.gz"
maskPath = "../seg_complete/150083_RV_mask.nii.gz"
outputPath = "../PT4LJ6_SPGR_FRC+bag_seg.nii.gz"

img = sitk.ReadImage(imgPath)
imgArray = sitk.GetArrayFromImage(img)

mask = sitk.ReadImage(maskPath)
maskArray = sitk.GetArrayFromImage(mask)

depth, height, width = imgArray.shape

def calculate_mean(img):
    return np.array([[np.mean(img)]])

def split4(img):
    h, w = img.shape
    h2, w2 = h // 2, w // 2
    return [
        img[:h2, :w2], 
        img[:h2, w2:],  
        img[h2:, :w2],  
        img[h2:, w2:], 
    ]

def checkEqual(myList):
    first = myList[0]
    return all((x == first).all() for x in myList)

def padding(img):
    h, w = img.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, h, w, size

class QuadTreeLevelMap:
    def __init__(self):
        self.level_map = None
        self.max_level = 0

    def insert(self, img, x=0, y=0, level=0, level_map=None):
        h, w = img.shape
        if level_map is None:
            level_map = np.zeros(img.shape, dtype=np.float32)
            self.level_map = level_map

        if h < 2 or w < 2 or checkEqual(split4(img)):
            level_map[y:y+h, x:x+w] = level
            self.max_level = max(self.max_level, level)
            return level_map

        h2, w2 = h // 2, w // 2

        self.insert(img[:h2, :w2], x, y, level + 1, level_map)
        self.insert(img[:h2, w2:], x + w2, y, level + 1, level_map)
        self.insert(img[h2:, :w2], x, y + h2, level + 1, level_map)
        self.insert(img[h2:, w2:], x + w2, y + h2, level + 1, level_map)

        return level_map

processed_volume = np.zeros_like(imgArray, dtype=np.float32)

for i in range(depth):
    imgSlice = imgArray[i]
    maskSlice = maskArray[i]
    segment = np.where(maskSlice > 0, imgSlice, 0)

    valid = segment[segment > np.min(segment)]
    if valid.size == 0:
        continue

    paddedSegment, orig_h, orig_w, segSize = padding(segment)
    qt = QuadTreeLevelMap()
    level_map = qt.insert(paddedSegment)

    if qt.max_level > 0:
        level_map /= qt.max_level

    mapped = (level_map * 255).astype(np.uint8)
    processed_volume[i] = level_map[:orig_h, :orig_w]

outImg = sitk.GetImageFromArray(processed_volume)
outImg.CopyInformation(img)
sitk.WriteImage(outImg, outputPath)
print(f"Saved to: {outputPath}")
