import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches

imgPath = "../Images/P-T4LJ6_SPGR_FRC+bag.nii.gz"
maskPath = "../Masks/PT4LJ6_SPGR_FRC+bag_seg.nii.gz"
outputPath = "../mrA/PT4LJ6_SPGR_FRC+bag_seg_quadtree.nii.gz"

img = sitk.ReadImage(imgPath)
mask = sitk.ReadImage(maskPath)

imgArray = sitk.GetArrayFromImage(img) 
maskArray = sitk.GetArrayFromImage(mask)

depth, height, width = imgArray.shape

def padding(img):
    h, w = img.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, h, w, size

def quadtree_decompose_to_array(img, out_array, x, y, size, orig_w, orig_h, globalStddev, valid_pixels):
    if x >= orig_w or y >= orig_h or size <= 1:
        return
    region = img[y:y+size, x:x+size]
    regionMax = np.max(region)
    regionMin = np.min(region)
    regionRange = regionMax - regionMin

    if regionRange <= globalStddev:
        meanIntensity = np.mean(region)
        out_array[y:y+size, x:x+size] = meanIntensity
    else:
        half = size // 2
        quadtree_decompose_to_array(img, out_array, x, y, half, orig_w, orig_h, globalStddev, valid_pixels)
        quadtree_decompose_to_array(img, out_array, x + half, y, half, orig_w, orig_h, globalStddev, valid_pixels)
        quadtree_decompose_to_array(img, out_array, x, y + half, half, orig_w, orig_h, globalStddev, valid_pixels)
        quadtree_decompose_to_array(img, out_array, x + half, y + half, half, orig_w, orig_h, globalStddev, valid_pixels)

processed_volume = np.zeros_like(imgArray, dtype=np.float32)

for i in range(depth):
    imgSlice = imgArray[i]
    maskSlice = maskArray[i]
    
    segment = np.where(maskSlice > 0, imgSlice, 0)

    valid_pixels = segment[segment > np.min(segment)]
    if valid_pixels.size == 0:
        continue

    fullRange = np.max(valid_pixels) - np.min(valid_pixels)
    globalStddev = fullRange * (10 / 100) 

    paddedSegment, orig_h, orig_w, segSize = padding(segment)
    quadtreeResult = np.zeros_like(paddedSegment)

    quadtree_decompose_to_array(paddedSegment, quadtreeResult, 0, 0, segSize, orig_w, orig_h, globalStddev, valid_pixels)

    processed_volume[i] = quadtreeResult[:orig_h, :orig_w]

# 저장
outImg = sitk.GetImageFromArray(processed_volume)
outImg.CopyInformation(img)
sitk.WriteImage(outImg, outputPath)
print(f"전체 Quadtree 결과가 저장되었습니다: {outputPath}")
