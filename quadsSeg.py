import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

"""
Quadtree Image, with segmentation. Without layer or tree masks.

"""
# MRI Img
# Raw data
img_path = "..Vent_Int/MultiInflation_150083_mr_sVent_medfilt_3.nii.gz"
# Segmentation : img_path = "Z:/polaris2/students/Will_Students_2025/acb20si/Quadtree/Dataset/P-T4LJ6/mrA/Masks/PT4LJ6_SPGR_FRC+bag_seg.nii.gz"
maskPath = "../seg_complete/150083_RV_mask.nii.gz"
output_path = "../mrA/PT4LJ6_SPGR_FRC+bag_seg.nii.gz"

img = sitk.ReadImage(img_path)
imgArray = sitk.GetArrayFromImage(img)

# Mask
mask = sitk.ReadImage(maskPath)
maskArray = sitk.GetArrayFromImage(mask)

percentage = 10

def padding(img):
    h, w = img.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, h, w

"""
padSegment = padding(segment)
segSize = padSegment.shape[0]
#Global_standard deviation
globalStddev = np.std(padSegment)

#percentage
valid_pixels = padSegment[padSegment > np.min(padSegment)]
fullRange = np.max(valid_pixels) - np.min(valid_pixels)
"""
def calculate_mean(img):
    mean = np.array([[np.mean(img)]], dtype = np.float32)
    return mean
def split(img):
    h, w = img.shape
    h2, w2 = h//2, w//2
    split = [img[:h2, :w2], img[:h2, w2:], img[h2:, :w2], img[h2:, w2:]]
    return split
def combine(nw, ne, sw, se):
    combine = np.vstack([np.hstack([nw,ne]), np.hstack([sw,se])])
    return combine

def quadtree_decompose(img, threshold):
    if img.shape[0]<2 or img.shape[1]<2:
        return np.tile(calculate_mean(img), img.shape)
    
    range = np.max(img) -np.min(img)
    if range <= threshold:
        return np.tile(calculate_mean(img), img.shape)
    
    sp = split(img)
    nw = quadtree_decompose(sp[0], threshold)
    ne = quadtree_decompose(sp[1], threshold)
    sw = quadtree_decompose(sp[2], threshold)
    se = quadtree_decompose(sp[3], threshold)
    return combine(nw,ne,sw,se)

def process_volume(imgArray, maskArray):
    depth, height, width = imgArray.shape
    process_volume = np.zeros_like(imgArray, dtype = np.float32)

    for i in range(depth):
        imgSlice = imgArray[i]
        maskSlice = maskArray[i]
        segment = np.where(maskSlice > 0, imgSlice, 0)

        valid = segment[segment > np.min(segment)]
        if valid.size == 0:
            continue

        global_range = np.max(valid) - np.min(valid)
        threshold = global_range * (percentage/100)

        padded, o_height, o_width = padding(segment)
        recon = quadtree_decompose(padded, threshold)
        process_volume[i] = recon[:o_height, :o_width]
    return process_volume

#globalStddev = fullRange*(percentage/100)
def save_file(volume, img, path):
    result = sitk.GetImageFromArray(volume.astype(np.float32))
    result.CopyInformation(img)
    path = output_path
    
    sitk.WriteImage(result, path)
    print(f"saved to {path}")

volume = process_volume(imgArray, maskArray)
save_file(volume, img, output_path)

plt.figure(figsize=(8,8))
plt.imshow(volume[32], cmap = 'gray')
plt.title("Slice 32")
plt.axis('off')
plt.show()
"""

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(segment, cmap='gray', origin = "lower")
quadtree_decompose(paddedSegment[:,31,:], quadtreeResult, 0, 0, segSize, orig_w, orig_h, globalStddev, valid)
ax.axis('off')
plt.tight_layout()
plt.show()"""