import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt

imgPath = "Z:/polaris2/students/Will_Students_2025/acb20si/Quadtree/Dataset/P-T4LJ6/mrA/Images/P-T4LJ6_SPGR_FRC+bag.nii.gz"

img = sitk.ReadImage(imgPath)

imgArray = sitk.GetArrayFromImage(img)

slicedImg = imgArray[217, :, :]

stretchedImg = np.repeat(slicedImg, 2, axis = 0 )
orig_h, orig_w = stretchedImg.shape
imgSize = stretchedImg.shape[0]

# Mask
# Segmentation : imgPath = "Z:/polaris2/students/Will_Students_2025/acb20si/Quadtree/Dataset/P-T4LJ6/mrA/Masks/PT4LJ6_SPGR_FRC+bag_seg.nii.gz"
maskPath = "Z:/polaris2/students/Will_Students_2025/acb20si/Quadtree/Dataset/P-T4LJ6/mrA/Masks/PT4LJ6_SPGR_FRC+bag_seg.nii.gz"

mask = sitk.ReadImage(maskPath)

maskArray = sitk.GetArrayFromImage(mask)

slicedMask = maskArray[217, :, :]
stretchedMask = np.repeat(slicedMask, 2, axis = 0 )

masked = np.where(stretchedMask > 0, stretchedImg, 0)

plt.show()