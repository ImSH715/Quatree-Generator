import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
# MRI Img
# Raw data
imgPath = "Z:/polaris2/students/Will_Students_2025/acb20si/Quadtree/Dataset/P-T4LJ6/mrA/Images/P-T4LJ6_SPGR_FRC+bag.nii.gz"
# Segmentation : imgPath = "Z:/polaris2/students/Will_Students_2025/acb20si/Quadtree/Dataset/P-T4LJ6/mrA/Masks/PT4LJ6_SPGR_FRC+bag_seg.nii.gz"
img = sitk.ReadImage(imgPath)
imgArray = sitk.GetArrayFromImage(img)
slicedImg = imgArray[:, :, :]
stretchedImg = np.repeat(slicedImg, 2, axis=0)
plt.imshow(stretchedImg,cmap="gray")
plt.show()