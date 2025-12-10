import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

image_path = "../Images/P-T4LJ6_SPGR_FRC+bag.nii.gz"
img = sitk.ReadImage(image_path)

imgArray = sitk.GetArrayFromImage(img)

slicedImg = imgArray[292, :, :]

stretch = np.repeat(slicedImg, 2, axis=0)
plt.imshow(stretch, cmap = "gray", origin = "lower")
plt.axis("off")
plt.show()