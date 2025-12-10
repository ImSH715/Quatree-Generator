import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

imgPath = "Z:/polaris2/students/Will_Students_2025/acb20si/Heterogenity/Ventilation/150083/mrA/Reg_TLC_2_RV/Vent_Int/MultiInflation_150083_mr_sVent_medfilt_3.nii.gz"
maskPath = "Z:/polaris2/students/Will_Students_2025/acb20si/Heterogenity/Data/150083/mrA/seg_complete/150083_RV_mask.nii.gz"

img = sitk.ReadImage(imgPath)
imgArray = sitk.GetArrayFromImage(img)
slicedImg = imgArray[32, :, :]

mask = sitk.ReadImage(maskPath)
maskArray = sitk.GetArrayFromImage(mask)
slicedMask = maskArray[32, :, :]
segment = np.where(slicedMask > 0, slicedImg, 0)

def padding(img):
    h, w = img.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[:h, :w] = img
    return padded

def calculate_mean(img):
    return np.array([[np.mean(img)]], dtype=np.float32)

def split4(img):
    h, w = img.shape
    h2, w2 = h // 2, w // 2
    return [
        img[:h2, :w2],
        img[:h2, w2:],
        img[h2:, :w2],
        img[h2:, w2:],
    ]

def concatenate4(nw, ne, sw, se):
    top = np.hstack((nw, ne))
    bottom = np.hstack((sw, se))
    return np.vstack((top, bottom))

class QuadTree:
    def __init__(self, max_level=10):
        self.max_level = max_level

    def insert(self, img, level=0):
        self.level = level
        self.mean = calculate_mean(img)
        self.resolution = img.shape
        self.final = True

        if level >= self.max_level or img.shape[0] < 2 or img.shape[1] < 2:
            return self

        self.final = False
        split_img = split4(img)
        self.north_west = QuadTree(self.max_level).insert(split_img[0], level + 1)
        self.north_east = QuadTree(self.max_level).insert(split_img[1], level + 1)
        self.south_west = QuadTree(self.max_level).insert(split_img[2], level + 1)
        self.south_east = QuadTree(self.max_level).insert(split_img[3], level + 1)
        return self

    def get_image(self, level):
        if self.final or self.level == level:
            return np.tile(self.mean, self.resolution)
        return concatenate4(
            self.north_west.get_image(level),
            self.north_east.get_image(level),
            self.south_west.get_image(level),
            self.south_east.get_image(level)
        )

padSegment = padding(segment)
orig_h, orig_w = segment.shape
max_level = 10

quadtree = QuadTree(max_level=max_level).insert(padSegment)
recon = quadtree.get_image(max_level)
recon_crop = recon[:orig_h, :orig_w]

plt.figure(figsize=(6, 6))
plt.title(f"Quadtree Forced Decomposition (Level {max_level})")
plt.imshow(recon_crop, cmap='gray', origin='lower')
plt.axis('off')
plt.tight_layout()
plt.show()
