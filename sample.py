import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches

imgPath = "../Vent_Int/MultiInflation_150083_mr_sVent_medfilt_3.nii.gz"
maskPath = "../seg_complete/150083_RV_mask.nii.gz"

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

def concatenate4(nw, ne, sw, se):
    top = np.hstack((nw, ne))
    bottom = np.hstack((sw, se))
    return np.vstack((top, bottom))

def checkEqual(myList):
    first = myList[0]
    return all((x == first).all() for x in myList)

class QuadTree:
    def insert(self, img, level=0):
        self.level = level
        self.mean = calculate_mean(img).astype(int)
        self.resolution = img.shape
        self.final = True

        if not checkEqual(split4(img)):
            if img.shape[0] < 2 or img.shape[1] < 2:
                return self

            split_img = split4(img)
            self.final = False
            self.north_west = QuadTree().insert(split_img[0], level + 1)
            self.north_east = QuadTree().insert(split_img[1], level + 1)
            self.south_west = QuadTree().insert(split_img[2], level + 1)
            self.south_east = QuadTree().insert(split_img[3], level + 1)
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
quadtree = QuadTree().insert(padSegment)

for lvl in [10]:
    try:
        recon = quadtree.get_image(lvl)
        plt.figure(figsize=(6, 6))
        plt.imshow(recon, cmap='gray', origin='lower')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Level {lvl}: Error - {e}")
