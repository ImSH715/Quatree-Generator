import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

imgPath = "../Reg_TLC_2_RV/Vent_Int/MultiInflation_150083_mr_sVent_medfilt_3.nii.gz"
maskPath = "../mrA/seg_complete/150083_RV_mask.nii.gz"
savePath = "../mrA/visual_quadtree_result_full.nii.gz"

def padding(img):
    h, w = img.shape
    size = 2 ** int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, h, w

def calculate_mean(img):
    return np.array([[np.mean(img)]], dtype=np.float32)

def split4(img):
    h, w = img.shape
    h2, w2 = h // 2, w // 2
    return [img[:h2, :w2], img[:h2, w2:], img[h2:, :w2], img[h2:, w2:]]

def concatenate4(nw, ne, sw, se):
    return np.vstack([np.hstack([nw, ne]), np.hstack([sw, se])])

def quadtree_decompose(img, threshold):
    """
    Recursive function that decomposes the image based on threshold.
    Returns a block-wise reconstructed image using region means.
    """
    if img.shape[0] < 2 or img.shape[1] < 2:
        return np.tile(calculate_mean(img), img.shape)

    region_range = np.max(img) - np.min(img)
    if region_range <= threshold:
        return np.tile(calculate_mean(img), img.shape)

    sp = split4(img)
    nw = quadtree_decompose(sp[0], threshold)
    ne = quadtree_decompose(sp[1], threshold)
    sw = quadtree_decompose(sp[2], threshold)
    se = quadtree_decompose(sp[3], threshold)
    return concatenate4(nw, ne, sw, se)

def process_volume(imgArray, maskArray):
    depth, height, width = imgArray.shape
    processed_volume = np.zeros_like(imgArray, dtype=np.float32)

    for i in range(depth):
        imgSlice = imgArray[i]
        maskSlice = maskArray[i]
        segment = np.where(maskSlice > 0, imgSlice, 0)

        valid = segment[segment > np.min(segment)]
        if valid.size == 0:
            continue

        global_range = np.max(valid) - np.min(valid)
        threshold = global_range * 0.10  # 자동 설정

        padded, orig_h, orig_w = padding(segment)
        recon = quadtree_decompose(padded, threshold)
        processed_volume[i] = recon[:orig_h, :orig_w]

    return processed_volume

def save_result(volume, reference_img, path):
    result_img = sitk.GetImageFromArray(volume.astype(np.float32))
    result_img.CopyInformation(reference_img)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(result_img, path)
    print(f"Saved to: {path}")

def main():
    img = sitk.ReadImage(imgPath)
    mask = sitk.ReadImage(maskPath)
    imgArray = sitk.GetArrayFromImage(img)
    maskArray = sitk.GetArrayFromImage(mask)

    processed_volume = process_volume(imgArray, maskArray)
    save_result(processed_volume, img, savePath)

    plt.figure(figsize=(6, 6))
    plt.imshow(processed_volume[32], cmap='gray', origin='lower')
    plt.title("Quadtree Auto (Slice 32)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
