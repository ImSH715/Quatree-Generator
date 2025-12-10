def quad_tree_mask(imgPath, maskPath, outputPath, comparator_type="percentage"):
    import numpy as np
    import SimpleITK as sitk

    def padding(img):
        h, w = img.shape
        size = 2 ** int(np.ceil(np.log2(max(h, w))))
        padded = np.zeros((size, size), dtype=img.dtype)
        padded[:h, :w] = img
        return padded, h, w, size

    def split_by_percentage(img, ratio=0.2):
        values = img[img > 0]
        if values.size == 0:
            return False
        mean = np.mean(values)
        if mean < 1e-5:
            return False
        return (np.max(values) - np.min(values)) > (ratio * mean)

    def split_by_std(img, ratio=0.15):
        values = img[img > 0]
        if values.size == 0:
            return False
        mean = np.mean(values)
        if mean < 1e-5:
            return False
        return np.std(values) > (ratio * mean)

    def should_split(img):
        if comparator_type == "percentage":
            return split_by_percentage(img)
        elif comparator_type == "std":
            return split_by_std(img)
        else:
            raise ValueError("Invalid comparator_type. Use 'percentage' or 'std'.")

    def insert(img, x=0, y=0, level=0, level_map=None, max_level_tracker=[0]):
        h, w = img.shape
        if level_map is None:
            level_map = np.zeros(img.shape, dtype=np.float32)

        if h < 2 or w < 2 or not should_split(img):
            level_map[y:y+h, x:x+w] = level
            max_level_tracker[0] = max(max_level_tracker[0], level)
            return level_map

        h2, w2 = h // 2, w // 2

        insert(img[:h2, :w2], x, y, level + 1, level_map, max_level_tracker)
        insert(img[:h2, w2:], x + w2, y, level + 1, level_map, max_level_tracker)
        insert(img[h2:, :w2], x, y + h2, level + 1, level_map, max_level_tracker)
        insert(img[h2:, w2:], x + w2, y + h2, level + 1, level_map, max_level_tracker)

        return level_map

    # ------------------ 쿼드트리 수행 ------------------
    img = sitk.ReadImage(imgPath)
    imgArray = sitk.GetArrayFromImage(img)

    mask = sitk.ReadImage(maskPath)
    maskArray = sitk.GetArrayFromImage(mask)

    depth = imgArray.shape[0]
    processed_volume = np.zeros_like(imgArray, dtype=np.float32)

    for i in range(depth):
        imgSlice = imgArray[i]
        maskSlice = maskArray[i]
        segment = np.where(maskSlice > 0, imgSlice, 0)

        valid = segment[segment > np.min(segment)]
        if valid.size == 0:
            continue

        paddedSegment, orig_h, orig_w, _ = padding(segment)
        max_level_tracker = [0]
        level_map = insert(paddedSegment, max_level_tracker=max_level_tracker)

        if max_level_tracker[0] > 0:
            level_map /= max_level_tracker[0]

        processed_volume[i] = level_map[:orig_h, :orig_w]

    outImg = sitk.GetImageFromArray(processed_volume)
    outImg.CopyInformation(img)
    sitk.WriteImage(outImg, outputPath)
    print(f"Saved to: {outputPath}")

quad_tree_mask(
    imgPath="Z:/path/to/image.nii.gz",
    maskPath="Z:/path/to/mask.nii.gz",
    outputPath="Z:/save/result.nii.gz",
    comparator_type="std"  # or "percentage"
)
