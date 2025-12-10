def process_volume(imgArray, maskArray):
    depth, height, width = imgArray.shape
    result_volume = np.zeros_like(imgArray, dtype=np.float32)

    for i in range(depth):
        imgSlice = imgArray[i]
        maskSlice = maskArray[i]

        if np.count_nonzero(maskSlice) == 0:
            # 마스크가 없는 slice는 0으로 남김
            continue

        # 마스크가 있는 slice만 분해
        segment = np.where(maskSlice > 0, imgSlice, 0)
        valid = segment[segment > np.min(segment)]

        if valid.size == 0:
            continue

        global_range = np.max(valid) - np.min(valid)
        threshold = global_range * (percentage / 100)

        padded, o_height, o_width = padding(segment)
        recon = quadtree_decompose(padded, threshold)

        result_volume[i] = recon[:o_height, :o_width]

    return result_volume