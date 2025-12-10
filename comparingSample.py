import numpy as np
import SimpleITK as sitk

def compare_image_intensity_full(quad_img, oct_img, mask=None, bins=100, hist_range=None):
    """
    Compare two images (e.g., Quad vs Oct) using MAE, MSE, Pearson correlation, and histograms.
    
    Returns:
    - metrics: dict with MAE, MSE, Pearson
    - hist_data: dict with histograms and bins
    """
    # Convert to arrays
    if isinstance(quad_img, sitk.Image):
        quad_arr = sitk.GetArrayFromImage(quad_img)
    else:
        quad_arr = quad_img

    if isinstance(oct_img, sitk.Image):
        oct_arr = sitk.GetArrayFromImage(oct_img)
    else:
        oct_arr = oct_img

    if quad_arr.shape != oct_arr.shape:
        raise ValueError("Image shapes do not match.")

    # Apply mask
    if mask is not None:
        if isinstance(mask, sitk.Image):
            mask = sitk.GetArrayFromImage(mask)
        if mask.shape != quad_arr.shape:
            raise ValueError("Mask shape does not match image shape.")
        mask = mask > 0
        quad_vals = quad_arr[mask]
        oct_vals = oct_arr[mask]
    else:
        quad_vals = quad_arr.flatten()
        oct_vals = oct_arr.flatten()

    # MAE & MSE
    mae = np.mean(np.abs(quad_vals - oct_vals))
    mse = np.mean((quad_vals - oct_vals) ** 2)

    # Pearson
    quad_mean = np.mean(quad_vals)
    oct_mean = np.mean(oct_vals)
    numerator = np.sum((quad_vals - quad_mean) * (oct_vals - oct_mean))
    denominator = np.sqrt(np.sum((quad_vals - quad_mean) ** 2)) * np.sqrt(np.sum((oct_vals - oct_mean) ** 2))
    pearson = numerator / denominator if denominator != 0 else 0.0

    # Histogram
    if hist_range is None:
        min_val = min(quad_vals.min(), oct_vals.min())
        max_val = max(quad_vals.max(), oct_vals.max())
        hist_range = (min_val, max_val)

    hist_quad, bin_edges = np.histogram(quad_vals, bins=bins, range=hist_range, density=True)
    hist_oct, _ = np.histogram(oct_vals, bins=bins, range=hist_range, density=True)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "Pearson": pearson
    }

    hist_data = {
        "hist_quad": hist_quad,
        "hist_oct": hist_oct,
        "bin_edges": bin_edges
    }

    return metrics, hist_data

quad = sitk.ReadImage("quad_image.nii.gz")
oct = sitk.ReadImage("oct_image.nii.gz")
mask = sitk.ReadImage("mask_image.nii.gz")

metrics, hist_data = compare_image_intensity_full(quad, oct, mask=mask)

print("MAE:", metrics["MAE"])
print("MSE:", metrics["MSE"])
print("Pearson:", metrics["Pearson"])

# 히스토그램 시각화
import matplotlib.pyplot as plt
bins = hist_data["bin_edges"]
plt.plot(bins[:-1], hist_data["hist_quad"], label="Quadtree", alpha=0.7)
plt.plot(bins[:-1], hist_data["hist_oct"], label="Octree", alpha=0.7)
plt.xlabel("Intensity")
plt.ylabel("Density")
plt.title("Histogram Comparison")
plt.legend()
plt.tight_layout()
plt.show()
histogram = {
        "Quad_histogram": hist_quad,
        "Oct_histogram": hist_oct,
        "Bin_edges": bin_edges
        }