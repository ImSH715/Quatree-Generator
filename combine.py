   #Contrasting Quad and Oct, trees
   """
   """
   #quad_img_path = 
   #quad_path = quad_output_path.replace(".nii.gz", "_QuadImage.nii.gz")

   quad_img_array = sitk.GetArrayFromImage(sitk.ReadImage(quad_img_path))
   print(quad_img_array.shape)
   quad_mask_array = pad_to_256(sitk.GetArrayFromImage(sitk.ReadImage(mask_1_path)))

   oct_img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_1_oct_path))
   oct_mask_array = pad_to_256(sitk.GetArrayFromImage(sitk.ReadImage(mask_1_path)))
   #Need implementation
   print("Contrasting Octree and Quadtree intensity ====")
   #Binary mask
   mask_array = pad_to_256(sitk.GetArrayFromImage(sitk.ReadImage(mask_1_path))>0)
   level_stats = []
   #Check levels
   levels_check = np.unique(oct_mask_array[mask_array])
   levels_check = levels_check[levels_check>0]

   #Percentage of voxel numbers
   voxel_percentage = 21
   threshold = (np.sum(mask_array)*(voxel_percentage/100))

"""

   for level in levels_check:
       quad_index = (quad_mask_array == level) & mask_array
       oct_index = (oct_mask_array == level) & mask_array
       common_index = quad_index & oct_index

       if np.sum(common_index) < 10:
           continue
       quad_values = quad_img_array[common_index].flatten()
       oct_values = oct_img_array[common_index].flatten()
       vmin = min(quad_values.min(), oct_values.min())
       vmax = max(quad_values.min(), oct_values.max())

       # Ranging vmin==vmax histogram
       if vmin == vmax:
           continue
       hist_range = (vmin, vmax)

       #Escape from NaN or Inf
       quad_values = quad_values[np.isfinite(quad_values)]
       oct_values = oct_values[np.isfinite(oct_values)]


       bins = 100

       hist_range = (min(quad_values.min(), oct_values.min()), max(quad_values.max(), oct_values.max()))
       hist1, _ = np.histogram(quad_values, bins=bins, range=hist_range, density = True)
       hist2, _ = np.histogram(oct_values, bins = bins, range = hist_range, density = True)
       hist_intersection = np.sum(np.minimum(hist1, hist2))

       level_stats.append((int(level), hist_intersection, len(quad_values)))
       """
"""
   print("Histogram")
   for level, hist_i, count in level_stats:
       print(f"level: {level} Hist: {hist_i} count:{count}")
   """
 # Peasron Correlation
def pearson_corr(a,b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    num = np.sum((a-a_mean)*(b-b_mean))
    den = np.sqrt(np.sum((a-a_mean)**2)) * np.sqrt(np.sum((b-b_mean)**2))
    return num/den if den !=0 else 0.0
common_levels = np.unique(quad_mask_array[mask_array])
common_levels = common_levels[common_levels > 0]

for level in common_levels:
    quad_index = (quad_mask_array == level)& mask_array
    oct_index = (oct_mask_array == level)& mask_array
    common_index = quad_index & oct_index

    if np.sum(common_index) < 10:
        continue
    quad_values = quad_img_array[common_index].flatten()
    oct_values = oct_img_array[common_index].flatten()
    r = pearson_corr(quad_values, oct_values)
    
    level_stats.append((int(level), r , len(quad_values)))
print("Level: Pearson r: Voxel Count:")

for level, r, count in level_stats:
    print(f"{level:5d} | {r:9.4f} | {count:11d}")

if len(level_stats) > 0:
    levels = [x[0]for x in level_stats]
    r_vals = [x[1]for x in level_stats]

    plt.figure(figsize=(6, 4))
    plt.plot(levels, r_vals, marker='o', linestyle= '-')
    plt.xlabel("decomposition level")
    plt.ylabel("pearson Correlation")
    plt.title(f"Level-wise Intensity Correlation: {tail}")
    plt.grid(True)