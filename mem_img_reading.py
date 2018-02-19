"""  This program read the satellite data patches created from the Vaihingen 
dataset, gets the pixels values, creates pixels labels and gets a main label for
each patch. """

from PIL import Image
import numpy as np
import os

# Parameters...
# Paths to data.
path_to_sat_img = "D:/Desktop/data/cropped/train/sat/"
path_to_gt_img = "D:/Desktop/data/cropped/train/gt/"

# Define image size, for verifications and reshaping.
img_size = 128

# Define empty lists for loops
sat_data = []
gt_data = []

# Define empty list to store a single main label for each patch.
patch_label_list = []

# Define empty list for quick verification of filename vs label associated.
filename_list = []

# Define counting vars for verification.
count_sat = 0
count_gt = 0

""" First loop on cropped satellite images"""

# Read all data files names for first loop on sat images.
input_filenames_sat = [f for f in os.listdir(path_to_sat_img) if f.endswith('.tif')]
# Loop, for each input image.
for input_filename in input_filenames_sat:
    
    # Open input image, convert to np array.
    img_sat = np.array(Image.open(path_to_sat_img+input_filename))
    
    # Check if the image has a valid shape, get RGB values and store into list.
    if((img_sat.shape[0]==img_size) and (img_sat.shape[1] ==img_size)):
        r = img_sat[:,:,0]
        g = img_sat[:,:,1]
        b = img_sat[:,:,2]
        sat_data.append([r,g,b])
    
        count_sat += 1

print(count_sat)        

# Convert list to np array.
feats = np.array(sat_data)
# Reshape the array to original image size and dimension.
feats = feats.reshape(feats.shape[0], img_size, img_size, 3)

""" Second loop on cropped ground truth images"""

# Read all data files names for first loop on ground truth images.
input_filenames_gt = [f for f in os.listdir(path_to_gt_img) if f.endswith('.tif')]
# Loop, for each input image.
for input_filename in input_filenames_gt:
    
    # Open input image, convert to greyscale first, then np array.
    img_gt = np.array(Image.open(path_to_gt_img+input_filename).convert("L")) 
    # Check if the image has a valid shape, get greyscale value and store into list.
    if((img_gt.shape[0]==img_size) and (img_gt.shape[1] ==img_size)):
        gt_data.append(img_gt)
        filename_list.append(input_filename)
        count_gt += 1
   
print(count_gt)

# Convert list to np array.        
pixel_lvl_labels = np.array(gt_data)

# Flatten ground truth pixel level labels, store most occurring value for each patch.
for data in xrange(len(pixel_lvl_labels)):
    patch_label_list.append(np.bincount(np.ravel(pixel_lvl_labels[data])).argmax())

# Print a sample patch to see if we did good.
sample_id = 0
sample_img = feats[sample_id]
sample_pix_labels = pixel_lvl_labels[sample_id]
sample_patch_label = patch_label_list[sample_id]
sample_filename = filename_list[sample_id]

print("Filename:", sample_filename)
print("Pixel values):", sample_img)
print("Pixel level labels:", sample_pix_labels)
print("Patch level label:", sample_patch_label)
# some stats
#print([sample_img.min(), sample_img.max(), sample_img.shape])



