"""
This code is used on the Vaihingen dataset to divide each satellite and ground
truth images into smaller patches. These patches will be used as a training 
dataset for the CNN.

The Vaihingen dataset is available at 
http://www2.isprs.org/commissions/comm3/wg4/detection-and-reconstruction.html
"""

from PIL import Image
from pylab import array, imshow, show
import sys, os

# Parameters...
# Paths to data.
path_to_sat_img = "D:/Desktop/data/original/train/sat/"
path_to_gt_img = "D:/Desktop/data/original/train/gt/"
# Define patch size for image cropping.
patch_size = 128
# Read all data files names for loop.
input_filenames = [f for f in os.listdir(path_to_gt_img) if f.endswith('.tif')]

def main():

    # Set an index that will increase after each input image has been processed.
    index = 0
    
    print(input_filenames)
    
    # Loop, for each input image, divide it into patches, and save them on disk.
    for input_filename in input_filenames:

        # Open input image
        img = Image.open(path_to_gt_img+input_filename)

        # Get input image dimensions
        sizeX = img.size[0]
        sizeY = img.size[1]

        # Compute total number of rows and columns that the image will be divided into.
        num_of_columns = sizeX/patch_size
        num_of_rows = sizeY/patch_size
        
        # Loop, goes through the image and crop the patches.
        for row in xrange(num_of_rows):
            for column in xrange(num_of_columns):
                coords = (
                    column * patch_size,
                    row * patch_size,
                    (column + 1) * patch_size,
                    (row + 1) * patch_size
                )
                
                patch = img.crop(coords)
                
                #print(patch.format, patch.size, patch.mode)
                #imshow(array(patch))
                #show()
                
                # Save patch to desired path, with unique ID.
                patch.save("D:/Desktop/data/cropped/train/gt/patch"
                + str(index)
                + str(row)
                + str(column)
                + ".tif" )
                
        # Increases for each input image.   
        index += 1
            
if __name__ == "__main__":
    main()