"""  This program reads the satellite data patches created from the Vaihingen 
dataset, gets the pixels values, creates pixels labels and gets a main label for
each patch. """

from __future__ import print_function
from PIL import Image
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

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
    # Check if the image has a valid shape, get greyscale values (= pixels label) and store into list.
    if((img_gt.shape[0]==img_size) and (img_gt.shape[1] ==img_size)):
        gt_data.append(img_gt)
        filename_list.append(input_filename)
        count_gt += 1
   
print(count_gt)

# Convert list to np array.        
pixel_lvl_labels = np.array(gt_data)

# Flatten ground truth pixel level labels, store most occurring value for each patch.
for data in range(len(pixel_lvl_labels)):
    patch_label_list.append(np.bincount(np.ravel(pixel_lvl_labels[data])).argmax())

    
# Replace label greyscale values by [0, 1, 2, 3, ...]...
# We need to do that in order to create one hot vectors later.

# Get unique label values from patch_label_list.
labels_list_values = set(patch_label_list)

# Define empty list where we can put these values, and make them iterable.
labels_list = []

# Loop to put the values in the list.
for label in labels_list_values:
    labels_list.append(label)

# Loop to replace label greyscale values by [0, 1, 2, 3, ...]
for label in range(len(labels_list)):
    for patch_label in range(len(patch_label_list)):
        if patch_label_list[patch_label] == labels_list[label]:
            patch_label_list[patch_label] = label
            
#print(patch_label_list)
    
# Print a sample patch to see if we did good.
sample_id = 0
sample_img = feats[sample_id]
sample_pix_labels = pixel_lvl_labels[sample_id]
sample_patch_label = patch_label_list[sample_id]
sample_filename = filename_list[sample_id]

print("Filename:", sample_filename)
print("Pixel values:", sample_img)
print("Pixel level labels:", sample_pix_labels)
print("Patch level label:", sample_patch_label)
print("Input shape:", sample_img.shape)

print(len(set(patch_label_list)))
#print(set(patch_label_list))
# some stats
#print([sample_img.min(), sample_img.max(), sample_img.shape])


""" CNN example adapted from:
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

It seems to work for now, but poor accuracy with the data we just created.
Data augmentation could be a solution. 
Also, the different layers of the models need to be adapted.
"""

batch_size = 64
num_classes = len(labels_list_values)
epochs = 50
train_size = 2500

# input image dimensions
img_rows, img_cols = 128, 128

# the data, split between train and test sets
x_train = np.array(feats[:train_size])
y_train = np.array(patch_label_list[:train_size])
x_test = np.array(feats[train_size:])
y_test = np.array(patch_label_list[train_size:])

input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#print('x_train shape:', x_train)
#print(y_train.shape[0], 'train samples')
#print(y_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])













