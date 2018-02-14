# cnn-for-satellite-data
A simple conv-deconv neural network for semantic segmentation on satellite data

The objective of this project is to build a simple convolution-deconvolution neural network model with Keras (TensorFlow engine backend), that can process satellite data and generate a semantic segmentation.

The data that will be used to elaborate and calibrate this model is the Vaihingen dataset. Description and download available at http://www2.isprs.org/commissions/comm3/wg4/detection-and-reconstruction.html

Prerequisites:
- Python 3
- TensorFlow, https://www.tensorflow.org/
- Keras, https://keras.io/

Steps:
- Data preprocessing (crop series of smaller images from the patches, determine the preponderant label for each image, for training and validation data).
- Simple convolutional neural network (Make the model guess a single label per image).
- Simple convolution-deconvolution neural network (Make the model guess a label for each pixel of the image).
- Evaluate the performances.
