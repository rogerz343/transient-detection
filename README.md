# transient-detection
A collection of machine learning techniques used to classify images of the night sky as either potential astronomical objects or as statistical defects or noise.

## required files
Some of the code references input image files (.gif) and a .csv file that
contains data about the images (labels, ids, features, etc.). All of these
files can be found at http://portal.nersc.gov/project/dessn/autoscan/.

## cnn
Contains the code for a convolutional neural network used to classify images.
The .py files should in theory work on windows, linux, and macOS. However, if they aren't working on windows, try using linux.
- read_data.py: used to read .gif files and convert them to .npy vectors
- run_cnn.py: used to train a CNN on .npy image vectors and run tests
