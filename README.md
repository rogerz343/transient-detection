# transient-detection
A collection of machine learning techniques used to classify images of the
night sky as either potential astronomical objects or as statistical defects or
noise.

## required files
Some of the code references input image files and a .csv file that
contains data about the images (labels, ids, features, etc.). All of these
files can be found at http://portal.nersc.gov/project/dessn/autoscan/.

## cnn
Contains the code for a convolutional neural network used to classify images.
The .py files should in theory work on windows, linux, and macOS. However, if
they aren't working on windows, try running on linux. More details about each file
can be found documented at the start of the files.
- `merge_autoscan_data.py`: used for the specific task of combining the diff, srch,
and temp files in the data (in the link above) into one image with 3 channels.
- `read_data.py`: used to read image files and convert them to .npy vectors
- `run_cnn.py`: used to train a CNN on .npy image vectors and run tests.
The link above (in the "required files" section) contains image files and a
corresponding .csv file with the image id's and labels. In order to use
`read_data.py` on this dataset, first remove any headers (not including column
names) so that it can be read directly as a table. When using this data, first
run `merge_autoscan_data.py`, then `read_data.py`, then `run_cnn.py`.
