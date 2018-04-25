# transient-detection
A collection of machine learning techniques used to classify images of the
night sky as either potential astronomical objects or as statistical defects or
noise.

### Note
Any files that are not mentioned here may be uncleaned outputs from
running the code and they are not needed to run the code. This includes any
directories that begin with "temp\_".

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

## nn
Contains the code for the nueral network that runs on the feature data.
Each of the `.py` files can be run independently, given that the data is present in the parent directory. They all run a NN on the features using the specifications indicated in the names of the `.py` files. More specifically we had a template NN (which is caled complete). All other files would encompass one change to this Complete algorithm. For example they may use a different architecture (ie in 'n_2n_n2.py' indicates n layers then 2n layers then n/2 layers) or would be use dropout to avoid a FCC 'sparse', or would alter the number of epochs run or would alter tthe use of PCA or would use Ridge regularization with different values of beta. Each of these scripts when run will call on the helpers fuinction to pmeasure all of their performance measures (including ROC, Precision Recall, TPR, FPR, F1, GM, and every other performance form the performance measures lecture notes).
- `extractObjLbl.pyc` parses through subdirectories of a given root directory and reads .pkl files.

## random forest
The helpers function contains code which measures all of the performance measures and saves it to a .pkl file. All of the other functions contains the original random forest algorithm. All of the other scripts are modifications of this complete script. The name of the script file indicates the modification (ex: mF3 indicates that 3 parameters are used in the script). When each of the scripts is run, it generates (using the helper function) a .pkl file recording all of the performacne measures and the precision-recall and ROC garphs in the output folder (./out). `comparenT.py` parses through output .pkl files and computes relevant statistics measures.