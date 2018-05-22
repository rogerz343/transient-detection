# transient-detection
A collection of machine learning techniques used to classify images of the
night sky as either potential astronomical objects or as statistical defects or
noise.

### Note
Any files that are not mentioned here may be uncleaned outputs from
running the code. This includes any directories that begin with "temp\_" or
end with "\_out".

## required files
Some of the code references input image files and a .csv file that
contains data about the images (labels, ids, features, etc.). All of these
files can be found at http://portal.nersc.gov/project/dessn/autoscan/ (they
are not included here because they are very large files).

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

## feature data
Contains some of the data needed for the code in `nn` and `random forest` to run.

## nn
### Note: this portion of code is poorly organized/documented. 
Contains the code for the neural network that runs on the feature data.
Other than the `comparePkls.py`, `extractObjLbl.pyc`, and `helpers.py` files, each of the `.py` files is a script that runs a neural network on the features data. Each of them can be run independently, given that the `feature data` folder is present in the parent directory. They all run a NN on the features using the specifications indicated in the names of the `.py` files. More specifically we had a template NN `Complete.py`. All other files test one modification (for example, a different architecture) of the `Complete.py` script. For example:
- `n_2n_n2.py` indicates n layers then 2n layers then n/2 layers
- `sparse` indicates dropout to be used
- A number as a suffix indicates the number of epochs

When run, each of these scripts will call on the helpers fuinction to compute all of their performance measures (including ROC, Precision Recall, TPR, FPR, F1, GM, and every other performance form the performance measures lecture notes).

## random forest
### Note: this portion of code is poorly organized/documented.
Contains the code for the random forest classification algorithm. Other than `helpers.py`, `comparenT.py`, and `comparePkls.py`, each of the `.py` files is a script that runs a random forest (with specific parameters). All of the other `.py` files contain a copy of the random forest algorithm. The name of the script file indicates the modification (for example, mF3 indicates that 3 feature parameters are used in the script. When each of the scripts is run, it generates a .pkl file recording all of the performance measures and the precision-recall and ROC graphs in an output folder. `comparenT.py` parses through output .pkl files and computes relevant statistics measures.
- `helpers.py`: measures all of the performance measures and saves the output to a `.pkl` file.
