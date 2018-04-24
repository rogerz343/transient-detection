"""
run_cnn_v2.py: Trains a CNN using the input images. The input data should be
the output of read_data_v2.py and these files should be found in
'./read_data_out/'. The output files will be saved to '/global/homes/l/liuto/520project/transient-detection/cnn/jobs/run_cnn_out/'

arguments:
- the suffix of the names of the .npy outputs files from read_data, used as
  the model name
- a name used to name output files

Additional notes
- parameters of the CNN to optimize can be found by searching for
  "parameters to change"
"""

from astropy.io import fits
from collections import Counter

import numpy as np
import random
import argparse
import os
import errno

# import keras
import keras
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

# make keras not eat threads for breakfast
# cori = 16, cori job = 32, personal computer = 8
MAX_THREADS = 16
import tensorflow as tf
config = tf.ConfigProto(intra_op_parallelism_threads=MAX_THREADS,
                        inter_op_parallelism_threads=MAX_THREADS,
                        allow_soft_placement=True)
                        # device_count={'CPU': args.jobs})
session = tf.Session(config=config)
keras.backend.set_session(session)

def normalize_img(image):
    """ Returns the image array such that all pixel values are between 0 and 1

    :type image: numpy.ndarray
    :rtype: numpy.ndarray
    """
    max_val = np.amax(image)
    if max_val == 0:
        return image
    return image / max_val

# parameters to change: the model layout
def setup_model(width, height, channels):
    """ Sets up and returns the keras model

    :type width: int The width of the image (in px)
    :type height: int The height of the image (in px)
    :type channels: int The number of color channels in the image
    :rtype: keras.models.Sequential The keras Sequential model
    """
    model = Sequential()

    # first layer: convolution
    model.add(Convolution2D(128, 6, activation='relu',
                            padding='same',
                            input_shape=(width, height, channels),
                            data_format='channels_last'))

    # hidden layer
    model.add(Convolution2D(64, 3, activation='relu', padding='same'))
    model.add(Dropout(0.2))

    # hidden layer
    model.add(Convolution2D(64, 3, activation='relu', padding='same'))
    model.add(Dropout(0.2))

    # hidden layer
    model.add(Convolution2D(32, 2, activation='relu', padding='same'))
    model.add(Dropout(0.2))

    # flatten to fully connected NN
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    # last layer: dense (prediction) with 1 output
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform'))

    print('Compiling...')
    
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# parameters to change: the weights for labels 0 and 1
def balanced_class_weights(labels):
    """ Assigns weights to each label class based on their representation in
    the training data

    :type labels: numpy.ndarray The labels (classes) of the training data
    :rtype: dict A dictionary from label/class (int) to a measure of its
        representation (float) in the training data. Underrepresented labels are
        assigned higher weights so that the training "pays more attention" to
        them
    """
    counter = Counter(labels)
    max_freq = max(counter.values())  # num of times the highest-freq class appears
    return {label: float(max_freq) / count for label, count in counter.items()}

def train_model(train_imgs, train_labels, val_imgs, val_labels, output_dir, output_name):
    """ Trains a CNN. This function assumes inputs are well-formed.

    :type train_imgs: numpy.ndarray The array of train images. Each image is a
        3-dimensional array of numbers
    :type train_labels: numpy.ndarray The array of corresponding train image
        labels.
    :type val_imgs: numpy.ndarray The array of validation images. These will
        *not* be used during training, but will be used to calculate metrics
        during training to measure progress.
    :type val_labels: numpy.ndarray The array of corresponding validation image
        labels.
    :type output_name: str A string used to output_name the output files.
    """
    print('========================')
    print('=====Training model=====')
    print('========================')

    # normalize images
    for i in range(train_imgs.shape[0]):
        train_imgs[i] = normalize_img(train_imgs[i])
    
    width, height, channels = train_imgs[0].shape

    # set up model
    model = setup_model(width, height, channels)
    model.summary()

    # parameters to change: training parameters
    batch_size = 48
    epochs = 20
    verbose = 1
    validation_data = (val_imgs, val_labels)
    shuffle = True
    class_weight = balanced_class_weights(train_labels)

    # parameters to change: EarlyStopping
    # stop early (don't go through all epochs) if model converges
    # NOTE: DO NOT USE THIS UNLESS YOU KNOW WHAT YOU ARE DOING
    # early_stopping = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0,
    #     patience=0,
    #     verbose=0,
    #     mode='auto'
    # )

    # save model after every epoch
    # checkpointer = ModelCheckpoint(output_dir + '/' + output_name + '_best.hd5', save_best_only=True)

    # possible future feature
    data_augmentation = False
    if data_augmentation:
        print('Using real time data augmentation.')
        # datagen = ImageDataGenerator(
        #     featurewise_center=False, 
        #     samplewise_center=False, 
        #     featurewise_std_normalization=False, 
        #     samplewise_std_normalization=False,
        #     zca_whitening=False, 
        #     rotation_range=45,
        #     width_shift_range=0.05,  
        #     height_shift_range=0.05, 
        #     horizontal_flip=True,
        #     vertical_flip=True,
        #     zoom_range=[0.75,1.3]
        # )
        # datagen.fit(train_imgs)

        # history = model.fit_generator(
        #     datagen.flow(train_imgs, train_labels, batch_size=batch_size),
        #     steps_per_epoch=numSteps,
        #     epochs=nb_epoch,
        #     validation_data=(val_imgs, val_labels),
        #     callbacks=[modelcheckpoint],
        #     class_weight=class_weight
        # )
    else:
        print('Not using data augmentation.')
        model.fit(
            x=train_imgs, y=train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            callbacks=[]
        )
    print('Finished training model')
    print(model.evaluate(x=val_imgs, y=val_labels))
    
    test_model(val_imgs, val_labels, [], output_dir, output_name, 0.5, model)
    model.save_weights(output_dir + '/' + output_name + '.hd5', overwrite=True)
    model_json = model.to_json()
    with open(output_dir + '/' + output_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    return

def test_model(val_imgs, val_labels, val_ids, output_dir, output_name, thresh, model):
    """ Tests a CNN model. This function assumes inputs are well-formed.

    :type val_imgs: numpy.ndarray The array of validation images. Each image is
        a 2 or 3-dimensional array of numbers
    :type val_labels: numpy.ndarray The array of corresponding validation image
        labels.
    :type val_ids: numpy.ndarray The array ofcorresponding validation image ids.
    :type output_name: str The name of the model, without file extensions (this
        should be the output file of train_model)
    :type thresh: float The threshold for determining whether an image is
        classed as 0 or 1. This value should be between 0 and 1.
    """
    # normalize images
    for i in range(val_imgs.shape[0]):
        val_imgs[i] = normalize_img(val_imgs[i])
    
    # find predictions
    print('=' * (28 + len(output_name)))
    print('=====validating model: ' + output_name + '=====')
    print('=' * (28 + len(output_name)))

    _, width, height, channels = val_imgs.shape
    # model = setup_model(width, height, channels
    # model.load_weights(output_dir + '/' + output_name + '.hd5')
    predicted_labels = model.predict(val_imgs)

    print(predicted_labels)   # TODO: get rid of this

    predicted_labels = predicted_labels.flatten()
    Y_one = (predicted_labels >= thresh).astype(int)
    Y_zero = (predicted_labels < thresh).astype(int)
    true_one = val_labels
    true_zero = 1 - val_labels

    TP = np.count_nonzero(Y_one * true_one)
    TN = np.count_nonzero(Y_zero * true_zero)
    FP = np.count_nonzero(Y_one * true_zero)
    FN = np.count_nonzero(Y_zero * true_one)

    # save predictions
    predictions = [val_ids, predicted_labels, val_labels]
    np.save(output_dir + '/' + output_name + '_pred.npy', predictions)

    print('TP: ' + str(TP))
    print('TN: ' + str(TN))
    print('FP: ' + str(FP))
    print('FN: ' + str(FN))

    print('Missed detection rate: ' + str(FN / (TP + FN)))
    print('False positive rate: ' + str(FP / (TN + FP)))
    
    # TODO: i don't know what this does
    # col1 = fits.Column(name='img_id', format='K', array=val_ids)
    # col2 = fits.Column(name='predicted_labels', format='D', array=predicted_labels)
    # cols = fits.ColDefs([col1, col2])
    # tbhdu = fits.BinTableHDU.from_columns(cols)
    # tbhdu.writeto(output_dir + '/' + name + '.fit', overwrite='True')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_name', help='The suffix of the names of the .npy '
                        + 'output files from read_data')
    parser.add_argument('output_name', help='duh')
    args = parser.parse_args()
    npy_name = args.npy_name
    output_name = args.output_name

    # create ouput directory (if doesn't exist)
    OUTPUT_DIR = '/global/homes/l/liuto/520project/transient-detection/cnn/jobs/run_cnn_out/'
    try:
        os.makedirs(OUTPUT_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # proportion of input data to be used for training
    trainRatio = 0.7
    
    print('Loading .npy files.')
    images = np.load('./read_data_out/img_arrays_' + npy_name + '.npy')
    labels = np.load('./read_data_out/img_labels_' + npy_name + '.npy')
    image_ids = np.load('./read_data_out/img_ids_' + npy_name + '.npy')

    # parameters to change: optionally crop images
    images_new = images[:, 10:40, 10:40, :]
    images = images_new
    
    print('Selecting a random sample to train.')
    num_img = np.size(images, 0)
    ntrain = int(trainRatio * num_img)
    all_indices = list(range(0, num_img))
    random.shuffle(all_indices)
    
    train_indices = all_indices[:ntrain]
    val_indices = all_indices[ntrain:num_img]

    train_imgs = images[train_indices]
    train_labels = labels[train_indices].astype(int)

    val_imgs = images[val_indices]
    val_labels = labels[val_indices].astype(int)   
    val_ids = image_ids[val_indices]

    # remove last channel if there are 4 color channels
    if len(train_imgs[0].shape) == 3 and train_imgs[0].shape[2] > 3:
        train_imgs = train_imgs[:, :, :, 0:3]
        val_imgs = val_imgs[:, :, :, 0:3]

    # train model
    train_model(train_imgs, train_labels, val_imgs, val_labels, OUTPUT_DIR, output_name)
    
    # test model
    accept_threshold = 0.5
    test_model(val_imgs, val_labels, val_ids, OUTPUT_DIR, output_name, accept_threshold)

if __name__ == '__main__':
    main()
