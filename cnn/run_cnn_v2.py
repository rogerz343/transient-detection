"""
run_cnn_v2.py: Trains and/or tests a CNN using the input images. The input
data should be the output of read_data_v2.py

arguments:
- the path to the directory containing the three .npy outputs of read_data_v2

parser = argparse.ArgumentParser()
    parser.add_argument('path_to_data', help='Path to the .npy outputs of read_data.py')
    parser.add_argument('name', help='The suffix of the names of the .npy '
                        + 'output files from read_data')
"""

from astropy.io import fits
from collections import Counter

import numpy as np
import random
import argparse
import os
import glob

# import keras
import keras
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

def normalize_img(image):
    """ Returns the image array such that all pixel values are between 0 and 1

    :type image: numpy.ndarray
    :rtype: numpy.ndarray
    """
    max_val = np.amax(image)
    if max_val == 0:
        return image
    return image / max_val

def setup_model(width, height, channels):
    """ Sets up and returns the keras model

    :type width: int The width of the image (in px)
    :type height: int The height of the image (in px)
    :type channels: int The number of color channels in the image
    :rtype: keras.models.Sequential The keras Sequential model
    """
    model = Sequential()

    # first layer: convolution
    model.add(Convolution2D(512, 5, activation='relu',
                            padding='same',
                            input_shape=(width, height, channels),
                            data_format='channels_last'))
    model.add(Dropout(0.15))

    # hidden layer
    model.add(Convolution2D(256, 5, activation='relu', padding='same'))
    model.add(Dropout(0.15))

    # hidden layer
    model.add(Convolution2D(64, 5, activation='relu', padding='same'))
    model.add(Dropout(0.15))

    # hidden layer
    model.add(Convolution2D(32, 5, activation='relu', padding='same'))
    model.add(Dropout(0.15))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))

    # last layer: dense (prediction) with 1 output
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform'))

    print("Compiling...")
    
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    return model

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

def train_model(train_imgs, train_labels, val_imgs, val_labels, model_name):
    """ Trains a CNN. This function assumes inputs are well-formed.

    :type train_imgs: numpy.ndarray The array of train images. Each image is a
        2 or 3-dimensional array of numbers
    :type train_labels: numpy.ndarray The array of corresponding train image
        labels.
    :type val_imgs: numpy.ndarray The array of validation images. These will
        *not* be used during training, but will be used to calculate metrics
        during training to measure progress.
    :type val_labels: numpy.ndarray The array of corresponding validation image
        labels.
    """
    print('========================')
    print('=====Training model=====')
    print('========================')

    # normalize images
    for i in range(train_imgs.shape[0]):
        train_imgs[i] = normalize_img(train_imgs[i])
    
    width, height, channels = np.shape(train_imgs[0])

    # set up model
    model = setup_model(width, height, channels)
    model.summary()

    # training parameters
    batch_size = len(train_imgs) / 20
    epochs = 15
    verbose = 1
    validation_data = (val_imgs, val_labels)
    shuffle = True
    class_weights = balanced_class_weights(train_labels)

    # SGD params
    # lr = 0.001
    # decay = 0 
    # momentum = 0.9
    # nesterov = True
    # numSteps = 4
    # depth = 32
    # nb_dense = 64
    # output params

    # stop early (don't go through all epochs) if model converges
    EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto'
    )

    # save model after every epoch
    ModelCheckpoint(model_name + '_best.hd5', save_best_only=True)

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
            class_weights=class_weights
        )
    print("Finished training model")
    print(model.evaluate(x=val_imgs, y=val_labels))
    
    print('Saving model.')
    model.save_weights(model_name + '.hd5', overwrite=True)
    model_json = model.to_json()
    with open(os.path.splitext(model_name)[0] + '.json', 'w') as json_file:
        json_file.write(model_json)
    return

def test_model(val_imgs, val_labels, val_ids, model_name, thresh, root_path):
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_data', help='Path to the .npy outputs of read_data.py')
    parser.add_argument('name', help='The suffix of the names of the .npy '
                        + 'output files from read_data')
    args = parser.parse_args()
    
    root_path = args.path_to_data
    name = args.name

    # proportion of input data to be used for training
    trainRatio = 0.7
    
    print('Loading .npy files.')
    images = np.load(root_path + "img_arrays_" + name + ".npy")
    labels = np.load(root_path + "img_labels_" + name + ".npy")
    image_ids = np.load(root_path + "img_ids_" + name + ".npy")
    
    print('Selecting a random sample to train.')
    num_img = np.size(images, 0)
    ntrain = int(trainRatio * num_img)
    all_indices = list(range(0, num_img))
    random.shuffle(all_indices)
    
    train_indices = all_indices[:ntrain]
    val_indices = all_indices[ntrain:num_img]

    train_imgs = images[train_indices]
    train_labels = labels[train_indices].astype(int)
    train_ids = image_ids[train_indices]
    val_imgs = images[val_indices]
    val_labels = labels[val_indices].astype(int)   
    val_ids = image_ids[val_indices]

    model_name = root_path + name

    # train model
    train_model(train_imgs, train_labels, val_imgs, val_labels, model_name)
    
    # test model
    accept_threshold = 0.5
    test_model(val_imgs, val_labels, val_ids, model_name, accept_threshold, root_path)

if __name__ == '__main__':
    main()