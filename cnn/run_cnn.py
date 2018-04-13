# import default packages

from astropy.io import fits
from astropy.table import Table
from astropy.modeling.models import Sersic2D

from math import log10
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

from scipy import misc
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate

import argparse
from collections import Counter
import os
import pdb
import glob

# import keras packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import rmsprop
import keras

from keras import backend as K
K.set_image_dim_ordering('th')
                         
from keras.models import model_from_json
from keras.models import model_from_yaml

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

# normalizes the images to less than 1
# TODO: should we really be doing this individually for each image?
#       or should we be finding the max over all images and applying
#       the same "transformation" to all of them?
def normalizeImage(image):
    max_val = np.amax(image)
    if max_val == 0:
        return image
    return image / max_val

"""setupModel: sets up and returns the keras model
   channels: number of color channels in image
   rows: number of rows in the image
   cols: number of cols in the image
   returns: the keras Sequential model
"""
def setupModel(channels, rows, cols):
    model = Sequential()

    # first layer: convolution
    model.add(Convolution2D(512, 5, activation='relu',
                            padding='same',
                            input_shape=(channels, rows, cols),
                            data_format='channels_first'))
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

def train_convnet_GZ00(train_imgs, train_labels, 
                       val_imgs, val_labels,
                       class_weight, model_name):
    ### model params ###
    batch_size = 3 * np.shape(train_imgs)[0]
    nb_epoch = 10
    data_augmentation = False
    # normalize = False
    # y_normalization = False
    # norm_constant = 255
    print(Counter(train_labels))
    ## SGD params ##
    # lr = 0.001
    # decay = 0 
    # momentum = 0.9
    # nesterov = True
    # numSteps = 4
    # depth = 32
    # nb_dense = 64
    # output params
    ver = 1

    img_rows, img_cols = train_imgs.shape[2:4]
    img_channels = 3
    # trainSize = train_imgs.shape[0]

    test_name = model_name
      #Avoid more iterations once convergence
    # patience_par = 10
    # earlystopping = EarlyStopping(monitor='val_loss',
    #                               patience = patience_par,    
    #                               verbose=0,mode='auto')
    # modelcheckpoint = ModelCheckpoint(test_name + "_best.hd5",
    #                 monitor='val_loss',
    #                 verbose=0, save_best_only=True)

    model = setupModel(img_channels, img_rows, img_cols)
    model.summary()

    # RECOVER_MODEL = False
    # if RECOVER_MODEL:
    #     print("we'll figure this out later")

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
        history = model.fit(train_imgs, train_labels,
            batch_size=batch_size,
            epochs=nb_epoch,
            validation_data=(val_imgs, val_labels),
            shuffle=True,
            verbose=ver
        )
    print(model.evaluate(val_imgs, val_labels, batch_size=batch_size))
    print('Weights used:' + str(class_weight))
    
    print('Saving model...')
    model.save_weights(test_name + '.hd5', overwrite=True)
    model_json = model.to_json()
    with open(test_name.split('.')[0]+'.json', 'w') as json_file:
        json_file.write(model_json)
    return

def balanced_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority / count) for cls, count in counter.items()}    

## sets up the model to be trained
## assumes images are m by 3 by x by y
def trainModel(train_imgs, train_labels, train_ids,
               val_imgs, val_labels, val_ids, model_name):
    print("")
    print("=====Training model=====")
    print("========================")
    print('Image sizes:' + str(train_imgs.shape))
    print('Label sizes:' + str(train_labels.shape))

    # normalize images
    for i in range(train_imgs.shape[0]):
        train_imgs[i, 0, :, :] = normalizeImage(train_imgs[i, 0, :, :])
        train_imgs[i, 1, :, :] = normalizeImage(train_imgs[i, 1, :, :])
        train_imgs[i, 2, :, :] = normalizeImage(train_imgs[i, 2, :, :])
        
    classWeights = balanced_class_weights(train_labels)
    train_convnet_GZ00(train_imgs, train_labels,
                       val_imgs, val_labels, classWeights, model_name)
        
    print("Finished training model")
    return

"""validate_convnet_GZ00 returns the predictions made for the input images
   val_imgs: the images to make predictions on
   model_name: the name of the model (used to find the .hd5 model file)
   returns: a vector of prediction *probabilities* (not yet 0-1)
"""
def validate_convnet_GZ00(val_imgs, model_name):
    img_rows, img_cols = val_imgs.shape[2:4]
    image_channels = 3
    model = setupModel(image_channels, img_rows, img_cols)
    model.load_weights(model_name + '.hd5')
    Y_pred = model.predict_proba(val_imgs)
    return Y_pred

def testModel(val_imgs, val_labels, val_ids, model_name, thresh, pathin):
    # normalize val_imgs
    for i in range(0, val_imgs.shape[0]):
        val_imgs[i, 0, :, :] = normalizeImage(val_imgs[i, 0, :, :])
        val_imgs[i, 1, :, :] = normalizeImage(val_imgs[i, 1, :, :])
        val_imgs[i, 2, :, :] = normalizeImage(val_imgs[i, 2, :, :])
    
    # find predictions
    os.chdir(pathin)
    for file in glob.glob('*.hd5'):
        model_name = file[:-4]
        print("=====validating model: " + model_name + "=====")
        print("============================" + ('=' * len(model_name)))

        Y_pred = validate_convnet_GZ00(val_imgs, model_name)
        print("Saving Y_preds")
        print("==============")
        print(Y_pred)
        col1 = fits.Column(name='actual_IDs', format='K', array=val_ids)
        col2 = fits.Column(name='predicted_IDs', format='F', array=Y_pred)

        Y_pred = Y_pred.flatten()
        Y_one = (Y_pred >= thresh).astype(int)
        Y_zero = (Y_pred < thresh).astype(int)
        true_one = val_labels
        true_zero = 1 - val_labels

        TP = np.count_nonzero(Y_one * true_one)
        TN = np.count_nonzero(Y_zero * true_zero)
        FP = np.count_nonzero(Y_one * true_zero)
        FN = np.count_nonzero(Y_zero * true_one)

        print('TP: ' + str(TP))
        print('TN: ' + str(TN))
        print('FP: ' + str(FP))
        print('FN: ' + str(FN))
        
        cols = fits.ColDefs([col1, col2])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.writeto(model_name+".fit", overwrite='True')
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_vectors', help='path to image data and labels')
    parser.add_argument('data_set_name', help='name given to the vectors. this should match the input name to Read_Data.py')
    args = parser.parse_args()
    
    pathin = args.path_to_vectors

    # begin initialize params

    trainRatio = 0.7        # proportion of input to use for training
    img_set_name = args.data_set_name

    # end initialize params

    print('Loading images as npy vectors.')
    images = np.load(pathin + "image_vector_" + img_set_name + ".npy")
    labels = np.load(pathin + "label_vector_" + img_set_name + ".npy")
    image_ids = np.load(pathin + "id_vector_" + img_set_name + ".npy")
    
    print('Selecting a random sample to train.')
    num_img = np.size(images, 0)
    ntrain = int(trainRatio * num_img)
    all_indices = list(range(0, num_img))
    random.shuffle(all_indices)
    
    train_indices = all_indices[:ntrain]
    val_indices = all_indices[ntrain:num_img]

    train_imgs = images[train_indices]
    val_imgs = images[val_indices]
    train_labels = labels[train_indices].astype(int)
    val_labels = labels[val_indices].astype(int)   
    train_ids = image_ids[train_indices]
    val_ids = image_ids[val_indices]

    model_name = pathin + img_set_name

    # train model
    trainModel(train_imgs, train_labels, train_ids, 
               val_imgs, val_labels, val_ids, model_name)
    
    # test model
    accept_threshold = 0.5
    testModel(val_imgs, val_labels, val_ids, model_name, accept_threshold, pathin)
    
if __name__ == '__main__':
    main()
