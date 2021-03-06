"""
run_cnn_v2.py: Trains a CNN using the input images. The input data should be
the output of read_data_v2.py and these files should be found in
'./read_data_out/'. The output files will be saved to './run_cnn_out/'

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
import pdb
# import keras
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.datasets import mnist

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
    max_val = 256.0#256.0 #np.amax(image)
    newImage = image/max_val
    return newImage

# parameters to change: the model layout
def setup_model(width, height, channels, lrn, modelParams=None):
    """ Sets up and returns the keras model

    :type width: int The width of the image (in px)
    :type height: int The height of the image (in px)
    :type channels: int The number of color channels in the image
    :rtype: keras.models.Sequential The keras Sequential model
    """

    if(modelParams is None):
        modelParams = ([(8, 4, 'relu', 'valid', 0.1),
                        (4, 3, 'relu', 'valid', 0.1)],
                        [(16, 'sigmoid', 0.1)])
    model = Sequential()
    '''
    model.add(Flatten())
    model.add(Dense(output_dim=512, input_dim=width*height*channels ,init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))
    '''
    
    model.add(Conv2D(64, kernel_size=3, activation='relu',
                            padding='valid',
                            input_shape=(width, height, channels)))#,
                            #data_format='channels_last'))    
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    '''
    model.add(Dropout(0.1))
    
    for layer in modelParams[0]:
        model.add(Convolution2D(layer[0], layer[1], 
                        activation=layer[2], padding=layer[3]))
        model.add(Dropout(layer[4]))
    '''
    # flatten to fully connected NN
    #for layer in modelParams[1]:
    #    model.add(Dense(layer[0], activation=layer[1]))
        #model.add(Dropout(layer[2]))

    # last layer: dense (prediction) with 1 output
    model.add(Dense(2, activation='softmax'))#, kernel_initializer='random_uniform'))

    print('Compiling...')
    sgd = optimizers.SGD(lr=lrn, momentum=0.9, nesterov=True)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
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
    weights = {label: float(max_freq) / count for label, count in counter.items()}

    #pdb.set_trace()
    return weights 


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
    print('**********************')
    # normalize images
    '''
    tempImages = []
    for i in range(train_imgs.shape[0]):
        newImage = normalize_img(train_imgs[i])
        tempImages.append(newImage)
    train_imgs = np.array(tempImages)
    '''
    #pdb.set_trace()
    width, height, channels = train_imgs[0].shape
    lrnRate = 0.1
    # set up model
    model = setup_model(width, height, channels, lrnRate)
 
    model.summary()
        # parameters to change: training parameters
    batch_size = 48
    epochs = 4
    verbose = 1
    validation_data = (val_imgs, val_labels)
    shuffle = False
    #class_weight = balanced_class_weights(train_labels)
    #xx = balanced_class_weights(val_labels)
    #pdb.set_trace()
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
    checkpointer = ModelCheckpoint(output_dir + '/' + output_name + '_best.hd5', save_best_only=True)
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
        pdb.set_trace()
        model.fit(
            x=train_imgs, y=train_labels,
            #batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_data,
            #shuffle=shuffle,
            #class_weight=class_weight,
            #callbacks=[checkpointer]
        )
    print('Finished training model')
    print(model.evaluate(x=val_imgs, y=val_labels))
    
    print('Saving model.')
    model.save_weights(output_dir + '/' + output_name + '.hd5', overwrite=True)
    model_json = model.to_json()
    with open(output_dir + '/' + output_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    return

def test_model(val_imgs, val_labels, output_dir, output_name, thresh):
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
    tempImages = []
    for i in range(val_imgs.shape[0]):
        
        temp = normalize_img(val_imgs[i])
        tempImages.append(temp)
    val_imgs = np.array(tempImages)
    # find predictions
    print('=' * (28 + len(output_name)))
    print('=====validating model: ' + output_name + '=====')
    print('=' * (28 + len(output_name)))
    lrnRate = 0.001
    _, width, height, channels = val_imgs.shape
    model = setup_model(width, height, channels, lrnRate)
    model.load_weights(output_dir + '/' + output_name + '.hd5')
    predicted_labels = model.predict(val_imgs)

    #print(predicted_labels)   # TODO: get rid of this

    predicted_labels = np.argmax(predicted_labels, axis=1)
    val_labels = np.argmax(val_labels, axis=1)
    Y_one = (predicted_labels >= thresh).astype(int)
    Y_zero = (predicted_labels < thresh).astype(int)
    true_one = val_labels
    true_zero = 1 - val_labels

    TP = np.count_nonzero(Y_one * true_one)
    TN = np.count_nonzero(Y_zero * true_zero)
    FP = np.count_nonzero(Y_one * true_zero)
    FN = np.count_nonzero(Y_zero * true_one)

    # save predictions
    predictions = [predicted_labels, val_labels]
    print(predictions)
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


def splitData(images, labels, trainRatio):
    print('Selecting a random sample to train.')
    num_img = np.size(images, 0)
    ntrain = int(trainRatio * num_img)
    all_indices = list(range(0, num_img))
    random.shuffle(all_indices)
    
    train_indices = all_indices[:ntrain]
    val_indices = all_indices[ntrain:num_img]

    train_imgs = images[train_indices].astype(float)
    train_labels = labels[train_indices].astype(int)

    val_imgs = images[val_indices].astype(float)
    val_labels = labels[val_indices].astype(int)   
    '''
 #download mnist data and split into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #reshape data to fit model
    X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)
    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    train_imgs = X_train
    train_labels = y_train
    val_imgs = X_test
    val_labels = y_test
    '''


    return train_imgs, train_labels, val_imgs, val_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_name', help='The suffix of the names of the .npy '
                        + 'output files from read_data')
    parser.add_argument('output_name', help='duh')
    parser.add_argument('-r', '--run', action='store_true', help='only test, no train')
    parser.add_argument('-t', '--thresh', help='positive threshold')
    args = parser.parse_args()
    npy_name = args.npy_name
    output_name = args.output_name

    # create ouput directory (if doesn't exist)
    OUTPUT_DIR = './run_cnn_out/'
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
    images_new = images[:, 11:39, 11:39, :]
    images = images_new
    numImgs = len(images)

    rf = 0.03    
    images = images[:int(numImgs*rf)]
    train_imgs, train_labels, val_imgs, val_labels = splitData(images, labels, 0.7)
    
    # remove last channel if there are 4 color channels
    if len(train_imgs[0].shape) == 3 and train_imgs[0].shape[2] > 2:
        train_imgs = train_imgs[:, :, :, 1:3]
        val_imgs = val_imgs[:, :, :, 1:3]
    print(train_imgs[0].shape)
    '''
    REMOVE LATER
   

    (train_imgs, train_labels), (val_imgs, val_labels) = mnist.load_data()

    train_imgs =train_imgs.reshape(60000,28,28,1)
    val_imgs = val_imgs.reshape(10000,28,28,1)

    
    END
    '''
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)
   

    # train model
    if(not args.run):
        train_imgs, train_labels, val_imgs1, val_labels1 = splitData(
                        train_imgs, train_labels, trainRatio)
        train_model(train_imgs, train_labels, 
            val_imgs1, val_labels1, OUTPUT_DIR, output_name)
    thresh = 0.5
    if(args.thresh):
        thresh = float(args.thresh) 
    # test model
    accept_threshold = thresh
    test_model(val_imgs, val_labels, OUTPUT_DIR, output_name, accept_threshold)

def main1():
    #download mnist data and split into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #reshape data to fit model
    X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)
    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    #create model
    model = Sequential()

    #add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    #compile model using accuracy as a measure of model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #train model
    model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=3)
if __name__ == '__main__':
    main()


