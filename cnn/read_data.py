"""
read_data_v2.py
Reads in image files for binary classification and saves three .npy files
(numpy arrays) of the image data.

arguments:
- The path to the root directory that contains the image files. Images will be
  read from this directory and all subdirectories.
- The image type (file extension).
- The path to a .csv file that contains the image labels.
- The maximum number of images to read.
- A name to be used to name the output files of this code

Each input image results in a pixel array, a label, and an id. This script
saves three arrays (as files) to the directory "./read_data_out/".
- "imgs_[name].npy": A numpy array of the image pixel arrays
- "labels_[name].npy": A numpy array of the image labels (classes)
- "ids_[name].npy": A numpy array of the image id's
where "[name]" is the name given as the fourth input argument. The
information about each input image is stored into the same index in all three
arrays (i.e. the i'th image's data is stored in the i'th index of each array).

Additional input format requirements
The images should follow the following requirements:
- all images should have the same size
- all images should have the same type
- all images should have a unique numeric id (a string containing only digits)
  associated with them, and this id should be the suffix of the file name
  (excluding the file extension)
The image type argument should be one of the following (without quotes)
- ".gif"
- ".fits"
The image label file should be a .csv table such that:
- 'ID' is one of the columns (this corresponds to the id in the filename)
- 'OBJECT TYPE' is one of the columns (this corresponds to the label/class of
  the image)
- the OBJECT TYPE value must be either 0 or 1

"""
import errno
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import imageio
import argparse

import scipy.misc
from scipy import ndimage

def get_img_paths(imgs_root, file_ext, max_img_num):
    """ Returns a list of the paths to the image files found. This function
    searches through the root directory as well as all subdirectories

    :type imgs_root: str The path to the root directory containing the image
        files
    :rtype: list A list of paths to the image files
    """
    counter = 0
    for root, _, files in os.walk(imgs_root, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if os.path.isfile(path) and path.lower().endswith(file_ext):
                yield path
                counter += 1
        sys.stderr.write('\rimages found: ' + str(counter))
        sys.stderr.flush()
        if(counter > max_img_num * 7):
            break
    print('\nNumber of files found: ' + str(len(paths)) + '.')


def throw_out_data(img_arrays, img_labels, img_ids, P, pred):
    """ Iterates through the three image-data arrays and, if the image satisfies
    pred, then with probability P, the image data is deleted from all three
    arrays.

    :type img_arrays: list A list of image data arrays (each index is one image)
    :type img_labels: list A list of the corresponding image labels
    :type img_ids: list A list of the corresponding image id's
    :type P: float The probability (a number between 0 and 1) of deleting a
        particular image
    :type pred: function A (lambda) function that takes in three arguments:
        an image array, the corresponding image label, and the corresponding
        image id, and returns a Boolean
    :rtype: tuple The tuple (img_arrays, img_labels, img_ids) of the modified
        input data
    """
    for i in reversed(range(0, len(img_arrays))):
        if pred(img_arrays, img_labels, img_ids) and random.random() < P:
            del img_arrays[i]
            del img_labels[i]
            del img_ids[i]
    return (img_arrays, img_labels, img_ids)


def blur_images(img_arrays, blurFactor = 0):
    """
takes a list of N channel images [:,:,k] where k runs from 0 to N-1, and blurs the images 
by some blur factor

:type img_arrays: list A list of image data arrays (each index is one image)
:type blurFactor: a real number >= 0, where 0 returns the same image

:retype: the same img_arrays but blurred
    """
    I, x, y, K = img_arrays.shape
    for i in range(I):
        for k in range(K):
            img_arrays[i, :, :, k] = (
                ndimage.gaussian_filter(img_arrays[i, :, :, k], sigma=blurFactor))
    return img_arrays


def read_data(img_paths, labels_file, max_num_imgs, name, blurFactor = 0):
    """ Reads in the images from img_path and the labels from labels_file, and
    saves them as .npy files

    :type img_paths: list A list of file paths to the images
    :type labels_file: str The path of the .csv file to read labels from
    :type max_num_imgs: int The maximum number of images to read
    :type name: str A name used for naming the output files
    """
    '''
    if len(img_paths) == 0:
        print("Error: no images found.")
        return
    '''
    num_imgs = max_num_imgs
    sample_img = imageio.imread(next(img_paths))
    if len(sample_img.shape) > 2:
        width, height, channels = sample_img.shape
    else:
        width, height = sample_img.shape
        channels = 1
    labels_table = pd.read_csv(labels_file)

    img_arrays = []
    img_labels = []
    img_ids = []
    for i, path in enumerate(img_paths):
        file_name_ext = os.path.basename(path)
        file_name = os.path.splitext(file_name_ext)[0]
        match = re.match(r'.*\D(\d+$)', file_name)
        if not match:
            match = re.match(r'(\d+$)', file_name)
        if not match:
            print('read_data: invalid filename detected. file name should '
                  + 'end with a suffix of one or more digits')
            return
        img_id = match.group(1)
        try:
            img_array = imageio.imread(path)
        except ValueError:
            continue
        if len(img_array.shape) < 3:
            img_array = img_array[:, :, np.newaxis]
        if img_array.shape != (width, height, channels):
            print('read_data: error: image dimensions are not consistent')
            return
        row = labels_table.loc[labels_table['ID'] == int(img_id)]
        label = row.iloc[0]['OBJECT_TYPE']
        if label != 0 and label != 1:
            print('Error: found label that is not 0 or 1')
            return

        img_arrays.append(img_array)
        img_labels.append(label)
        img_ids.append(img_id)

        if(i % 500 == 0):
            print('Reading images: ' + str(int(100 * i / num_imgs)) + '%')
        if(i > num_imgs):
            break
    print('Reading images: 100%')

    # post-processing, printing information, etc.

    # throw out data here if u want to
    # img_arrays, img_labels, img_ids = \
    #     throw_out_data(img_arrays, img_labels, img_ids, \
    #     0, lambda x, y, z: y == 1)
    
    num_pos = 0
    num_neg = 0
    for label in img_labels:
        if label == 1:
            num_pos += 1
        else:
            num_neg += 1
    print('Number of positive samples: ' + str(num_pos))
    print('Number of negative samples: ' + str(num_neg))

    # save files
    img_arrays_np = np.array(img_arrays)
    img_labels_np = np.array(img_labels)
    img_ids_np = np.array(img_ids)
    try:
        os.makedirs('./read_data_out/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    blur_images(img_arrays_np, blurFactor)
    np.save('./read_data_out/img_arrays_' + name + '.npy', img_arrays_np) 
    np.save('./read_data_out/img_labels_' + name + '.npy', img_labels_np) 
    np.save('./read_data_out/img_ids_' + name + '.npy', img_ids_np)
    print('Data has been saved to ./read_data_out/')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs_root', help='The path to the root directory '
                        + 'containing the images.')
    parser.add_argument('file_ext', help='The file extension of the '
                        + 'images. Should be one of: [.gif, .fits]')
    parser.add_argument('labels_file', help='The path to the .csv file '
                        + 'containing the image labels (classes)')
    parser.add_argument('max_num_imgs', help='The maximum number of images to read.')
    parser.add_argument('name', help='a string used to name the output files.')
    parser.add_argument('-b', '--blur', help='blur factor')
    args = parser.parse_args()

    print('Reading images from: ' + args.imgs_root + '.')
    print('Reading image labels from: ' + args.labels_file)

    
    img_paths = get_img_paths(args.imgs_root, args.file_ext, int(args.max_num_imgs))
    read_data(img_paths, args.labels_file, int(args.max_num_imgs), args.name)

if __name__ == '__main__':
    main()
