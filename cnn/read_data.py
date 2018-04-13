#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 01:58:15 2018

Read_Data.py: creates npy vectors (.npy files) based on input image data.
    Each input image yields a corresponding ID, pixel array, and label
    param: a directory of .gif files (51 x 51)
    param: a csv of labels for the .gif files
    param: the number of images to read
    param: a name used for the output files of this code

updates
- windows and linux compatible
- fixed some bug-Os
- renamed some variables and stuff
"""

import os
import errno
import numpy as np
from scipy import misc
from scipy.ndimage import zoom
import re
import pandas as pd
import imageio

import argparse

"""
    get_gif_paths: returns a list of the paths to the .gif files in the
                   root directory as well as all subdirectories
    @param img_path the path to the root folder containing the .gif files
    @return a list of paths to each of the .gif files
"""
def get_gif_paths(img_path):
    file_list = list()
    for root, _, files in os.walk(img_path, topdown=False):
        for file in files:
            file_list.append(os.path.join(root, file))

    gifnamevector = [path for path in file_list if os.path.isfile(path) and '.gif' in path]
    print("number of .gif files found: " + str(np.size(gifnamevector)))
    return gifnamevector

"""throw_out_data: iterates through the input vectors and throws out
                   samples with probability P. Useful for getting a more
                   even ratio of positive to negative samples
        P: probability of throwing out a positive sample
        img_vector: the vector of images
        Y: the labels corresponding to the images in img_vector
        idvec: the id's of the labels corresponding to the images in img_vector
        returns: a 3-tuple (img_vector, Y, idvec) of the three input vectors
                 with some of the data thrown out
"""
def throw_out_data(P, img_vector, Y, idvec):
    import random
    to_keep = list(range(0, len(Y)))
    for i in reversed(range(0, len(Y))):
        if Y[i] == 1 and random.random() < P:
            del to_keep[i]
    to_keep = np.array(to_keep)
    Y = Y[to_keep]
    img_vector = img_vector[to_keep]
    idvec = idvec[to_keep]
    return (img_vector, Y, idvec)

"""
    img_path: path to images
    featuresPath: path to features csv file
    max_num_img: number of images to read
"""
def read_data(img_path, featurePath, max_num_img, name):
    size_im = 51
    print('getting names from directories')
    gifnamevector = get_gif_paths(img_path)
    max_num_img = min(max_num_img, np.shape(gifnamevector)[0])

    img_vector = np.zeros([max_num_img, 3, size_im, size_im])
    Y = np.zeros(max_num_img)
    idvec = np.array([''] * max_num_img)
    autoscan = pd.read_csv(featurePath)
    print('looping through images')
    for index, gif_path in enumerate(gifnamevector):
        gif_fullname = os.path.basename(gif_path)
        gif_name = gif_fullname[:-4]
        gif_id = re.search(r"\d+(\.\d+)?", gif_name).group(0)
        matching = [s for s in gifnamevector if gif_id in s]
        if gif_id not in idvec:
            d_t_s_found = [False, False, False]
            for gif_address in matching:
                img_data = imageio.imread(gif_address)
                if 'diff' in gif_address:
                    img_vector[index, 0, :, :, ] = img_data
                    d_t_s_found[0] = True
                elif 'temp' in gif_address:
                    img_vector[index, 1, :, :, ] = img_data
                    d_t_s_found[1] = True
                elif 'srch' in gif_address:
                    img_vector[index, 2, :, :, ] = img_data
                    d_t_s_found[2] = True
            if d_t_s_found != [True, True, True]:
                print("id = " + gif_id + ": [diff, temp, srch] = " + str(d_t_s_found))
            
            idvec[index] = gif_id
            featurerow = autoscan.loc[autoscan['ID'] == int(gif_id)] #1 row by 40 column dataframe
            Y[index] = featurerow.iloc[0]['OBJECT_TYPE']

        if index % 100 == 0:
            print ("Iteration: ", index)

    # printing potentially useful information
    if len(Y) != len(img_vector) or len(Y) != len(idvec):
        print("Error: number of samples != number of labels")
    img_vector, Y, idvec = throw_out_data(0, img_vector, Y, idvec)
    
    num_positive = 0
    num_negative = 0
    for val in Y:
        if val == 1:
            num_positive += 1
        elif val == 0:
            num_negative += 1
    print("number of positive samples: " + str(num_positive))
    print("number of negative samples: " + str(num_negative))

    print("Saving image and target vector")
    try:
        os.makedirs('./read_data_out/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    np.save("./read_data_out/image_vector_" + name + ".npy", img_vector) 
    np.save("./read_data_out/id_vector_" + name + ".npy", idvec) 
    np.save("./read_data_out/label_vector_" + name + ".npy", Y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgpath', help='path to images')
    parser.add_argument('features', help='path to feature data')
    parser.add_argument('num_images', help='number of images to read')
    parser.add_argument('name', help='name used for naming output file')
    args = parser.parse_args()
    img_path = args.imgpath
    featurePath = args.features
    print('reading images from: ' + img_path)
    print('reading feature data from: ' + featurePath)
    max_num_img = int(args.num_images)
    read_data(img_path, featurePath, max_num_img, args.name)

if __name__ == '__main__':
    main()
