"""
merge_autoscan_data.py: Converts 'diff', 'srch', and 'temp' files into one
image with 3 channels. The terms 'diff', 'srch', and 'temp' refer specifically
to the files in the raw data from
<http://portal.nersc.gov/project/dessn/autoscan/>. This script has a very
specific use case and is not general purpose. It also assumes perfectly-formed
input.
"""

import errno
import os
import re

import numpy as np
import imageio
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs_root', help='The path to the root directory '
                        + 'containing the images.')
    args = parser.parse_args()
    imgs_root = args.imgs_root

    root_folder_name = os.path.basename(os.path.normpath(imgs_root))
    new_folder_name = root_folder_name + '_merged/'
    try:
        os.makedirs('./' + new_folder_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # get paths to all images
    img_paths = []
    for root, _, files in os.walk(imgs_root, topdown=False):
        for file in files:
            path = os.path.join(root, file)
            if os.path.isfile(path) and path.lower().endswith('.gif'):
                img_paths.append(path)

    if len(img_paths) == 0:
        print("Error: no images found.")
        return

    # convert all input images to arrays
    id_to_arrays = {}
    for path in img_paths:
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
        img_array = imageio.imread(path)

        if img_id not in id_to_arrays:
            id_to_arrays[img_id] = [None, None, None]
        if file_name.startswith('diff'):
            id_to_arrays[img_id][0] = img_array
        elif file_name.startswith('temp'):
            id_to_arrays[img_id][1] = img_array
        elif file_name.startswith('srch'):
            id_to_arrays[img_id][2] = img_array
        else:
            print('merge_autoscan_data: invalid file name: ' + path)
            return
    
    for img_id, channels in id_to_arrays.items():
        if channels[0] is None or channels[1] is None or channels[2] is None:
            print('merge_autoscan_data: could not find all three files for '
                + 'the image with id ' + str(img_id))
            return
        image_array_full = [channels[0], channels[1], channels[2]]
        image_array_full = np.array(image_array_full)
        image_array_full = np.transpose(image_array_full, (1, 2, 0))
        imageio.imwrite(
            './' + new_folder_name + '/' + str(img_id) + '.gif',
            image_array_full
        )
    print('Data has been saved to ./' + root_folder_name)

if __name__ == '__main__':
    main()