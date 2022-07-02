# Nebulae NN

import numpy as np
import argparse
import os
import sys
from PIL import Image
 
def read_data(data_path):
    '''
    reads the image dataset
    '''

    print('\nloading data...\n ')
    i = 0
    for image in os.listdir(data_path):
        image = image.lower()
        if image.endswith(('.png', '.jpg', 'jpeg')) != True:
            print ('\t WARNING: ignoring file ',image, ' format not supported')
        else:
            im=Image.open(data_path + image)
            imarr = np.array(im)
            print('\timage:\t', i, '\t', image, '\tsample shape is', imarr.shape)
            if imarr.ndim == 3:
                sample = np.reshape(imarr, (imarr.shape[0]*imarr.shape[1]*imarr.shape[2],1))
                if i==0: 
                    firstshape = imarr.shape
                    dataset = sample
                else: 
                    if firstshape == imarr.shape:
                        dataset = np.append(dataset, sample, axis=1)
                    else:
                        print('\n\tError: shape of image', image, 'dont match\t\n')
                        i -= 1
                        sys.exit
                i += 1
            elif imarr.ndim == 2:
                sample = np.reshape(imarr, (imarr.shape[0]*imarr.shape[1],1))
                if i==0: 
                    firstshape = imarr.shape
                    dataset = sample
                else: 
                    if firstshape == imarr.shape:
                        dataset = np.append(dataset, sample, axis=1)
                    else:
                        print('\n\tError: shape of image', image, 'mismatch\t\n')
                        i -= 1
                        sys.exit
                i += 1
            else:
                print('\tError: Check dimensions of the data')

    print('\nSize of data set: ', dataset.shape[1], '\nshape of dataset',dataset.shape, '\n')

def main():
    parser = argparse.ArgumentParser(description=' ################ Nebulae NN ################', usage='%(prog)s')
    parser.add_argument('-in', '--input', type=str, required=True, help='dataset path', dest='data_path')

    args = parser.parse_args()

    data_path = args.data_path
    read_data(data_path)
    
    

if __name__ == "__main__":

    main()