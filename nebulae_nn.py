# Nebulae NN

from ast import arg
import numpy as np
import argparse
import os
import sys
from PIL import Image
from nn_math_utils import *
 
def read_data(data_path):
    '''
    reads and preprocess the image dataset
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

    print('\nSize of dataset: ', dataset.shape[1], '\nshape of dataset',dataset.shape, '\n')
    dataset = dataset/255
    print(dataset)

    return dataset


def define_layers_prop(dataset, layers_dim, activation):
    '''
    Create the layers of the networks
    initialize the weights using Glorot initialization and bias with zeros
    Define the activation function for earch layer
    '''
    dim_layers = []
    dim_layers.append(dataset.shape[0])
    for i in layers_dim:
        dim_layers.append(i)

    params = {}
    activations = []
    np.random.seed(10)
    print('\n\n\n\tBuilding network of',len(layers_dim), 'layers\n')
    print('\n\tInitializing weights for each layer...\n')
    for l in range(1, len(dim_layers)):
        glorot_min = -1 / np.sqrt(dim_layers[l-1]), 
        glorot_max = 1 / np.sqrt(dim_layers[l-1])
        params['W' + str(l)] = glorot_min + np.random.randn(dim_layers[l], dim_layers[l-1]) * (glorot_max - glorot_min)
        params['b' + str(l)] = np.zeros((dim_layers[l], 1))
        print('\tDimensions for layer', l)
        print('\tweight matrix: ', params['W' + str(l)].shape)
        print('\tbias vector: ', params['b' + str(l)].shape, '\n')
        activations.append(activation)

    activations[-1]='sigmoid'
    print('\n\tactivations for each layer:\n')
    print('\t',activations, '\n') 

    return dim_layers, params, activations


def forward_prop(X, params, dim_layers, activation):
    A = X
    for l in range(1, len(dim_layers)):
        A_prev = A
        Z = linear_forward(A_prev,params['W' + str(l)], params['b' + str(l)]) # STORE LINEAR CACHE TD
        if activation[l]=='sigmoid': # STORE ACTIVATION TD
            A = sigmoid(Z)
        elif activation[l]=='relu':
            A = relu(Z)
        elif activation[l]=='tanh':
            A = tanh(Z)
    return
        

def main():
    parser = argparse.ArgumentParser(description=' ################ Nebulae NN ################', usage='%(prog)s')
    parser.add_argument('-in', '--input', type=str, required=True, help='dataset path', dest='data_path')
    parser.add_argument('-dim', '--dim_layers', action='store', nargs='+', default=[16,8,1], type=int, help='dim of layers separated with spaces', dest='dim_layers')
    parser.add_argument('-act', '--activation', type=str, choices=['sigmoid', 'relu', 'tanh'], default='relu', dest='activation', help='select activation function for inner layers [sigmoid, relu, tanh]')
    args = parser.parse_args()

    data_path = args.data_path
    dataset = read_data(data_path)
    dim_layers, params, activations = define_layers_prop(dataset, args.dim_layers, args.activation)


if __name__ == "__main__":

    main()