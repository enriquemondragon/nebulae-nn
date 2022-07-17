# Nebulae NN

from ast import arg
import numpy as np
import argparse
import os
import sys
from PIL import Image
import pandas as pd
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

def read_labels(labels_file):
    '''
    Extracting label values
    '''
    labels = pd.read_csv(labels_file).values
    Y = np.array(labels[:,1])
    return Y

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
        #params['W' + str(l)] = glorot_min + np.random.randn(dim_layers[l], dim_layers[l-1]) * (glorot_max - glorot_min)
        params['W' + str(l)] = np.random.randn(dim_layers[l], dim_layers[l-1]) * glorot_max *0.5
        #params['W' + str(l)] = np.random.randn(dim_layers[l], dim_layers[l-1]) *0.1
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
    '''
    Computes forward propagation
    '''
    A = X
    cache = {}
    cache['A' + str(0)] = A
    for l in range(1, len(dim_layers)):
        A_prev = A
        Z = linear_forward(A_prev,params['W' + str(l)], params['b' + str(l)])
        cache['Z' + str(l)] = Z
        if activation[l-1]=='sigmoid':
            A = sigmoid(Z)
        elif activation[l-1]=='relu':
            A = relu(Z)
        elif activation[l-1]=='tanh':
            A = tanh(Z)
        cache['A' + str(l)] = A
    return A, cache
        
def cost(A,Y):
    '''
    Compute the cost of the objective function
    '''
    m = Y.shape[0]
    C = np.squeeze(binary_cross_entropy(m,A,Y))
    return C

def backward_prop(A,Y, cache, params, activation):
    '''
    Computes the Jacobian's elements of the loss function
    '''
    grads = {}
    L = round((len(cache)-1)/2)
    dCdA = d_binary_cross_entropy(cache['A'+str(L)],Y)

    for l in reversed(range(1,L+1)):
        if activation[l-1]=='sigmoid':
            dAdZ = d_sigmoid(cache['Z'+str(l)])
        elif activation[l-1]=='relu':
            dAdZ = d_relu(cache['Z'+str(l)],dCdA)
        elif activation[l-1]=='tanh':
            dAdZ = d_tanh(cache['Z'+str(l)])

        dCdZ = dCdA * dAdZ

        grads["dA" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_backward(dCdZ,cache['A'+str(l-1)], params['W' + str(l)], params['b' + str(l)]) # I will use params, CHECK
        dCdA = grads["dA" + str(l-1)]

    return grads

def update_params(params, grads, alpha):
    '''
    update parameters
    '''
    L = len(params) // 2
    for l in range(1,L+1):
        params["W" + str(l)] = params["W" + str(l)] - alpha * grads["dW" + str(l)]
        params["b" + str(l)] = params["b" + str(l)] - alpha * grads["db" + str(l)]
    return params


def main():
    parser = argparse.ArgumentParser(description=' ################ Nebulae NN ################', usage='%(prog)s')
    parser.add_argument('-ind', '--input_data', type=str, required=True, help='dataset path', dest='data_path')
    parser.add_argument('-lb', '--labels', type=str, required=True, help='labels csv file', dest='labels_file')
    parser.add_argument('-dim', '--dim_layers', action='store', nargs='+', default=[16,8,1], type=int, help='dim of layers separated with spaces', dest='dim_layers')
    parser.add_argument('-act', '--activation', type=str, choices=['sigmoid', 'relu', 'tanh'], default='relu', dest='activation', help='select activation function for inner layers [sigmoid, relu, tanh]')
    args = parser.parse_args()

    print('\n\t',parser.description)
    dataset = read_data(args.data_path)
    Y = read_labels(args.labels_file)
    print(Y)
    assert len(Y)==dataset.shape[1], 'Number of samples does not match with number of labels'

    dim_layers, params, activations = define_layers_prop(dataset, args.dim_layers, args.activation)

    X = dataset # split in future


    alpha = 0.0075
    iterations=3000
    for i in range(iterations):
        print("iteration",i)
        A, cache = forward_prop(X, params, dim_layers, activations)
        C = cost(A,Y)
        print("cost is",C)
        grads = backward_prop(A,Y,cache, params, activations)

        params = update_params(params, grads, alpha)




if __name__ == "__main__":

    main()