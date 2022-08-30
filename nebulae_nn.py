# Copyright (c) 2022 Enrique Mondragon Estrada
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Nebulae-NN

from ast import arg
import numpy as np
import argparse
import os
import sys
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re
from nn_math_utils import *
 
def read_data(data_path, labels_file):
    '''
    reads and preprocess the image dataset
    '''

    print('\nloading data...\n ')
    i = 0
    if os.path.isdir(data_path):
        images = sorted(os.listdir(data_path))
        if labels_file != None:
            images = pd.read_csv(labels_file).values[:,0]
    
        for image in images:
            image = image.lower()
            if image.endswith(('.png', '.jpg', '.jpeg')) != True:
                print ('\t WARNING: ignoring file ',image, ' format not supported')
            else:
                im=Image.open(data_path + image)
                imarr = np.array(im)
                print('\timage:\t', i+1, '\t', image, '\tsample shape is', imarr.shape)
                if imarr.ndim == 3:
                    sample = np.reshape(imarr, (imarr.shape[0]*imarr.shape[1]*imarr.shape[2],1))
                    if i==0: 
                        firstshape = imarr.shape
                        dataset = sample
                    else: 
                        if firstshape == imarr.shape:
                            dataset = np.append(dataset, sample, axis=1)
                        else:
                            print('\n\tError: shape of image', image, 'does not match\t\n')
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

    elif os.path.isfile(data_path):
        if data_path.endswith(('.png', '.jpg', '.jpeg')) != True:
            print ('\t WARNING: ignoring file ',data_path, ' format not supported')
        else:
            im = Image.open(data_path)
            imarr = np.array(im)
            print('\timage:\t', str(1), '\t', data_path, '\tsample shape is', imarr.shape)
            if imarr.ndim == 3:
                    sample = np.reshape(imarr, (imarr.shape[0]*imarr.shape[1]*imarr.shape[2],1))
            else:
                print('\n\tError: shape of image', data_path, 'does not match\t\n')
            dataset = sample

    print('\nSize of dataset: ', dataset.shape[1], '\nshape of dataset',dataset.shape, '\n')
    dataset = dataset/255

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
        #params['W' + str(l)] = glorot_min + np.random.randn(dim_layers[l], dim_layers[l-1]) * (glorot_max - glorot_min) # langsam
        #params['W' + str(l)] = np.random.randn(dim_layers[l], dim_layers[l-1]) * glorot_max * 0.5 # langsam
        params['W' + str(l)] = np.random.randn(dim_layers[l], dim_layers[l-1]) * 0.1
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
    C = np.squeeze(binary_cross_entropy(m,A,Y)).astype(np.float64)
    return C


def backward_prop(A,Y, cache, params, activation):
    '''
    Computes the Jacobian's elements of the loss function
    '''
    grads = {}
    L = round((len(cache)-1)/2)
    dCdA = d_binary_cross_entropy(cache['A'+str(L)],Y)
    dA_da = np.ones_like(dCdA) 
    for l in reversed(range(1,L+1)):

        if activation[l-1]=='sigmoid':
            dAdZ = d_sigmoid(cache['Z'+str(l)])
        elif activation[l-1]=='relu':
            dAdZ = d_relu(cache['Z'+str(l)],dA_da)
        elif activation[l-1]=='tanh':
            dAdZ = d_tanh(cache['Z'+str(l)])
        
        dCdZ = dCdA * dA_da * dAdZ

        grads["dA" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_backward(dCdZ,cache['A'+str(l-1)], params['W' + str(l)], params['b' + str(l)])
        dA_da = grads["dA" + str(l-1)] 
    
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

def training(X, Y, params, alpha, epochs, dim_layers, activations):
    '''
    perform forward and backward propagation
    '''
    history = []
    best_cost = 1000 # arbitrary large
    best_epoch = 0
    model_best = params
    for i in range(epochs):
        A, cache = forward_prop(X, params, dim_layers, activations)
        C = cost(A,Y)
        history.append(C)
        grads = backward_prop(A,Y,cache, params, activations)
        params = update_params(params, grads, alpha)
        if C < best_cost:
            model_best = params
            best_cost = C
            best_epoch = i
            print("epoch : ", i, "\tcost:", C, '\t - Current best model')
        else:
            print("epoch : ", i, "\tcost:", C)
    print('\nBest model summary:\n Epoch: ',best_epoch,'Cost: ',best_cost)
    return model_best, history


def save_model(model, epochs, history, alpha):
    '''
    Saves model to a npy file
    and plots the training
    '''
    dir = r"./models"
    if not os.path.exists(dir):
        print("\ncreating output directory...")
        os.mkdir(dir)

    model_path = os.path.join(dir, 'model' + '.npy')
    history_path = os.path.join(dir, 'history' + '.csv')
    print("\nWriting model...")
    np.save(model_path,model) 
    np.savetxt(history_path, history)
    print("model saved!")

    plt.style.use('ggplot')
    plt.plot(range(0,epochs), history)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Nebulae-NN: Training\n'+'\u03B1 = '+str(alpha))
    plt.grid(True)
    
    plot_path = os.path.join(dir, 'training' + '.png')
    plt.savefig(plot_path)
    plt.show()


def load_model(model_path):
    '''
    Load model from a npy file
    '''
    trained_model = np.load(model_path,allow_pickle=True)
    L = int(len(trained_model[()].keys()) / 2)

    params = {}
    layers_dim = []
    for l in range(1,L+1):
        params['W' + str(l)] = trained_model[()]['W'+str(l)]
        params['b' + str(l)] = trained_model[()]['b'+str(l)]
        layers_dim.append(trained_model[()]['W'+str(l)].shape[0])

    return params, layers_dim


def retrieve_model(dataset, layers_dim, activation, params):
    '''
    Retrieve pretrained model
    '''
    dim_layers = []
    dim_layers.append(dataset.shape[0])
    for i in layers_dim:
        dim_layers.append(i)

    activations = []
    np.random.seed(10)
    print('\n\n\n\tRetrieving model of',len(layers_dim), 'layers\n')
    for l in range(1, len(dim_layers)):
        print('\tDimensions for layer', l)
        print('\tweight matrix: ', params['W' + str(l)].shape)
        print('\tbias vector: ', params['b' + str(l)].shape, '\n')
        activations.append(activation)

    activations[-1]='sigmoid'
    print('\n\tactivations for each layer:\n')
    print('\t',activations, '\n') 

    return dim_layers, activations


def main():
    parser = argparse.ArgumentParser(description=' ################ Nebulae-NN ################', usage='%(prog)s')
    parser.add_argument('-in', '--input', type=str, required=True, help='dataset path', dest='data_path')
    parser.add_argument('-lb', '--labels', type=str, required=False, help='labels csv file', dest='labels_file')
    parser.add_argument('-dim', '--dim_layers', action='store', nargs='+', default=[16,8,1], type=int, help='dim of layers separated with spaces', dest='dim_layers') 
    parser.add_argument('-act', '--activation', type=str, choices=['sigmoid', 'relu', 'tanh'], default='relu', dest='activation', help='select activation function for inner layers [sigmoid, relu, tanh]')
    parser.add_argument('-a', '--alpha', type=float, action='store', default=0.001, dest='alpha', help='value at which the parameters will be updated (learning rate)')
    parser.add_argument('-e', '--epochs', type=int, action='store', default=2500, dest='epochs', help='number of complete forward - backward propagation cycles')
    parser.add_argument('-m', '--model', type=str, required=False, help='model path for making predicitions with', dest='model_path')

    args = parser.parse_args()

    print('\n\t',parser.description)
    dataset = read_data(args.data_path, args.labels_file)
    X = dataset # split in future

    if args.model_path:
        # load model and predict
        params, layers_dim = load_model(args.model_path)
        assert len(args.activation)-1==len(layers_dim), 'Check that the number of activations mathches the number of hidden layers'

        dim_layers, activations = retrieve_model(dataset, args.dim_layers, args.activation, params)
        A,_ = forward_prop(X, params, dim_layers, activations)
        print(A)

    else:
        # train
        Y = read_labels(args.labels_file)
        print(Y)
        assert len(Y)==dataset.shape[1], 'Number of samples does not match with number of labels'

        dim_layers, params, activations = define_layers_prop(dataset, args.dim_layers, args.activation)
        print(params.keys(),'\n')

        model, history = training(X, Y, params, args.alpha, args.epochs, dim_layers, activations)

        save_model(model, args.epochs, history, args.alpha)
    

if __name__ == "__main__":

    main()