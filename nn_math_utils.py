import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def d_sigmoid(X):
    return sigmoid(X) * (1-sigmoid(X))

def tanh(X):
    return(np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

def d_tanh(X):
    return 1 - tanh(X)**2

def relu(X):
    return np.maximum(0,X)

def d_relu(X):
    d_relu[X <= 0] = 0
    d_relu[X > 0] = X
    return d_relu

def linear_forward(A,W, b):
    Z = W @ A + b
    #STORE A,W,B TD
    return Z
