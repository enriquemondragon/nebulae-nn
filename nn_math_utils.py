import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def d_sigmoid(Z):
    return sigmoid(Z) * (1-sigmoid(Z))

def tanh(Z):
    return(np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def d_tanh(Z):
    return 1 - tanh(Z)**2

def relu(Z):
    return np.maximum(0,Z)

def d_relu(Z):
    d_relu[Z <= 0] = 0
    d_relu[Z > 0] = Z
    return d_relu

def linear_forward(A,W, b):
    Z = W @ A + b
    #STORE A,W,B TD
    return Z

def linear_backward(dCdZ,A_prev, W, b):
    m = A_prev.shape[1]
    dZdW = A_prev
    dCdW = (1/m) * dCdZ @ dZdW.T
    dZdb = 1
    dCdb = (1/m) * np.sum(dCdZ * dZdb, axis=1,keepdims=True)
    dA_prev = W.T @ dCdZ
    assert (dA_prev.shape == A_prev.shape)
    assert (dCdW.shape == W.shape)
    assert (dCdb.shape == b.shape)
    return dA_prev, dCdW, dCdb


def binary_cross_entropy(m,A,Y):
    C= 1/m * -(Y @ np.log(A).T) - (1-Y) @ (np.log(1-A).T)
    return C

def d_binary_cross_entropy(A,Y):
    dCdA = - Y / A - ((1 - Y) / (1 - A))
    return dCdA

    