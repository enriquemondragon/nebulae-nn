import numpy as np
import warnings

def sigmoid(X):
    warnings.filterwarnings('ignore') #RuntimeWarning: overflow encountered in exp
    X = np.array(X, dtype=np.float32)
    A = 1 / (1 + np.exp(-X))

    return A


def d_sigmoid(Z):

    return sigmoid(Z) * (1-sigmoid(Z))


def tanh(Z):

    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


def d_tanh(Z):

    return 1 - tanh(Z)**2


def relu(Z):

    return np.maximum(0,Z)


def d_relu(Z,dCdA):
    dAdZ = np.zeros((dCdA.shape))
    dAdZ[Z <= 0] = 0
    dAdZ[Z > 0] = 1
    assert (dAdZ.shape == Z.shape)

    return dAdZ


def linear_forward(A,W, b):
    Z = W @ A + b

    return Z


def linear_backward(dCdZ,A_prev, W, b):
    m = A_prev.shape[1]
    dZdW = A_prev
    dCdW = (1./m) * dCdZ @ dZdW.T
    dZdb = 1
    dCdb = (1./m) * np.sum(dCdZ * dZdb, axis=1,keepdims=True)
    dA_prev = W.T @ dCdZ
    assert (dA_prev.shape == A_prev.shape)
    assert (dCdW.shape == W.shape)
    assert (dCdb.shape == b.shape)

    return dA_prev, dCdW, dCdb


def binary_cross_entropy(m,A_uc,Y):
    A = clipping(A_uc)
    C= 1./m * ((-Y @ np.log(A).T) - (1-Y) @ (np.log(1-A).T))

    return C


def d_binary_cross_entropy(A_uc,Y):
    A = clipping(A_uc)
    dCdA = - (Y / A) + ((1 - Y) / (1 - A))

    return dCdA
    
def clipping(A_uc, epsilon=1e-5):
    '''clipping implementation to avoid 
    undefined values in log function'''
    A = np.array(A_uc, copy=True) 
    A[A_uc <= 0] = epsilon
    A[A_uc >= 1] = 1 - epsilon

    return A