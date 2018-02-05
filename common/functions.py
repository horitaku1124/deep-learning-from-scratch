# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


"""
    Retutn: ndarray[Double]
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y, t):
    """
        Returns:
            numpy.float64:
    """
    result = 0.5 * np.sum((y-t)**2)
    return result


def cross_entropy_error(y, t):
    """
        Args:
            y(numpy.ndarray): 2d array
            t(numpy.ndarray): 1d array
        Returns:
            numpy.float64:
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    result = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return result


def softmax_loss(X, t):
    y = softmax(X)
    result = cross_entropy_error(y, t)
    print("softmax_loss.result", result, type(result))
    return result
