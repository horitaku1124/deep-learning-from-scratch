# coding: utf-8
import numpy as np


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    result = y[5:len(y)-5]
    print("smooth_curve.result", type(result))
    return result


def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    # Array https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html#numpy.random.permutation
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    result = x, t
    print("shuffle_dataset.result", type(result))
    return result

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    result = (input_size + 2*pad - filter_size) / stride + 1
    return result


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Args:
        input_data(numpy.ndarray) : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
        filter_h(Int) : フィルターの高さ
        filter_w(Int) : フィルターの幅
        stride(Int) : ストライド
        pad(Int) : パディング

    Returns:
        col(numpy.ndarray) : 2次元配列
    """

    # Int N, C, H, W
    N, C, H, W = input_data.shape

    # Int out_h, out_w
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # numpy.ndarray img
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    # Initialize array with 0 Nth array
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    # numpy.ndarray col
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # numpy.ndarray col
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    　P222を参照
    Args:
        col(numpy.ndarray) :
        input_shape(tuple) : 入力データの形状（例：(10, 1, 28, 28)）
        filter_h(Int) :
        filter_w(Int)
        stride(Int)
        pad(Int)

    Returns
        numpy.ndarray: 

    """
    # Int N, C, H, W
    N, C, H, W = input_shape
    # Int out_h, out_w
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # numpy.ndarray col
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    result = img[:, :, pad:H + pad, pad:W + pad]
    return result