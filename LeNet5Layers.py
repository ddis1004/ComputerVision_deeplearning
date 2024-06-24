import numpy as np
from util import *

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 중간 데이터（backward 시 사용）
        self.x = None
        self.col = None
        self.col_W = None

        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        out = np.tanh(out)

        return out

    def backward(self, dout):
        dout = dout * ( 1 - np.tanh(dout) **2)
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class AveragePooling:
    def __init__(self, windowSize=2, padding = 0, size = 6):
        self.windowSize = windowSize
        self.Stride = windowSize
        self.W = np.full((1, size), 1/(self.windowSize * self.windowSize))
        self.b = np.zeros((1, size))
        self.padding = padding

    def forward(self, x):

        reshaped_elements = np.array([np.full((self.windowSize, self.windowSize), w) for w in self.W.flatten()])

        N, C, H, W = x.shape
        self.x_shape = x.shape

        filter_h = self.windowSize
        filter_w = self.windowSize
        stride = self.Stride
        pad = self.padding

        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        col = im2col(x, filter_h, filter_w, stride, pad)
        col = col.reshape(-1, filter_h * filter_w)

        out = np.mean(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        out = np.tanh(out)

        return out

    def backward(self, dout):
        dout = dout * ( 1 - np.tanh(dout) **2)
        N, C, H, W = self.x_shape
        filter_h = self.windowSize
        filter_w = self.windowSize
        stride = self.Stride
        pad = self.padding

        dout = dout.transpose(0, 2, 3, 1).flatten()
        dcol = np.zeros((dout.size, filter_h * filter_w))
        dcol[:, :] = dout[:, np.newaxis]

        dx = col2im(dcol, self.x_shape, filter_h, filter_w, stride, pad)
        return dx

class ConvolutionalLayerC3:
    def __init__(self, num_filters=16, filter_size=5, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(num_filters, 6, filter_size, filter_size) * 0.1
        self.b = np.zeros((num_filters, 1))
        self.connections = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5],
            [0, 1, 3, 4],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 1, 4, 5],
            [0, 1, 2, 5],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5],
            [0, 1, 2, 3, 4, 5],
        ]

    def forward(self, x):
        N, C, H, W = x.shape
        F, _, HH, WW = self.W.shape
        pad = self.padding
        stride = self.stride

        out_height = (H + 2 * pad - HH) // stride + 1
        out_width = (W + 2 * pad - WW) // stride + 1

        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

        out = np.zeros((N, F, out_height, out_width))

        for f in range(F):
            filter_connections = self.connections[f]
            for c in filter_connections:
                col = im2col(x_padded[:, c:c + 1, :, :], HH, WW, stride, pad)
                col_W = self.W[f, c].reshape(-1, 1)
                out[:, f, :, :] += col.dot(col_W).reshape(N, out_height, out_width)
            out[:, f, :, :] += self.b[f]
        out = np.tanh(out)
        return out

    def backward(self, dout):
        dout = dout * ( 1 - np.tanh(dout) **2)
        N, C, H, W = self.x_shape
        F, _, HH, WW = self.W.shape
        stride = self.stride
        pad = self.padding

        dx = np.zeros((N, C, H, W))
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)

        for f in range(F):
            filter_connections = self.connections[f]
            for c in filter_connections:
                col = im2col(self.x_padded[:, c:c + 1, :, :], HH, WW, stride, pad)
                col_W = self.W[f, c].reshape(-1, 1)

                dW[f, c] += np.dot(dout_reshaped[f], col).reshape(HH, WW)
                dx[:, c:c + 1, :, :] += col2im(np.dot(dout_reshaped[f].reshape(-1, 1), col_W.T), (N, 1, H, W), HH, WW,
                                               stride, pad)

        db = np.sum(dout, axis=(0, 2, 3)).reshape(F, 1)

        self.dW = dW
        self.db = db
        return dx

class ConvolutionalLayerC5:
    def __init__(self, filter_size=5, stride=1, padding=0):
        self.num_filters = 120
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(120, 16, filter_size, filter_size) * 0.1
        self.b = np.zeros((120, 1))

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        F, _, HH, WW = self.W.shape

        col = im2col(x, HH, WW, self.stride, self.padding)
        col_W = self.W.reshape(F, -1).T
        out = np.dot(col, col_W) + self.b.T
        out = out.reshape(N, F, 1, 1).transpose(1, 2, 3, 0)


        return out

    def backward(self, dout):
        dout = dout * ( 1 - np.tanh(dout) **2)

        F, C, HH, WW = self.W.shape
        N, _, H, W = self.x.shape

        dout = dout.reshape(N, F)
        db = np.sum(dout, axis=0, keepdims=True).T
        col = im2col(self.x, HH, WW, self.stride, self.padding)
        dW = np.dot(dout.T, col).reshape(self.W.shape)
        dcol = np.dot(dout, self.W.reshape(F, -1))
        dx = col2im(dcol, (N, C, H, W), HH, WW, self.stride, self.padding)

        return dx

class FullyConnectedLayerF6:

    def __init__(self, filter_size=1):
        self.W = np.random.randn(84, 120) * 0.1
        self.b = np.zeros((84, 1))

    def forward(self, x):
        out = np.dot(self.W, x)
        out += self.b

        return out

    def backward(self, dout):
        x, col, W_col = self.cache
        F, _, HH, WW = self.W.shape

        dout = dout.transpose(1, 2, 3, 0).reshape(F, -1)
        db = np.sum(dout, axis=1, keepdims=True)
        dW = np.dot(dout, col.T)
        dW = dW.reshape(self.W.shape)
        dcol = np.dot(W_col.T, dout)
        dx = col2im(dcol, x.shape, HH, WW, self.stride, self.padding)

        self.W -= dW * 0.01
        self.b -= db * 0.01

        return dx