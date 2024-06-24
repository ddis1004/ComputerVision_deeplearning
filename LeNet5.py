from collections import OrderedDict
from LeNet5Layers import *

# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)


class LeNet5:
    def __init__(self, input_dim=(1, 32, 32),
                 conv_param={'filter_num': 16, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 weight_init_std=0.01):

        filter_num = 16
        filter_size = 5
        filter_pad = 0
        filter_stride = 1
        input_size = input_dim[1]

        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)


        self.layers = OrderedDict()
        self.layers['C1'] =  Convolution(self.params['W1'], self.params['b1'],
                                         conv_param['stride'], conv_param['pad'])
        self.layers['S2'] = AveragePooling()
        self.layers['C3'] = ConvolutionalLayerC3()
        self.layers['S4'] = AveragePooling(size = 16)
        self.layers['C5'] = ConvolutionalLayerC5()
        self.layers['F6'] = FullyConnectedLayerF6()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)


input_data = np.ones((1, 1, 32, 32))

# input_data = np.ones((120, 1))

lenet5 = LeNet5()
print(lenet5.predict(input_data))
# a = AveragePooling()
# a = ConvolutionalLayerC3()
# a = AveragePooling(size = 16)
# a = ConvolutionalLayerC5()
# a = FullyConnectedLayerF6()

# print(a.forward(input_data).shape)