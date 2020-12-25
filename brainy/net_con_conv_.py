from preprocess import get_train_test
from net_con_ import NetCon
import numpy as np

from nn_constants import *


class Conv2D:
    def __init__(self):
        self.f_height = 0  # сторона фильтра - ширина
        self.f_width = 0  # сторона фильтра - высота
        self.n_f = 0  # количество фильтров
        self.s = 0  # шаг

        self.out = None  # выходная карта признаков
        self.filt = None  # фильтр
        self.bias = None  # биас

        self.act_func = RELU  # функция активации


class NetConConv(NetCon):
    def __init__(self):
        super().__init__()

        self.net_conv = [None]*3
        for i in range(3):  # Статическое выделение слоев
            self.net_conv[i] = Conv2D()
        self.sp_conv = -1  # алокатор для conv слоев

    def np_operations(self, op, x):
        y = np.zeros(x.shape[0])
        height = x.shape[0]

        if op == RELU:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = 0
                else:
                    y[row] = x[row][0]

            return np.array([y]).T

        elif op == RELU_DERIV:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = 0
            else:
                y[row] = 1
            return np.array([y]).T
        elif op == TRESHOLD_FUNC_HALF:
            for row in range(height):
                if (x[row][0] > 0.5):
                    y[row] = 1
                else:
                    y[row] = 0
            return np.array([y]).T
        elif op == TRESHOLD_FUNC_HALF_DERIV:
            return 1
        elif op == LEAKY_RELU:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = self.alpha_leaky_relu * x[row][0]
                else:
                    y[row] = x[row][0]
            return np.array([y]).T

        elif op == LEAKY_RELU_DERIV:
            for row in range(height):
                if (x[row][0] <= 0):
                    y[row] = self.alpha_leaky_relu
                else:
                    y[row] = 1
            return np.array([y]).T
        elif op == SIGMOID:
            y = 1 / (1 + np.exp(-self.alpha_sigmoid * x))
            return y
        elif op == SIGMOID_DERIV:
            return self.alpha_sigmoid * x * (1 - x)
        elif op == TAN:
            y = self.alpha_tan * np.tanh(self.beta_tan * x)
            return y
        elif op == TAN_DERIV:
            return self.beta_tan * self.alpha_tan * 4 / ((np.exp(self.beta_tan * x) + np.exp(-self.beta_tan * x))**2)
        else:
            print("Op or function does not support ", op)

    def cr_conv2d_lay(self, filt_shape, s, act_func=RELU, first_image_shape=(None, None, None)):
        n_f, f_height, f_width, n_c_f = filt_shape

        in_dim_height, in_dim_width, n_c = first_image_shape

        out = None
        # если здесь мы задаем первое 'изображение' и определяем выходную карту признаков
        # для данного слоя
        if not in_dim_height and not in_dim_width and not n_c:
            nc_after_conv = 1
            out_dim_width = int((in_dim_width - f_width)/s)+1
            out_dim_height = int((in_dim_height - f_height)/s)+1

            assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"

            out = np.zeros((n_f, out_dim_height, out_dim_width))

        filt = np.zeros((n_f, f_height, f_width, n_c_f))

        for fil in range(n_f):
            for row in range(f_height):
                for elem in range(f_width):
                    for channel in range(n_c_f):
                        filt[fil, row, elem, channel] = self.operations(
                            INIT_W_MY, 0)

        bias = np.zeros((n_f, 1))

        for fil in range(n_f):
            bias[fil, 0] = self.operations(INIT_W_MY, 0)

        self.sp_conv += 1
        layer: Conv2D = None
        layer = self.net_conv[self.sp_conv]
        layer.f_height = f_height
        layer.f_width = f_width
        layer.s = s
        layer.n_f = n_f
        layer.out = out
        layer.filt = filt
        layer.bias = bias
        layer.act_func = act_func

        self.b_c_forward.append(CONV2D_OP)
        self.b_c_forward.append(self.sp_conv)

    def convolution(self, layer, inputs):

        in_dim_height = inputs.shape[0]
        in_dim_width = inputs.shape[1]

        out_dim_width = int((in_dim_width - layer.f_width)/layer.s)+1
        out_dim_height = int((in_dim_height - layer.f_height)/layer.s)+1

        layer.out = np.zeros((layer.n_f, out_dim_height, out_dim_width))

        # convolve each filter over the image
        len_layer_n_f = layer.n_f
        for curr_f in range(len_layer_n_f):
            print('cur f', curr_f)
            curr_y = out_y = 0
            # move filter vertically across the image
            while curr_y + layer.f_height <= in_dim_height:
                curr_x = out_x = 0
                # move filter horizontally across the image
                while curr_x + layer.f_width <= in_dim_width:
                    # perform the convolution operation and add the bias
                    # summ - одно число, результат свертки
                    summ = np.sum(
                        # cur_f 3 d тензор фильтр
                        layer.filt[curr_f] *
                        # inputs - тоже 3 d тензор изображение
                        inputs[curr_y:curr_y+layer.f_height,\
                               curr_x:curr_x+layer.f_width,\
                               :]) + layer.bias[curr_f]
                    # out без curr_f 2 d тензор - матрица, так это ряд карт признаков, 2 d матриц
                    layer.out[curr_f, out_y, out_x] = summ 

                    curr_x += layer.s
                    out_x += 1
                curr_y += layer.s
                out_y += 1


if __name__ == '__main__':
    def main():
        cnv: NetConConv = None
        cnv = NetConConv()
        x, x_test, y, y_test = get_train_test()
        x_0 = x[0].reshape(x[0].shape[0], x[0].shape[1], 1)
        print('x', x[0])
        print('y', y[0])
        cnv.cr_conv2d_lay((1, 5, 5, 1), 1, first_image_shape=(20, 18, 1))
        cnv.convolution(cnv.net_conv[0], x_0)

    main()
