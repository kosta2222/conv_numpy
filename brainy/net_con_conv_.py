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

    def cr_conv2d_lay(self, filt_shape, s, act_func=RELU, first_image_shape=(None, None, None)):
        n_f, f_height, f_width, n_c_f = filt_shape

        in_dim_height, in_dim_width, n_c = first_image_shape

        out = None
        # если здесь мы задаем первое 'изображение' и определяем выходную карту признаков
        # для данного слоя
        if not in_dim_height and not in_dim_width and not n_c:
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
        self.b_c_bacward_tmp.append(self.sp_conv)
        self.b_c_bacward_tmp.append(CONV2D_OP)

    def cr_flatten(self):
        self.sp_d += 1
        layer = self.net_dense[self.sp_d]
        self.b_c_forward.append(FLATTEN)
        self.b_c_forward.append(self.sp_d)
        self.b_c_bacward_tmp.append(self.sp_d)
        self.b_c_bacward_tmp.append(FLATTEN)

    def convolution(self, layer: Conv2D, inputs):

        in_dim_height = inputs.shape[0]
        in_dim_width = inputs.shape[1]

        out_dim_width = int((in_dim_width - layer.f_width)/layer.s)+1
        out_dim_height = int((in_dim_height - layer.f_height)/layer.s)+1

        layer.out = np.zeros((layer.n_f, out_dim_height, out_dim_width))

        # convolve each filter over the image
        len_layer_n_f = layer.n_f
        layer_act_func = layer.act_func
        for curr_f in range(len_layer_n_f):
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
                    layer.out[curr_f, out_y, out_x] = self.operations(
                        layer_act_func, summ)

                    curr_x += layer.s
                    out_x += 1
                curr_y += layer.s
                out_y += 1
        print('layer out shp', layer.out.shape)

    def feed_forwarding(self, inputs):
        len_b_c_forward = len(self.b_c_forward)
        while self.ip < len_b_c_forward:
            op = self.b_c_forward[self.ip]
            if op == DENSE:
                self.ip += 1
                i = self.b_c_forward[self.ip]

                if i == 0:
                    layer = self.net_dense[0]
                    self.make_hidden(layer, inputs)
                else:
                    layer = self.net_dense[i]
                    layer_prev = self.net_dense[i - 1]
                    self.make_hidden(layer, self.get_hidden(layer_prev))
            elif op == FLATTEN:
                self.ip += 1
                i = self.b_c_forward[self.ip]

                layer_conv = self.net_conv[self.sp_conv]
                inputs2fcn = layer_conv.out.flatten().tolist()
                layer = self.net_dense[i]
                layer.hidden = inputs2fcn
            elif op == CONV2D_OP:
                self.ip += 1
                i = self.b_c_forward[self.ip]

                if i == 0:
                    layer = self.net_conv[0]
                    self.convolution(layer, inputs)
                else:
                    layer = self.net_conv[i]
                    layer_prev = self.net_conv[i - 1]
                    self.make_hidden(layer, layer_prev.out)

            self.ip += 1

        self.ip = 0  # сбрасываем ip так прямое распространение будет в цикле

        last_layer = self.net_dense[self.sp_d]

        return self.get_hidden(last_layer)

    def backpropagate(self, y, x, l_r):
        len_b_c_bacward = len(self.b_c_bacward)

        while self.ip < len_b_c_bacward:
            op = self.b_c_bacward[self.ip]
            if op == DENSE:
                self.ip += 1
                i = self.b_c_bacward[self.ip]
                layer = self.net_dense[i]
                if i == self.sp_d:
                    self.calc_out_error(layer, y)
                else:
                    layer_next = self.net_dense[i + 1]
                    self.calc_hid_error(layer, layer_next)
            elif op == FLATTEN:
                self.ip += 1
                i = self.b_c_bacward[self.ip]
                layer = self.net_dense[i]
                layer_next = self.net_dense[i + 1]
                layer.in_ = layer_next.in_
                layer.errors = layer_next.errors

            self.ip += 1

        self.ip = 0

        while self.ip < len_b_c_bacward:
            op = self.b_c_bacward[self.ip]
            if op == DENSE:
                self.ip += 1
                i = self.b_c_bacward[self.ip]
                layer = self.net_dense[i]
                layer_prev = self.net_dense[i - 1]
                if i == 0:
                    self.upd_matrix(self.net_dense[i], self.net_dense[i].errors,
                                    x, l_r)
                else:
                    self.upd_matrix(layer, layer.errors,
                                    layer_prev.hidden, l_r)
            elif op == CONV2D_OP:
                layer: Conv2D = None
                self.ip += 1
                i = self.b_c_bacward[self.ip]
                if i == 0:
                    layer = self.net_conv[i]
                    layer_next = self.net_dense[0]  # Flatten слой
                    upd = np.array(layer_next.errors)
                    layer_next_in = layer_next.in_
                    upd_sliced = upd[: layer_next_in]
                    upd_reshaped = upd_sliced.reshape(
                        layer.f_height, layer.f_width)
                    layer_n_f = layer.n_f
                    layer_filt = layer.filt
                    for fil in range(layer_n_f):
                        layer_filt[fil] -= upd_reshaped

            self.ip += 1

        self.ip = 0


if __name__ == '__main__':
    def main():
        cnv: NetConConv = None
        cnv = NetConConv()
        x, x_test, y, y_test = get_train_test()
        x_0 = x[0].reshape(x[0].shape[0], x[0].shape[1], 1)
        # print('x', x[0])
        # print('y', y[0])
        cnv.cr_conv2d_lay((1, 5, 4, 1), 2, first_image_shape=(
            20, 18, 1), act_func=SIGMOID)
        cnv.cr_flatten()
        cnv.cr_dense(in_=64,  out_=2)
        print(cnv.feed_forwarding(x_0))

    #main()
    def main2():

        epochs = 100
        l_r = 0.1

        errors_y = []
        epochs_x = []

        train_inp, x_test, train_out, y_test = get_train_test(split_ratio=0.9) 
        # Создаем обьект параметров сети
      
        cnv = NetConConv()
        # Создаем слои
        cnv.cr_conv2d_lay((1, 5, 4, 1), 2, first_image_shape=(
            20, 18, 1), act_func=SIGMOID)
        cnv.cr_flatten()
        cnv.cr_dense(in_=64,  out_=3, act_func=SIGMOID)
        cnv.get_b_c_bacward()
        for ep in range(epochs):  # Кол-во повторений для обучения
            gl_e = 0
            for single_array_ind in range(len(train_inp)):

                inputs = train_inp[single_array_ind]
                inputs_reshaped=inputs.reshape(inputs.shape[0], inputs.shape[1], 1)
                output = cnv.feed_forwarding(inputs_reshaped)

                e = cnv.calc_diff(output, train_out[single_array_ind])

                gl_e += cnv.get_err(e)

                cnv.backpropagate(train_out[single_array_ind],
                                  train_inp[single_array_ind], l_r)

            # gl_e /= 2
            print("error", gl_e)
            print("ep", ep)
            print()

            errors_y.append(gl_e)
            epochs_x.append(ep)

            if gl_e == 0:
                break

        cnv.plot_gr('gr.png', errors_y, epochs_x)
        acc = net.evaluate(train_inp, train_out)
        print('acc', acc)

    main2()
