import math
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from .serial_deserial import to_file, deserialization
import sys


TRESHOLD_FUNC = 0
TRESHOLD_FUNC_DERIV = 1
SIGMOID = 2
SIGMOID_DERIV = 3
RELU = 4
RELU_DERIV = 5
TAN = 6
TAN_DERIV = 7
INIT_W_MY = 8
INIT_W_RANDOM = 9
LEAKY_RELU = 10
LEAKY_RELU_DERIV = 11
INIT_W_CONST = 12
INIT_W_RANDN = 13
SOFTMAX = 14
SOFTMAX_DERIV = 15
PIECE_WISE_LINEAR = 16
PIECE_WISE_LINEAR_DERIV = 17
TRESHOLD_FUNC_HALF = 18
TRESHOLD_FUNC_HALF_DERIV = 19
MODIF_MSE = 20

NOOP = 0
ACC_GRAD = 1
APPLY_GRAD = 2
DENSE = 3



class Dense:
    def __init__(self):  # конструктор
        self.in_ = None  # количество входов слоя
        self.out_ = None  # количество выходов слоя
        self.matrix = [0] * 10  # матрица весов
        self.biases = [0] * 10
        self.cost_signals = [0] * 10  # вектор взвешенного состояния нейронов
        self.act_func = RELU
        self.hidden = [0] * 10  # вектор после функции активации
        self.errors = [0] * 10  # вектор ошибок слоя
        self.with_bias = False
        for row in range(10):  # создаем матрицу весов
            # подготовка матрицы весов,внутренняя матрица
            self.inner_m = list([0] * 10)
            self.matrix[row] = self.inner_m


################### Функции обучения ######################


class NetCon:
    def __init__(self, alpha_sigmoid=1, alpha_tan=1, beta_tan=1):
        self.net_dense = [None] * 2  # Двойной перпецетрон
        self.alpha_sigmoid = alpha_sigmoid
        self.alpha_tan = alpha_tan
        self.beta_tan = beta_tan
        for l_ind in range(2):
            self.net_dense[l_ind] = Dense()
        self.sp_d = -1  # алокатор для слоев
        self.nl_count = 0  # количество слоев
        self.b_c=[]
        self.ip=0
        self.ready = False

    def make_hidden(self, layer, inputs: list):
        # layer = self.net[layer_ind]
        for row in range(layer.out_):
            tmp_v = 0
            for elem in range(layer.in_):
                tmp_v += layer.matrix[row][elem] * inputs[elem]
            if layer.with_bias:
                tmp_v += layer.biases[row]
            layer.cost_signals[row] = tmp_v
            val = self.operations(layer.act_func, tmp_v)
            layer.hidden[row] = val

    def get_hidden(self, objLay: Dense):
        return objLay.hidden

    def feed_forwarding(self, inputs):
        while self.ip < len(self.b_c):
            op = self.b_c[self.ip]
            if op == DENSE:
                self.ip+=1
                arg = self.b_c[self.ip]
                if arg == 0:
                   layer = self.net_dense[0] 
                   self.make_hidden(layer, inputs)
                else:
                    layer = self.net_dense[arg]
                    layer_prev = self.net_dense[arg - 1]
                    self.make_hidden(layer, self.get_hidden(layer_prev))    
            self.ip+=1     

        """ j = self.nl_count
        for i in range(1, j):
            inputs = self.get_hidden(self.net_de[i - 1])
            self.make_hidden(i, inputs)
        """
        self.ip=0 # сбрасываем ip так прямое распространение будет в цикле
        
        j = self.nl_count 
        last_layer = self.net_dense[j-1] 

        return self.get_hidden(last_layer)

    def cr_dense(self,   in_=0, out_=0, act_func=None, with_bias=False, init_w=INIT_W_RANDOM):
        self.sp_d += 1
        layer = self.net_dense[self.sp_d]
        layer.in_ = in_
        layer.out_ = out_
        layer.act_func = act_func

        if with_bias:
            layer.with_bias = True
        else:
            layer.with_bias = False

        for row in range(out_):
            for elem in range(in_):
                layer.matrix[row][elem] = self.operations(
                    init_w, 0)
            if layer.with_bias:
                layer.biases[row] = self.operations(
                    init_w, 0)

        self.b_c.append(DENSE)
        self.b_c.append(self.sp_d)            
        self.nl_count += 1

    # Различные операции по числовому коду

    def operations(self, op, x):
        alpha_leaky_relu = 1.7159
        alpha_sigmoid = 2
        alpha_tan = 1.7159
        beta_tan = 2/3
        if op == RELU:
            if (x <= 0):
                return 0
            else:
                return x
        elif op == RELU_DERIV:
            if (x <= 0):
                return 0
            else:
                return 1
        elif op == TRESHOLD_FUNC:
            if (x > 0):
                return 1
            else:
                return 0
        elif op == TRESHOLD_FUNC_HALF:
            if x >= 1/2:
                return 1
            else:
                return 0
        elif op == TRESHOLD_FUNC_HALF_DERIV:
            return 1
        elif op == PIECE_WISE_LINEAR:
            if x >= 1/2:
                return 1
            elif x < 1/2 and x > -1/2:
                return x
            elif x <= -1/2:
                return 0
        elif op == PIECE_WISE_LINEAR_DERIV:
            return 1
        elif op == TRESHOLD_FUNC_DERIV:
            return 1
        elif op == LEAKY_RELU:
            if (x <= 0):
                return alpha_leaky_relu
            else:
                return 1
        elif op == LEAKY_RELU_DERIV:
            if (x <= 0):
                return alpha_leaky_relu
            else:
                return 1
        elif op == SIGMOID:
            y = 1 / (1 + math.exp(-self.alpha_sigmoid * x))
            return y
        elif op == SIGMOID_DERIV:
            return self.alpha_sigmoid * x * (1 - x)
        elif op == INIT_W_MY:
            if self.ready:
                self.ready = False
                return -0.567141530112327
            self.ready = True
            return 0.567141530112327
        elif op == INIT_W_RANDOM:

            return random.random()
        elif op == TAN:
            y = alpha_tan * math.tanh(beta_tan * x)
            return y
        elif op == TAN_DERIV:
            y = alpha_tan * math.tanh(beta_tan * x)
            return beta_tan / alpha_tan * (alpha_tan * alpha_tan - y * y)
        elif op == INIT_W_CONST:
            return 0.567141530112327
        elif op == INIT_W_RANDN:
            return np.random.randn()
        else:
            print("Op or function does not support ", op)

    def calc_out_error(self,  targets):
        layer = self.net_dense[self.nl_count-1]
        out_ = layer.out_
        for row in range(out_):
            layer.errors[row] =\
                (layer.hidden[row] - targets[row]) * self.operations(
                layer.act_func + 1, layer.hidden[row])

    def calc_hid_error(self,  layer_ind):
        layer = self.net_dense[layer_ind]
        layer_next = self.net_dense[layer_ind + 1]
        for elem in range(layer_next.in_):
            summ = 0
            for row in range(layer_next.out_):
                summ += layer_next.matrix[row][elem] * \
                    layer_next.errors[row]
            layer.errors[elem] = summ * self.operations(
                layer.act_func + 1, layer.hidden[elem])

    def upd_matrix(self, layer_ind, errors, inputs, lr):
        layer = self.net_dense[layer_ind]
        for elem in range(layer.in_):
            for row in range(layer.out_):
                error = errors[row]
                layer.matrix[row][elem] -= lr * \
                    error * inputs[elem]
                if layer.with_bias:
                   layer.biases[row] -= error * 1

    def calc_diff(self, out_nn, teacher_answ):
        diff = [0] * len(out_nn)
        for row in range(len(teacher_answ)):
            diff[row] = out_nn[row] - teacher_answ[row]
        return diff

    def get_err(self, diff):
        sum = 0
        for row in range(len(diff)):
            sum += diff[row] * diff[row]
        return sum

    def backpropagate(self, y, x, l_r):
        j = self.nl_count
        for i in range(j - 1, -1, - 1):
            if i == j - 1:
                self.calc_out_error(y)
            else:
                self.calc_hid_error(i)

        for i in range(j - 1, 0, - 1):
            layer = self.net_dense[i]
            layer_prev = self.net_dense[i - 1]
            self.upd_matrix(i, layer.errors, layer_prev.hidden, l_r)

        self.upd_matrix(0, self.net_dense[0].errors,
                        x, l_r)

    def answer_nn_direct(self, inputs):
        out_nn = self.feed_forwarding(inputs)
        return out_nn

    def evaluate(self, X_test, Y_test):
        """
         Оценка набора в процентах
         X_test: матрица обучающего набора X
         Y_test: матрица ответов Y
         return точность в процентах
        """
        scores = []
        res_acc = 0
        rows = len(X_test)
        wi_y_test = len(Y_test[0])
        elem_of_out_nn = 0
        elem_answer = 0
        is_vecs_are_equal = False
        for row in range(rows):
            x_test = X_test[row]
            y_test = Y_test[row]

            out_nn = self.answer_nn_direct(x_test)
            for elem in range(wi_y_test):
                elem_of_out_nn = out_nn[elem]
                elem_answer = y_test[elem]
                if elem_of_out_nn > 0.5:
                    elem_of_out_nn = 1
                    print("output vector elem -> ( %f ) " % 1, end=' ')
                    print("expected vector elem -> ( %f )" %
                          elem_answer, end=' ')
                else:
                    elem_of_out_nn = 0
                    print("output vector elem -> ( %f ) " % 0, end=' ')
                    print("expected vector elem -> ( %f )" %
                          elem_answer, end=' ')
                if elem_of_out_nn == elem_answer:
                    is_vecs_are_equal = True
                else:
                    is_vecs_are_equal = False
                    break
            if is_vecs_are_equal:
                print("-Vecs are equal-")
                scores.append(1)
            else:
                print("-Vecs are not equal-")
                scores.append(0)
        # print("in eval scores",scores)
        res_acc = sum(scores) / rows * 100

        return res_acc

    def plot_gr(self, f_name: str, errors: list, epochs: list) -> None:
        fig: plt.Figure = None
        ax: plt.Axes = None
        fig, ax = plt.subplots()
        ax.plot(epochs, errors,
                label="learning",
                )
        plt.xlabel('Эпоха обучения')
        plt.ylabel('loss')
        ax.legend()
        plt.savefig(f_name)
        print("Graphic saved")
        plt.show()

    def __str__(self):
        return str(self.b_c)    
#############################################


train_inp = ((1, 1), (0, 0), (0, 1), (1, 0))  # Логическое И
train_out = ([1], [0], [0], [0])

if __name__ == '__main__()':

    def main():
        epochs = 1000
        l_r = 0.1

        errors_y = []
        epochs_x = []

        # Создаем обьект параметров сети

        net = NetCon()
        # Создаем слои
        net.cr_lay(2, 3, PIECE_WISE_LINEAR, True, INIT_W_MY)
        net.cr_lay(3, 1, PIECE_WISE_LINEAR, True, INIT_W_MY)

        for ep in range(epochs):  # Кол-во повторений для обучения
            gl_e = 0
            for single_array_ind in range(len(train_inp)):

                inputs = train_inp[single_array_ind]
                output = net.feed_forwarding(inputs)

                e = net.calc_diff(output, train_out[single_array_ind])

                gl_e += net.get_err(e)

                net.backpropagate(train_out[single_array_ind],
                                  train_inp[single_array_ind], l_r)

            # gl_e /= 2
            print("error", gl_e)
            print("ep", ep)
            print()

            errors_y.append(gl_e)
            epochs_x.append(ep)

            if gl_e == 0:
                break

        plot_gr('gr.png', errors_y, epochs_x)

        # пост оценка - evaluate()
        for single_array_ind in range(len(train_inp)):
            inputs = train_inp[single_array_ind]

            output_2_layer = net.feed_forwarding(inputs)

            equal_flag = 0
            out_net = net.net[1].out_
            for row in range(out_net):
                elem_net = output_2_layer[row]
                elem_train_out = train_out[single_array_ind][row]
                if elem_net > 0.5:
                    elem_net = 1
                else:
                    elem_net = 0
                print("elem:", elem_net)
                print("elem tr out:", elem_train_out)
                if elem_net == elem_train_out:
                    equal_flag = 1
                else:
                    equal_flag = 0
                    break
            if equal_flag == 1:
                print('-vecs are equal-')
            else:
                print('-vecs are not equal-')

            print("========")

        # to_file(nn_params, nn_params.net, loger, 'wei1.my')

    main()
