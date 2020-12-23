import math
from .nn_constants import RELU, RELU_DERIV, INIT_W_MY, SIGMOID, SIGMOID_DERIV, TAN, TAN_DERIV,\
    SOFTMAX, CROS_ENTROPY, MODIF_MSE, INIT_RANDN, INIT_W_CONST, INIT_W_RANDOM
from .nn_params_ import Nn_params   # импортруем параметры сети
from .lay_ import Lay, Dense   # импортируем слой
from .work_with_arr import copy_vector
from .operations import operations, softmax_ret_vec
from .work_with_arr import copy_vector
import logging
import numpy as np


def make_hidden(nn_params, layer_ind, inputs: list):
    layer = nn_params.net[layer_ind]
    for row in range(layer.out):
        tmp_v = 0
        for elem in range(layer.in_):
            if layer.with_bias:
                if elem == 0:
                    tmp_v += layer.matrix[row][elem] * 1
                else:
                    tmp_v += layer.matrix[row][elem] * inputs[elem]

            else:
                tmp_v += layer.matrix[row][elem] * inputs[elem]

        layer.cost_signals[row] = tmp_v

        if layer.act_func != SOFTMAX:
            val = operations(layer.act_func, tmp_v, nn_params)
            layer.hidden[elem] = val

    if layer.act_func == SOFTMAX:
        ret_vec = softmax_ret_vec(layer.cost_signals, layer.out)
        copy_vector(ret_vec, layer.hidden, layer.out)


def get_hidden(objLay: Dense):
    return objLay.hidden


def feed_forwarding(nn_params: Nn_params, inputs):
    make_hidden(nn_params, 0, inputs)
    j = nn_params.nl_count
    for i in range(1, j):
        inputs = get_hidden(nn_params.net[i - 1])
        make_hidden(nn_params, i, inputs)

    last_layer = nn_params.net[j-1]

    return get_hidden(last_layer)


def cr_lay(nn_params: Nn_params, in_, out, act_func=None, with_bias=True, init_w=INIT_W_RANDOM):
    nn_params.sp_d += 1
    layer = nn_params.net[nn_params.sp_d]
    layer.in_ = in_
    layer.out = out
    layer.act_func = act_func

    if with_bias:
        layer.with_bias = True
    else:
        layer.with_bias = False

    if with_bias:
        in_ += 1
        layer.in_+=1
    for row in range(out):
        for elem in range(in_):
            layer.matrix[row][elem] = operations(
                init_w, 0, nn_params)

    nn_params.nl_count += 1
    return nn_params


def calc_out_error(nn_params, targets):
    # Последний слой
    layer = nn_params.net[nn_params.nl_count-1]
    out = layer.out
    if nn_params.loss_func == MODIF_MSE:
        for row in range(out):
            # накапливаем ошибку на выходе
            layer.errors[row] =\
                (layer.hidden[row] - targets[row]) * operations(
                layer.act_func + 1, layer.hidden[row], nn_params)
    elif nn_params.loss_func == CROS_ENTROPY and layer.act_func == SOFTMAX:
        # Для Softmax
        for row in range(out):
            tmp_v = layer.hidden[row] - targets[row]
            layer.errors[row] = tmp_v


def calc_hid_error(nn_params, layer_ind: int):
    layer = nn_params.net[layer_ind]
    layer_next = nn_params.net[layer_ind + 1]
    for elem in range(layer_next.in_):
        summ = 0
        for row in range(layer_next.out):
            summ += layer_next.matrix[row][elem] * layer_next.errors[row]
        layer.errors[elem] = summ * operations(
            layer.act_func + 1, layer.hidden[elem], nn_params)


# def upd_matrix(nn_params, layer_ind, errors, inputs, lr):
#     layer = nn_params.net[layer_ind]
#     func_point_l = [0] * layer.in_
#     for elem in range(layer.in_):
#         func_point = 0
#         for row in range(layer.out):
#             if layer.with_bias:
#                 if elem == 0:
#                     func_point += layer.matrix[row][elem] * lr * \
#                         errors[row] * 1
#                 else:
#                     func_point += layer.matrix[row][elem] * lr * \
#                         errors[row] * inputs[elem]
#             else:
#                 func_point += layer.matrix[row][elem] * lr * \
#                     errors[row] * inputs[elem]
#         func_point_l[elem] = func_point

   # or row in range(layer.out):
    #    for elem in range(layer.in_):
     #       layer.matrix[row][elem] -= func_point_l[elem]

def upd_matrix(nn_params, layer_ind, errors, inputs, lr):
    layer = nn_params.net[layer_ind]
    for row in range(layer.out):
        error = errors[row]
        for elem in range(layer.in_):
            if layer.with_bias:
                if elem == 0:
                    layer.matrix[row][elem] -= lr * \
                        error * 1
                else:
                    layer.matrix[row][elem] -= lr * \
                        error * inputs[elem]
            else:
                layer.matrix[row][elem] -= lr * \
                    error * inputs[elem]


def calc_diff(out_nn, teacher_answ):
    diff = [0] * len(out_nn)
    for row in range(len(teacher_answ)):
        diff[row] = out_nn[row] - teacher_answ[row]
    return diff


def get_err(diff):
    sum = 0
    for row in range(len(diff)):
        sum += diff[row] * diff[row]
    return sum


def get_cros_entropy(ans, targ, n):
    E = 0
    for row in range(n):
        E += -(targ[row] *
               math.log(ans[row]))
    return E


def answer_nn_direct(nn_params: Nn_params, inputs, loger=None):
    out_nn = feed_forwarding(nn_params, inputs)
    return out_nn


def backpropagate(nn_params, y, x, l_r, loger=None):
    j = nn_params.nl_count
    for i in range(j - 1, -1, - 1):
        if i == j - 1:
            calc_out_error(nn_params, y)
        else:
            calc_hid_error(nn_params, i)

    for i in range(j - 1, 0, - 1):
        layer = nn_params.net[i]
        layer_prev = nn_params.net[i + 1]
        upd_matrix(nn_params, i, layer.errors, layer_prev.hidden, l_r)

    upd_matrix(nn_params, 0, nn_params.net[0].errors,
               x, l_r)


############################
