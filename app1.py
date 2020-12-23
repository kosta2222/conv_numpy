#-*-coding: cp1251-*-
from brainy.net_con_ import NetCon, PIECE_WISE_LINEAR, INIT_W_MY


def main():

    train_inp = ((1, 1), (0, 0), (0, 1), (1, 0))  # Логическое И
    train_out = ([1], [0], [0], [0])

    epochs = 100
    l_r = 0.1

    errors_y = []
    epochs_x = []

    # Создаем обьект параметров сети

    net = NetCon()
    # Создаем слои
    net.cr_dense(2, 3, PIECE_WISE_LINEAR, True, INIT_W_MY)
    net.cr_dense(3, 1, PIECE_WISE_LINEAR, True, INIT_W_MY)
    print('net', net)

    for ep in range(epochs):  # Кол-во повторений для обучения
        gl_e = 0
        for single_array_ind in range(len(train_inp)):

            inputs = train_inp[single_array_ind]
            output = net.feed_forwarding(inputs)

            e = net.calc_diff(output, train_out[single_array_ind])

            gl_e += net.get_err(e)

            net.backpropagate(train_out[single_array_ind],
                              train_inp[single_array_ind], l_r)

        #gl_e /= 2
        print("error", gl_e)
        print("ep", ep)
        print()

        errors_y.append(gl_e)
        epochs_x.append(ep)

        if gl_e == 0:
            break

    net.plot_gr('gr.png', errors_y, epochs_x)
    acc=net.evaluate(train_inp, train_out)
    print('acc', acc)
    
    
main()