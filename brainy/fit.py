from .evaluate_ import evaluate
from .learn import backpropagate, get_mse, feed_forwarding, get_mean, get_err, calc_diff, get_cros_entropy
from .NN_params import Nn_params
from .nn_constants import MODIF_MSE, CROS_ENTROPY
import matplotlib.pyplot as plt


def plot_gr(_file: str, errors: list, epochs: list, name_gr: str, logger) -> None:
    fig: plt.Figure = None
    ax: plt.Axes = None
    fig, ax = plt.subplots()
    # plt.text(0.1, 1.1, name_gr)
    ax.plot(epochs, errors,
            label="learning",
            )
    plt.xlabel('Эпоха обучения')
    plt.ylabel('loss')
    ax.legend()
    plt.savefig(_file)
    print("Graphic saved")
    # logger.info("Graphic saved")
    plt.show()


def fit(nn_params: Nn_params, X, Y, X_test, Y_test,with_run_all_eps, eps, with_adap_lr, l_r_, with_loss_threshold, mse_, with_acc_shureness, acc_, loger):
    alpha = 0.99
    beta = 1.01
    gama = 1.01
    error_pr = 0
    delta_error = 0
    l_r = l_r_
    net_is_running = True
    it = 0
    exit_flag = False
    eps_l = []
    errs_l = []
    acc = 0.0

    while net_is_running:
        print("ep:", it)
        error = 0
        for retrive_ind in range(len(X)):
            x = X[retrive_ind]
            y = Y[retrive_ind]
            out_nn = feed_forwarding(nn_params, x, loger)
            if nn_params.loss_func == MODIF_MSE:
                error += get_err(calc_diff(out_nn, y, nn_params.outpu_neurons))
            elif nn_params.loss_func == CROS_ENTROPY:
                error = get_cros_entropy(out_nn, y, nn_params.outpu_neurons)
            backpropagate(nn_params, out_nn, y, x, l_r_, loger)

            ac = evaluate(nn_params, X_test, Y_test, loger)
            if with_acc_shureness:
                if int(acc - acc_) == 0:
                    print("op!")
                    break
        print("acc", ac)

        if with_adap_lr:
            delta_error = error - gama * error_pr
            if delta_error > 0:
                l_r = alpha * l_r
            else:
                l_r = beta * l_r
            error_pr = error

        if exit_flag:
            break
        print("err", error)
        eps_l.append(it)
        errs_l.append(error)
        it += 1
        if with_loss_threshold:
            if nn_params.loss_func == MODIF_MSE:
                if error <= mse_:
                    break
            elif nn_params.loss_func == CROS_ENTROPY:
                if error < 0 and error < mse_:
                    break

        elif with_run_all_eps:
            if it == eps:
                break

    plot_gr('gr.png', errs_l, eps_l, 'test', None)
