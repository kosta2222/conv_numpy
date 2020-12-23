from .evaluate_ import evaluate
from .learn import backpropagate, feed_forwarding, get_err, calc_diff, get_cros_entropy
from .nn_params_ import Nn_params
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


def fit(nn_params: Nn_params, X, Y, nb_epochs, l_r=0.07):    
    nb_epoch = 0 
    errs_l=[]
    eps_l=[]
    error = 0
    while nb_epoch < nb_epochs:
        error = 0
        print("ep:", nb_epoch)
        for retrive_ind in range(len(X)):
            x = X[retrive_ind]
            y = Y[retrive_ind]
            out_nn = feed_forwarding(nn_params, x)
            if nn_params.loss_func == MODIF_MSE:
                error+= get_err(calc_diff(out_nn, y))/2
                #print("err", error)
            elif nn_params.loss_func == CROS_ENTROPY:
                error = get_cros_entropy(out_nn, y, nn_params.outpu_neurons)
            backpropagate(nn_params, y, x, l_r)

        ac = evaluate(nn_params, X, Y)
        print('acc', ac)

        print('err', error)
            
        eps_l.append(nb_epoch)
        errs_l.append(error)
        nb_epoch += 1

        if error==0:
            break

    plot_gr('gr.png', errs_l, eps_l, 'test', None)
