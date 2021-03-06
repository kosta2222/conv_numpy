#NN_params.[py]
from .nn_constants import max_in_nn_1000,max_trainSet_rows,max_validSet_rows,max_rows_orOut_10,\
    max_am_layer,max_am_epoch,max_am_objMse,max_stack_matrEl,max_stack_otherOp_10,bc_bufLen, NOP, SIGMOID, MODIF_MSE
from .lay_ import  Dense
from .util import print_obj
# Параметры сети
class  Nn_params:
    def __init__(self):
        self.net=[]
        for i in range(max_am_layer):
            ob_lay=Dense()
            self.net.append(ob_lay)  # вектор слоев
        
        self.sp_d=-1
        self.input_neurons=0  # количество выходных нейронов
        self.outpu_neurons=0  # количество входных нейронов
        self.nl_count=0  # количество слоев
        self.inputs=[0]*(max_rows_orOut_10 )  # входа сети
        self.targets=[0]*(max_rows_orOut_10)  # ответы от учителя
        self.out_errors = [0] * (max_rows_orOut_10)  # вектор ошибок слоя
        self.loss_func=MODIF_MSE
        self.alpha_leaky_relu = 0.01
        self.alpha_sigmoid = 1
        self.alpha_tan = 1.7159
        self.beta_tan = 2 / 3

    #def __str__(self):
        # b_codes = ['x', 'RELU', 'x', 'SIGMOID', 'x', 'TRESHHOLD_FUNC', 'x', 'LEAKY_RELU', 'x', 'TAN']
        # func_s=b_codes[self.act_fu]
        # ind=b_codes.index(func_s)
        # act_fu=b_codes[ind]
        # info=f'with-adap-lr: {self.with_adap_lr}\nwith-bias: {self.with_bias}\n'+\
        #      f'act-fu: {act_fu}\n'+\
        #      f'alpha-leaky-relu: {self.alpha_leaky_relu} alpha-sigmoid: {self.alpha_sigmoid} alpha-tan: {self.alpha_tan} beta-tan: {self.beta_tan}\n'+\
        #      f'mse-treshold: {self.mse_treshold}'
        # return info
        #return print_obj('NN_params',self.__dict__)
