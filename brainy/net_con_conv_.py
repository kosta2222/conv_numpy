from .net_con_ import NetCon

class Conv2D:
    def __init__(self):
        self.prefix='conv2d'
        self.f=0 # сторона фильтра
        self.s=0 # шаг


class NetConConv(NetCon):
    def __init__(self):
        pass

    def cr_conv2d_lay(self, f, s, image_shape):
        pass