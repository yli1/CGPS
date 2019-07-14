from simple_model import SimpleModel
from simple_model import LSTMModel
from simple_model import S2SModel
from simple_model import S2SAttentionModel
from extended_model import RandRegModel
from extended_model import NormalModel


def get_model(name, args):
    if name == 'simple':
        return SimpleModel(args)
    elif name == 'lstm':
        return LSTMModel(args)
    elif name == 's2s':
        return S2SModel(args)
    elif name == 's2s_att':
        return S2SAttentionModel(args)
    elif name == 'rand_reg':
        return RandRegModel(args)
    elif name == 'normal':
        return NormalModel(args)
    else:
        raise ValueError("Model name is not defined: " + name)
