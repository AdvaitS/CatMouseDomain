import torch as tr
from nn import NeurNet


def lin_lin_0006():
    model = NeurNet(6, 16)
    model = tr.load('params_lin_lin_0006.txt')
    return model


def lin_lin_0001():
    model = NeurNet(6, 16)
    model = tr.load('params_lin_lin_0001.txt')
    return model


def lin_0005():
    model = tr.nn.Sequential(
        tr.nn.Flatten(),
        tr.nn.Linear(6 * 6 * 6, 1),
    )
    model = tr.load('params_lin_0005.txt')
    return model


def conv_lin_001():
    model = tr.nn.Sequential(
        tr.nn.Conv2d(6, 10, 3),
        tr.nn.ReLU(),
        tr.nn.Flatten(),
        tr.nn.Linear(160, 1),
    )
    model = tr.load('params_conv_lin_001.txt')
    return model
