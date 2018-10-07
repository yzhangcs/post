# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def init_embedding(tensor):
    std = (1. / tensor.size(1)) ** 0.5
    nn.init.normal_(tensor, mean=0, std=std)
