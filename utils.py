# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def init_embedding(tensor):
    bias = (3. / tensor.size(1)) ** 0.5
    nn.init.uniform_(tensor, -bias, bias)
