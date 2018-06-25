import logging as lg
lg.basicConfig(level=lg.DEBUG, format="%(levelname)-8s: %(message)s")

import numpy as np
import yajl
from argparse import Namespace
from torch.nn import Module
import torch
from torch import nn

class pair_feat(nn.Module) :
  def __init__(self, network) :
    super().__init__()
    self.network = network

  def forward(inputs) :
    x1, x2 = inputs
    return self.network(x1), self.network(x2)

class triple_feat(nn.Module) :
  def __init__(self, network) :
    super().__init__()
    self.network = network

  def forward(inputs) :
    x, x_pos, x_neg = inputs
    return self.network(x), self.network(x_pos), self.network(x_neg)

class pair_xent(nn.Module) :
  def __init__(self, network, fc,
               feat_length=512) :
    self.network = network
    fc_out = fc
    fc_in = [feat_length] + fc_out[:-1]
    self.fc = nn.Sequential(*[
      nn.Linear(in_size, out_size)
      for in_size, out_size in zip(fc_in, fc_out)
    ])

  def forward(inputs) :
    x1, x2 = inputs
    y1, y2 = self.network(x1), self.network(x2)
    new_x = torch.concat(y1, y2)
    return self.fc(new_x)
class triple_concat(nn.Module) :
  def __init__(self, *args, **kwargs) :
    super().__init__()
    self.network = pair_concat(*args, **kwargs)

  def forward(inputs) :
    x, x_pos, x_neg = inputs
    y_pos_hat = self.network((x, x_pos))
    y_neg_hat = self.network((x, x_neg))

    return (y_pos_hat, y_neg_hat)


if __name__ == "__main__" :
  
  pass
