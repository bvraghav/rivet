import numpy as np
import yajl
from argparse import Namespace
from torch.nn import Module
import logging as lg

import torch
import torch.nn.functional as F


class DistancePowerN(torch.nn.PairwiseDistance) :
  def __init__(self, power=2, **kwargs) :
    super().__init__(**kwargs)
    self.power = power
    # lg.info("DistancePowerN isinstance torch.nn.Module: %s",
    #         isinstance(self, torch.nn.Module))

  def forward(self, x1, x2) :
    d = super().forward(x1, x2)

    if self.power != 1 :
      d = torch.pow(d, self.power)

    return d

## Copied from this gist:
#  https://gist.github.com/harveyslash/725fcc68df112980328951b3426c0e0b

## Modified to include a distance measure

class ContrastiveLoss(torch.nn.Module):
  """
  Contrastive loss function.
  Based on:
  http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

  The distance measure is parametrised here. By default uses 
  squared euclidean distance. Another alternative may be 
  torch.nn.KLDivLoss - The KLDivergence as the distance metric.

  Labels are binary {0, 1}; 0 means similar, and 1 means dissimilar.

  """

  def __init__(self, margin=4.0,
               distance_fn=DistancePowerN(2)):
    super().__init__()

    # lg.info("ContrastiveLoss isinstance torch.nn.Module: %s", 
    #         isinstance(self, torch.nn.Module))

    self.distance_fn = distance_fn
    self.margin = margin

  def forward(self, outputs, label):
    # lg.info("len(outputs): %s", len(outputs))
    distance = self.distance_fn(*outputs)
    loss_contrastive = torch.mean(
      (1-label) * distance + 
      (label) * torch.clamp(self.margin - distance, min=0.0))

    return loss_contrastive

  def interpret(self, _Y) :
    d = self.distance_fn(*_Y)
    # lg.info("ContrastiveLoss.interpret: d.size(): %s", d.size())
    return (d > self.margin).float()

class TripletLoss(ContrastiveLoss) :
  '''
  A sum of Losses over each pair of positive and negative pairs

  The pair loss e.g. ContrastiveLoss
  '''

  def __init__(self, *args,
               label={
                 'pos': 0,
                 'neg': 1
               },
               **kwargs):
    super().__init__(*args, **kwargs)
    self.label=label

  def forward(self, output, label=None) :
    if label is None :
      y_pos, y_neg = self.label

    y_pos, y_neg = label
    _y, _y_pos, _y_neg = output

    return (super().forward((y, y_pos), y_pos)
            + super().forward((y, y_neg), y_neg))

def npy_var(x, volatile=False, cuda=True) :

  X = torch.from_numpy(label)
  if cuda :
    X = X.cuda()

  return torch.autograd.Variable(X)

class BCETripletLoss(torch.nn.BCELoss) :
  def __init__(self, *args,
               label=[
                 np.array([1, 0], dtype=np.float),
                 np.array([0, 1], dtype=np.float)
               ], **kwargs) :
    super().__init__(*args, **kwargs)
    self.label = [npy_var(l) for l in label]

  def forward(self, _Y, Y=None) :
    if Y is None:
      Y = self.label

    Y_pos, Y_neg = Y
    _Y_pos, _Y_neg = _Y

    return (super().forward(_Y_pos, Y) 
            + super().forward(_Y_neg, Y))


if __name__ == "__main__" :
  import logging as lg
  lg.basicConfig(level=lg.DEBUG, format="%(levelname)-8s: %(message)s")
  
  
  pass
