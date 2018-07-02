import numpy as np
import yajl
from argparse import Namespace
from torch.nn import Module
import torch
import torchvision.models as models
from torch import nn
import logging as lg
import time
import os
import shutil
from datetime import datetime as Dt

def load_options(options_file) :
  with open(options_file, 'r') as J :
    options = yajl.load(J)

  options = Namespace(**options)
  return options


class Identity(Module) :
  def forward(self, inputs) :
    return inputs

def pretrained_resnet(weights_file, fc=Identity()) :
  weights = torch.load(weights_file)
  pretrained = weights['state_dict']
  pretrained = {k: pretrained[k]
                  for k in pretrained
                  if 'fc' not in k}

  resnet = models.resnet18()
  resnet.load_state_dict(pretrained, strict=False)
  resnet.fc = fc
  resnet.eval()
  resnet.cuda()

  return resnet

def create_fc(layers=[128, 1]) :
  if layers is None :
    return Identity()

  fc_out = layers
  fc_in = [512] + fc_out[:-1]

  return nn.Sequential(*[
    nn.Linear(in_size, out_size)
    for (in_size, out_size) in zip(fc_in, fc_out)
  ])

class BvrAccuracy(Module) :
  def __init__(self, transform=None):
    super().__init__()
    self.transform = transform
    # lg.info("accuracy: transform: %s", self.transform)

  def forward(self, _Y, Y) :
    # lg.info("size: _Y: %s", _Y.size())
    # lg.info("size: Y: %s", Y.size())

    if self.transform :
      # lg.info("callable(self.transform): %s", 
      #   callable(self.transform))
      # lg.info("transforming _Y")
      _Y = self.transform(_Y)

    return torch.mean((_Y == Y).float())

class StopWatch(object) :
  def __init__(self):
    self.start()

  def start(self) :
    self.time = time.time()

  def record(self) :
    old = self.time
    self.time = time.time()
    return self.time - old
class BvrSaver(object) :
  options = Namespace(
      save_frequency = 1,
      save_location = '.',
      saver_current = 'checkpoint.pth.tar',
      saver_best = 'model_best.pth.tar'
  )

  def __init__(self, options=dict()) :
    self.count = 0
    self.best_prec = 0.

    self.update_options(options)

  def update_options(self, options) :

    if isinstance(options, Namespace) :
      options = vars(options)
    self.options = vars(self.options)

    # lg.info(self.options)
    # lg.info(options)
    self.options.update(options)
    # lg.info(self.options)

    self.options = Namespace(**self.options)

    self.update_locations()

  def update_locations(self) :
    self.options.save_location = os.path.join(
      self.options.save_location,
      BvrSaver.timestamp()
    )

    self.options.saver_current = os.path.join(
      self.options.save_location,
      self.options.saver_current
    )

    self.options.saver_best = os.path.join(
      self.options.save_location,
      self.options.saver_best
    )

  @staticmethod
  def timestamp() :
    return Dt.now().strftime('%Y%m%d-%H%M%S')

  def __call__(self, model, stats, idx_range) :
    self.count += 1
    if self.count < self.options.save_frequency :
      return

    if not os.path.isdir(self.options.save_location) :
      os.makedirs(self.options.save_location)

    lg.info("Saving Model")
    torch.save(model, self.options.saver_current)

    i0, i1 = idx_range
    prec = np.mean(stats[i0:i1]['accuracy'])

    lg.info("idx_range, prec, best_prec: %s, %s, %s", idx_range, prec, self.best_prec)
    if prec > self.best_prec :
      self.count = 0
      self.best_prec = prec
      shutil.copyfile(self.options.saver_current,
                      self.options.saver_best)
      lg.info("Saving best model")


if __name__ == "__main__" :
  import logging as lg
  lg.basicConfig(level=lg.DEBUG, format="%(levelname)-8s: %(message)s")
  
  options_file = "/home/bvr/code/rivet/sample_options.json"
  options = load_options(options_file)
  
  lg.info(options)
  lg.info(Identity())
  

  pass
