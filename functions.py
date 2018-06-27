import numpy as np
import yajl
from argparse import Namespace
from torch.nn import Module
from torch import nn
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
  pretrained = weigths['state_dict']
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


if __name__ == "__main__" :
  import logging as lg
  lg.basicConfig(level=lg.DEBUG, format="%(levelname)-8s: %(message)s")
  
  options_file = "/home/bvr/code/rivet/sample_options.json"
  options = load_options(options_file)
  
  lg.info(options)
  lg.info(Identity())
  

  pass
