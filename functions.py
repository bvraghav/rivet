import logging as lg
lg.basicConfig(level=lg.DEBUG, format="%(levelname)-8s: %(message)s")

import numpy as np
import yajl
from argparse import Namespace
from torch.nn import Module
def load_options(options_file) :
  with open(options_file, 'r') as J :
    options = yajl.load(J)

  options = Namespace(**options)
  return options
class Identity(Module) :
  def forward(self, inputs) :
    return inputs


lg.info(Identity())


if __name__ == "__main__" :
  options_file = "/home/bvr/code/rivet/sample_options.json"
  options = load_options(options_file)
  
  lg.info(options)

  pass
