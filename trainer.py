import numpy as np
import yajl
from argparse import Namespace
from torch.nn import Module
import torch
from argparse import Namespace
from reporter import BvrReporter
from functions import BvrAccuracy, StopWatch, BvrSaver


stat_dtype = [('index', 'i4', (2,)),
              ('loss', 'f4'),
              ('accuracy', 'f4'),
              ('t_data', 'f4'),
              ('t_batch', 'f4')]

class Trainer :

  var = None
  reporter = None
  saver = None

  def __init__(self, data, network,
               loss, optimizer,
               reporter=BvrReporter(stat_dtype),
               accuracy=BvrAccuracy(), # TODO: Create nn.Module
               saver=BvrSaver(),
               lr_adjuster=None,
               net_adjuster=None,
               var=None,
               mode='train',
               options=Namespace(
                 num_epochs=1,
                 report_frequency=100,
                 save_frequency=1,
                 cuda=True
               )) :
    '''
    To use a python dict for options, use options=Namespace(**py_dict)
    '''
    ## Mandatory
    self.data = data
    self.network = network
    self.loss = loss
    self.optimizer = optimizer

    ## Misc k:v pairs
    self.options = options

    ## Function / Value based Options
    self.accuracy = accuracy
    self.mode=mode
    self.var = var

    ## Class based Options 
    if reporter :
      self.reporter = reporter

    self.lr_modify = lr_adjuster
    self.net_modify = net_adjuster

    ## Stats
    self.stats = np.ndarray((self.options.report_frequency,),
                            dtype=stat_dtype)

    ## Saver
    if saver :
      self.saver = saver

  def is_eval_mode(self) :
    return self.mode == 'eval'

  def is_train_mode(self) :
    return self.mode == 'train'

  def to_var(self, X, vol=None, cuda=True) :
    if vol is None :
      vol = self.is_eval_mode()

    if isinstance(X, torch.Tensor) :
      if self.options.cuda :
        X = X.cuda(async=True)

      return torch.autograd.Variable(X)

    if isinstance(X, list) :
      return [self.to_var(x, vol) for x in X]

    raise TypeError("X in neither a Tensor nor a list of Tensors.")

  def stat_names(self) :
    return [stat[0] for stat in stat_dtype]

  def train(self) :
    opt = self.options

    for j in range(opt.num_epochs) :

      # adjust learning rate
      if self.lr_modify :
        self.lr_modify.step(j)

      # adjust layerwise training
      if self.net_modify :
        self.net_modify(j)

      stop_watch = StopWatch()
      stop_watch.start()

      for i, data in enumerate(self.data) :
        ii = self.train_1((j, i), data, stop_watch)

      if self.saver :
        i1 = self.reporter.cursor
        i0 = i1 - len(self.data)
        self.saver(self.network, self.reporter.stats, (i0, i1))

  def train_1(self, idx, data, stop_watch) :
    opt = self.options
    i_max = len(self.data)
    i = idx[-1]

    ## Create variables
    if self.var :
      Y, X = self.var(data, self.to_var, self.is_eval_mode())

    else :
      Y, X = data
      Y, X = self.to_var(Y, vol=True), self.to_var(X)

    ## Data Timer
    t_data = stop_watch.record()

    ## Forward Pass
    _Y = self.network(X)

    ## Loss and Accuracy
    loss = self.loss(_Y, Y)
    accuracy = self.accuracy(_Y, Y)

    ## Backward Pass
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    ## Batch Timer
    t_batch = t_data + stop_watch.record()

    ## Record Stats
    s_i = i % opt.report_frequency
    self.stats[s_i] = (
      idx,
      loss.data,
      accuracy.data,
      t_data,
      t_batch
    )

    ## Report Stats
    ii = 1 + i
    if ii % opt.report_frequency == 0 or ii == i_max :
      self.reporter.report(self.stats[:ii])


if __name__ == "__main__" :
  import logging as lg
  lg.basicConfig(level=lg.DEBUG, format="%(levelname)-8s: %(message)s")
  
  
  pass
