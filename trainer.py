import numpy as np
import yajl
from argparse import Namespace
from torch.nn import Module

from argparse import Namespace


class Trainer :
  stat_dtype = [('index', 'i4', (2,)),
                ('loss', 'f4'),
                ('accuracy', 'f4'),
                ('t_data', 'f4'),
                ('t_batch', 'f4')]

  var = None
  reporter = None
  lr_modify = None

  def __init__(self, data, network,
               loss, optimizer,
               reporter=BvrReporter,
               accuracy=BvrAccuracy, # TODO: Create nn.Module
               lr_adjuster=None,
               net_adjuster=None,
               var=None,
               mode='train',
               options=Namespace(
                 num_epochs=1,
                 report_frequency=100
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
    self.accuracy = accuracy()
    self.mode=mode
    self.var = var

    ## Class based Options 
    if reporter :
      self.reporter = reporter(self.stat_dtype)

    if lr_adjuster :
      self.lr_modify = lr_adjuster(self.optimizer, self.options)

    if net_adjuster :
      self.net_modify = net_adjuster(self.network, self.options)

  def is_eval_mode(self) :
    return self.mode == 'eval'

  def is_train_mode(self) :
    return self.mode == 'train'

  def to_var(self, X, vol=None) :
    if vol is None :
      vol = self.is_eval_mode()

    return torch.Autograd.Variable(
      X.cuda(async=True), volatile=vol
    )

  def stat_names(self) :
    return [stat[0] for stat in self.stat_dtype]

  def train(self) :
    opt = self.options
    i_max = len(self.data)

    stats = np.ndarray((opt.report_frequency,),
                       dtype=self.stat_dtype)

    stop_watch = StopWatch()
    stop_watch.start()

    for j in opt.num_epochs :
      for i, data in enumerate(self.data) :
        # adjust learning rate
        if self.lr_modify :
          self.lr_modify((j, i))

        # adjust layerwise training
        if self.net_modify :
          self.net_modify((j, i))

        train_1((j, i), data)

  def train_1(self, idx, data) :
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
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ## Batch Timer
    t_batch = stop_watch.record()

    ## Record Stats
    s_i = i % opt.report_frequency
    stats[s_i] = (
      idx,
      loss.data,
      torch.average(accuracy).data,
      t_data,
      t_batch
    )

    ## Report Stats
    ii = 1 + i
    if ii % opt.report_frequency == 0 or ii == i_max :
      reporter.report(stats[:ii])

if __name__ == "__main__" :
  import logging as lg
  lg.basicConfig(level=lg.DEBUG, format="%(levelname)-8s: %(message)s")
  
  
  pass
