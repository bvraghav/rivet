import numpy as np


# stat_dtype = [('index', 'i4', (2,)),
#               ('loss', 'f4'),
#               ('accuracy', 'f4'),
#               ('t_data', 'f4'),
#               ('t_batch', 'f4')]

def is_int(n) :
  try:
    int(str(n))
  except ValueError:
    return False

  return True      

class BvrReporter(object) :
  stats = None
  chunk_size = 1024

  queue = []

  def __init__(self, stat_dtype,
               chunk_size=None,
               queue=(
                 log_average,
               )) :

    if is_int(chunk_size) :
      self.chunk_size = int(str(chunk_size))

    self.stats = np.ndarray((self.chunk_size,), dtype=stat_dtype)
    self.cursor = 0

    self.queue = queue

  def extend(self) :
    np.concatenate((self.stats, np.empty_like(self.stats)))

  def report(self, stats) :
    i0, i1 = self.cursor, self.cursor + stats.shape[0]
    if i1 > self.stats.shape[0] :
      self.extend()

    self.stats[i0:i1] = stats

    for consume in queue :
      consume((i0, i1), self.stats)

def log_average(id_range, stats) :
  i0, i1 = id_range
  stats = stats[i0:i1]

  stat_mean = np.mean(stats, axis=0)
  stat_std = np.std(stats, axis=0)

  lg.info(stat_mean)
  lg.info(stat_std)

def grapher(id_range, stats) :
  pass

if __name__ == '__main__' :
  import logging as lg
  lg.basicConfig(level=lg.DEBUG, format="%(levelname)-8s: %(message)s")
  
  
  pass
