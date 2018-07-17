import logging as lg
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

def log_average(id_range, stats) :

  # lg.info(stats.dtype.fields)
  # lg.info(stats.dtype.names)
  # lg.info(stats.dtype.itemsize)

  i0, i1 = id_range
  stats = stats[i0:i1]

  indices = [
    "%s:%s" % (stats[name][0], stats[name][i1-1 - i0])
    for name in stats.dtype.names
    if 'index' in name
  ]

  summary = [
    "%s:(%s %c %s)" % (name,
                       np.mean(stats[name]),
                       chr(177),
                       np.std(stats[name]))
    for name in stats.dtype.names
    if 'index' not in name
  ]

  lg.info("%s %s", ' '.join(indices), ' '.join(summary))


def grapher(id_range, stats) :
  pass

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
    self.cursor = i1

    for consume in self.queue :
      consume((i0, i1), self.stats)


if __name__ == '__main__' :
  import logging as lg
  lg.basicConfig(level=lg.DEBUG, format="%(levelname)-8s: %(message)s")
  
  
  pass
