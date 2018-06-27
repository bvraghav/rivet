import numpy as np
# import copy

def uniq_edges(adjacency) :
  adj_view = adjacency
  if isinstance(adj_view, list) :
    adj_view = {
      i: list(l) for (i, l) in enumerate(adj_view)
    }
  elif isinstance(adj_view, dict) :
    adj_view = {
      int(k): list(adj_view[k]) for k in adj_view
    }
  else :
    raise TypeError("Expected list or dict, but found %s for `adjacency'" % type(adjacency))

  mask = [list(l) for l in adj_view.values()]
  N = np.sum([len(l) for l in mask]) / 2
  N = int(N)
  edges = np.zeros([N, 2], dtype=np.int)

  ei = 0
  for i, l in adj_view.items() :
    l = list(l)
    for j in l :
      # Remove i from mask
      lj = adj_view[j]
      ji = lj.index(i)
      lj.pop(ji)

      # Append to edges
      edges[ei] = [i, j]
      ei += 1

  return edges

def edge_to_adjacency(edges) :
  edges = np.array(edges).tolist()
  # lg.info((edges.shape, edges.dtype))
  adjacency = {}

  for i, j in edges :
    # lg.info((i, j))
    if i in adjacency and j in adjacency[i] :
      continue

    li = ([j]
          if str(i) not in adjacency
          else adjacency[str(i)] + [j])
    lj = ([i]
          if str(j) not in adjacency
          else adjacency[str(j)] + [i])

    adjacency.update({
      str(i): li,
      str(j): lj
    })

  return adjacency

if __name__ == "__main__" :
  import logging as lg
  lg.basicConfig(level=lg.DEBUG, format='%(levelname)-8s: %(message)s')
  # Creating a simple adjacency list for a diamond on string:
  #         *
  #        / \
  #    *--*   *--*
  #        \ /
  #         *
  #
  adjacency = [
    (1,),
    (0, 2, 3),
    (1, 4),
    (1, 4),
    (2, 3, 5),
    (4,)
  ]
  
  adjacency
  edges = uniq_edges(adjacency)
  lg.info((edges.shape, edges.dtype))
  lg.info(edges)
  adj = {'0': [1], '1': [0, 2, 3], '2': [1, 4], '3': [1, 4], '4': [2, 3, 5], '5': [4]}
  edg = uniq_edges(adj)
  lg.info((edg.shape, edg.dtype))
  lg.info(edg)
  edges = [[0,1],
           [1,2],
           [1,3],
           [2,4],
           [3,4],
           [4,5]]
  adj = edge_to_adjacency(edges)
  lg.info(adj)
  pass
