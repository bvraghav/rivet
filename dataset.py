from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import random
from graph_algo import uniq_edges
import numpy as np

from functools import reduce
import operator
import yajl
from torchvision import transforms
class pairwise_dataset(Dataset) :
  '''Uses image_list and adjacency_list for similar pairs. For each
  image in similar_pair, randomly generates a dissimilar pair. (1:2)
  positive to negative samples.

  Labels may be initialized as an ordered pair: 0: similar, 1: dissimilar
  '''

  def __init__(self, adjacency, image_list,
               labels=[0, 1],
               transform = None,
               dissimilar_fn = None) :

    self.adjacency = adjacency
    self.image_list = image_list
    self.labels = torch.tensor(labels).float()

    self.transform = transform

    self.dissimilar = dissimilar_fn
    if self.dissimilar is None :
      self.dissimilar = self.find_dissimilar

    self.init_pairs()

  def init_pairs(self) :
    pairs = uniq_edges(self.adjacency) #gives me a numpy array (N, 2)
    flat_pairs = pairs.reshape([-1])
    undef = np.full_like(flat_pairs, -1)
    more_pairs = np.stack([flat_pairs, undef], axis=1)
    self.pairs = np.concatenate([pairs, more_pairs], axis=0)

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, index) :
    x1, x2 = self.pairs[index]
    y = int(x2 == -1)
    if y != 0 :
      x2 = self.dissimilar(x1)

    y = self.labels[y]

    # lg.info((x1, x2))
    x1 = self.load_image(x1)
    x2 = self.load_image(x2)

    if self.transform :
      x1 = self.transform(x1)
      x2 = self.transform(x2)

    return y, (x1, x2)

  def find_dissimilar(self, index) :
    indices = set((int(i) for i in self.adjacency.keys()))

    # lg.info("indices(%d): %s", len(list(indices)), sorted(list(indices)))
    # lg.info("index: %s", index)
    # lg.info("adjacency(%d): %s", index, self.adjacency[str(index)])

    indices = indices - set(self.adjacency[str(index)] + [int(index)])
    indices = list(indices)
    # lg.info("indices(%d): %s", len(indices)), sorted(indices))

    return random.choice(indices)

  def load_image(self, image_index) :
    image_name = self.image_list[image_index]
    return Image.open(image_name)

class triplet_dataset(pairwise_dataset) :
  '''Uses image_list and adjacency_list for similar pairs. For each
  image in similar_pair, randomly generates a dissimilar pair. (1:2)
  positive to negative samples.

  '''

  def __init__(self, *args, **kwargs) :
    super().__init__(*args, **kwargs)

  def init_pairs(self) :
    self.pairs = uniq_edges(self.adjacency) #gives me a numpy array (N, 2)

  def __len__(self):
    return 2 * self.pairs.shape[0]

  def __getitem__(self, index) :
    i = index // self.pairs.shape[0]
    index = index % self.pairs.shape[0]

    if i > 0:
      x_pos, x = self.pairs[index]
    else :
      x, x_pos = self.pairs[index]

    x_neg = self.dissimilar(x)

    x = self.load_image(x)
    x_pos = self.load_image(x_pos)
    x_neg = self.load_image(x_neg)

    if self.transform :
      x = self.transform(x)
      x_pos = self.transform(x_pos)
      x_neg = self.transform(x_neg)

    return self.labels, (x, x_pos, x_neg)


class Create(object) :
  def __init__(self, base_module) :
    self.base_module = base_module

  def __call__(self, adjacency, image_list, labels, transform):
    with open(adjacency, 'r') as J :
      adjacency = yajl.load(J)

    with open(image_list, 'r') as J :
      image_list = yajl.load(J)['image_list']

    transforms = {
      'sketch_transform': sketch_transform
    }
    transform = transforms.get(transform, sketch_transform)()

    return self.base_module(adjacency, image_list, labels, transform)
def flatten(inp_list) :
  return reduce(operator.concat, inp_list)
def sketch_transform() :
  return transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 255 - x)
  ])

if __name__ == '__main__' :
  # To Test
  # -----------------------------------
  # combinations_dataset(similar_pairs, image_list,
  #                      transform = None,
  #                      dissimilar_fn = None)
  
  # Logging:
  # -----------------------------------
  import logging as lg
  lg.basicConfig(level=lg.INFO, format='%(levelname)-8s: %(message)s')
  
  from graph_algo import edge_to_adjacency
  
  # With transforms
  # -----------------------------------
  from torchvision.transforms import Compose, Grayscale, ToTensor
  from torchvision.transforms import Resize, RandomCrop
  T = Compose([Grayscale(), Resize(224), RandomCrop(224), ToTensor()])
  
  ## Json Loader
  # -----------------------------------
  import yajl
  
  combinations_json = '/home/bvr/data/pytosine/k_nearest/20180526-153919/combinations.json'
  with open(combinations_json, 'r') as J :
    similar_pairs = yajl.load(J)['combinations']
  lg.info('Loaded similar pairs: size:%s', len(similar_pairs))
  
  adjacency = edge_to_adjacency(similar_pairs)
  # TODO: include edge_to_adjacency before tangle
  
  image_list_json = '/home/bvr/data/pytosine/k_nearest/20180521-205730/image_list.json'
  image_list_key = 'image_list'
  with open(image_list_json, 'r') as J :
    image_list = yajl.load(J)[image_list_key]
  lg.info('Loaded image_list: size:%s', len(image_list))
  
  def test_dataset(dataset_name) :
    global adjacency, image_list, T
  
    dataset = dataset_name(
      adjacency, image_list,
      transform = T,
      labels=[np.array([1, 0]), np.array([0, 1])])
  
    dataloader = DataLoader(
      dataset, shuffle=True, batch_size = 64
    )
  
    for i, (y, x) in enumerate(dataloader) :
      lg.info('sizes: len(y), y[0].size, len(x), x[0].size: %s, %s, %s, %s',
              len(y), y[0].size(), len(x), x[0].size())
  
  
  test_dataset(pairwise_dataset)
  test_dataset(triplet_dataset)
