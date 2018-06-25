from PIL import Image
from torch.utils.data import Dataset, DataLoader

import random

from functools import reduce
import operator
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
    self.labels = labels

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
    indices = list(range(len(self.adjacency)))

    similar = sorted(self.adjacency[index])
    for i, ix in enumerate(similar) :
      indices.pop((ix - i))

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
    similar_pairs = yajl.load(J)
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
