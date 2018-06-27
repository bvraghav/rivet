import logging as lg
lg.basicConfig(level=lg.DEBUG, format="%(levelname)-8s: %(message)s")

from functions import load_options, pretrained_resnet, create_fc
from argparse import ArgumentParser as Parser

import networks
import losses
import reporter
import trainer

## Define parser
parser = Parser()
parser.add_argument('options_file')
args = parser.parser_args()

## Load options
options = load_options(args.options_file)

lg.info(options)

## Data
datasets = {
  "pairwise" : dataset.create(dataset.pairwise_dataset),
  "triplet"  : dataset.create(dataset.triplet_dataset)
}
dataset = datasets.get(
  options.dataset, dataset.pairwise_dataset
)(**options.dataset_params)

dataloader =  torch.utils.data.DataLoader(
  dataset,
  **options.dataloader_params
)

## Pretrained network 
# TODO: Refactor to generalize
resnet = lambda: pretrained_resnet(
  options.weights_file
)
network = {
  "resnet18": resnet
}.get(options.network, resnet)()

## Create Model Pair/Triplet
models = {
  "feature_pair"   : networks.pair_feat,
  "feature_triple" : networks.triple_feat,
  "concat_pair"    : networks.pair_concat
  "concat_triple"  : networks.triple_concat
}
model = models.get(
  options.model, feature_pair_model
)(network, **options.model_params)

## Create Optimizer
optimizers = {
  "Adadelta",   : torch.optim.Adadelta,
  "Adagrad",    : torch.optim.Adagrad,
  "Adam",       : torch.optim.Adam,
  "SparseAdam", : torch.optim.SparseAdam,
  "Adamax",     : torch.optim.Adamax,
  "ASGD",       : torch.optim.ASGD,
  "LBFGS",      : torch.optim.LBFGS,
  "RMSprop",    : torch.optim.RMSprop,
  "Rprop",      : torch.optim.Rprop,
  "SGD"         : torch.optim.SGD
}
optimizer = optimizers.get(
  options.optimizer, torch.optim.SGD
)(model.parameters(), **options.optimizer_params)

## LR_Adjuster
lr_adjusters = {
  "LambdaLR"          : torch.optim.lr_scheduler.LambdaLR,,
  "StepLR"            : torch.optim.lr_scheduler.StepLR,,
  "MultiStepLR"       : torch.optim.lr_scheduler.MultiStepLR,,
  "ExponentialLR"     : torch.optim.lr_scheduler.ExponentialLR,,
  "CosineAnnealingLR" : torch.optim.lr_scheduler.CosineAnnealingLR,,
  "ReduceLROnPlateau" : torch.optim.lr_scheduler.ReduceLROnPlateau,
}
lr_adjuster = lr_adjusters.get(
  options.lr_adjuster, torch.optim.ExponentialLR
)(optimizer, **options.lr_adjuster_params)

## Criterion
distance_fn = {
  "euclidean" : losses.DistancePowerN,
  "kldiv"     : torch.nn.KLDivLoss
}.get(
  options.criterion_params.get(
    "distance", "euclidean"
))
criteria = {
  "contrastive" : lambda: losses.ContrastiveLoss(
    distance_fn = distance_fn
  ),
  "triplet"     : lambda: losses.TripletLoss(
    distance_fn = distance_fn
  ),
  "bce"         : torch.nn.BCELoss,
  "bce_triplet" : losses.BCETripletLoss
}
criterion = criteria.get(
  options.criterion, torch.nn.BCELoss
)()

## Reporting
reportrs = {
  "log_average": reporter.log_average,
  "grapher": reporter.grapher
}
reportr = reporter.BvrReporter(queue=[
  reportrs[r] for r in options.reporters
])

## Trainer
trainr = trainer.Trainer(
  data = dataloader,
  network = model,
  loss = criterion,
  optimizer = optimizer,
  reporter = reportr,
  lr_adjuster = lr_adjuster
)
trainr.train()
