#!/usr/bin/env python
# -*- coding: utf-8 -*-

import model
from network import cnnTextNetwork
from configurable import Configurable
import torch
import numpy as np
import os

if __name__=='__main__':

  import argparse

  argparser = argparse.ArgumentParser()
  argparser.add_argument('--train', action='store_true')
  argparser.add_argument('--validate', action='store_true')
  argparser.add_argument('--test', action='store_true')
  argparser.add_argument('--load', action='store_true')

  args, extra_args = argparser.parse_known_args()
  # args.train = True/False ...
  # extra_args['--some': "xxxx"]
  cargs = {k: v for (k, v) in vars(Configurable.argparser.parse_args(extra_args)).iteritems() if v is not None}

  if 'model_type' not in cargs:
    print("You need to specify the model_type")
    exit()
  print('*** '+cargs['model_type']+" ***")

  if args.load and 'restore_from' in cargs:
    print("Loading model from [%s]..." % (cargs['restore_from']))
    try:
      m = torch.load(cargs['restore_from'])
      cargs.pop(cargs['restore_from'], "")
    except:
      print("The model doesn't exist")
      exit()
  elif args.validate or args.test:
    pass
    "Restore from save_dir"
  else:
    m = getattr(model, cargs['model_type'])


  network = None

  if cargs['model_type'] == "CNNText":
    cargs.pop("model_type", "")
    network = cnnTextNetwork(args, m, **cargs)
  else:
    print("The model type is not supported")
    exit()

  if not os.path.exists(network.save_dir):
    os.mkdir(network.save_dir)

  torch.manual_seed(1)
  np.random.seed(1)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)



  if args.train:
    network.train()
  elif args.validate:
    network.test(validate=True)
  elif args.test:
    network.test(validate=False)