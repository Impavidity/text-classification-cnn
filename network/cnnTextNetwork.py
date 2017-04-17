#!/usr/bin/env python
# -*- coding: utf-8 -*-

from configurable import Configurable
from vocab import Vocab
from dataset import Dataset
import os
import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class cnnTextNetwork(Configurable):
  """
  Network class
  - build the vocabulary
  - build the dataset
  - control the training
  - control the validation and testing
  - save the model and store the best result
  """

  def __init__(self, option, model, *args, **cargs):

    '''check args?'''
    super(cnnTextNetwork, self).__init__(*args, **cargs)
    if not os.path.isdir(self.save_dir):
      os.mkdir(self.save_dir)

    with open(os.path.join(self.save_dir, 'config_file'), 'w') as f:
      self._config.write(f)


    self._vocabs = []
    vocab_file = [(self.word_file, 'Words'),
                  (self.target_file, "Targets")]

    for i, (vocab_file, name) in enumerate(vocab_file):
      vocab = Vocab(vocab_file, self._config,
                    name = name,
                    #load_embed_file = (not i),
                    load_embed_file= False,
                    lower_case = (not i)
                    )
      self._vocabs.append(vocab)

    print("################## Data ##################")
    print("There are %d words in training set" % (len(self.words) - 2))
    print("There are %d targets in training set" % (len(self.targets) - 2))
    print("Loading training set ...")
    self._trainset = Dataset(self.train_file, self._vocabs, self._config, name="Trainset")
    print("There are %d sentences in training set" % (self._trainset.sentsNum))
    print("Loading validation set ...")
    self._validset = Dataset(self.valid_file, self._vocabs, self._config, name="Validset")
    print("There are %d sentences in validation set" % (self._validset.sentsNum))
    print("Loading testing set ...")
    self._testset =  Dataset(self.test_file, self._vocabs, self._config, name="Testset")
    print("There are %d sentences in testing set" % (self._testset.sentsNum))

    self.args = {'input_channels':1,
                 'kernel_sizes':[3,4,5],
                 'embed_num': len(self.words),
                 'embed_dim':100,
                 'target_class': len(self.targets),
                 'output_channels': 4,
                 'dropout': 0.9}
    self.model = model(self.args)
    return

  def train_minibatch(self):
    return self._trainset.minibatch(self.train_batch_size, self.input_idx, self.target_idx, shuffle=True)

  def valid_minibatch(self):
    return self._validset.minibatch(self.test_batch_size, self.input_idx, self.target_idx, shuffle=False)

  def test_minibatch(self):
    return self._testset.minibatch(self.test_batch_size, self.input_idx, self.target_idx, shuffle=False)

  def train(self):
    if torch.cuda.is_available(): # and use_cuda
      self.model.cuda()

    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    # The optimizer doesn't have adaptive learning rate

    step = 0
    best_score = 0
    valid_accuracy = 0
    test_accuracy = 0

    acc_corrects = 0 # count the corrects for one log_interval
    acc_sents = 0 # count sents number for one log_interval

    while True:
      for batch in self.train_minibatch():
        feature, target = batch['text'], batch['label']
        feature = Variable(torch.from_numpy(feature))
        target = Variable(torch.from_numpy(target))[:,0]
        optimizer.zero_grad() # Clears the gradients of all optimized Variable
        logit = self.model(feature)
        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()
        step += 1
        preds = torch.max(logit, 1)[1].view(target.size())  # get the index
        acc_corrects += preds.eq(target).cpu().sum()
        acc_sents += batch['batch_size']
        if step % self.log_interval == 0:
          accuracy = acc_corrects.data.numpy() / float(acc_sents) * 100.0
          print("## [Batch %d] Accuracy : %5.2f" % (step, accuracy))

        if step == 1 or step % self.valid_interval == 0:
          accuracy = self.test(validate=True)
          print("## Validation: %5.2f" % (accuracy))
          if accuracy > best_score:
            best_score = accuracy
            valid_accuracy = accuracy
            print("## Update Model ##")
            torch.save(self.model, self.save_model_file)
            print("## Testing ##")
            test_accuracy = self.test(validate=False)
            print("## Testing: %5.2f" % (test_accuracy))
          print("## Currently the best validation: Accucacy %5.2f" % (valid_accuracy))
          print("## Currently the best testing: Accuracy %5.2f" % (test_accuracy))


  def test(self, validate=False):
    if validate:
      dataset = self._validset
      minibatch = self.valid_minibatch
    else:
      dataset = self._testset
      minibatch = self.test_minibatch

    test_corrects = 0
    test_sents = 0
    for batch in minibatch():
      # TODO: Prediton to Text
      feature, target = batch['text'], batch['label']
      feature = Variable(torch.from_numpy(feature))
      target = Variable(torch.from_numpy(target))[:,0]


      logit = self.model(feature)
      preds = torch.max(logit, 1)[1].view(target.size())  # get the index
      test_corrects += preds.eq(target).cpu().sum()
      test_sents += batch['batch_size']
    return test_corrects.data.numpy() / float(test_sents) * 100.0

  @property
  def words(self):
    return self._vocabs[0]

  @property
  def targets(self):
    return self._vocabs[1]


  @property
  def input_idx(self):
    return (0,)


  @property
  def target_idx(self):
    return (0,)


