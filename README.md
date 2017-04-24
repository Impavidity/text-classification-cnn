# text-classification-cnn
Implementation for Convolutional Neural Networks for Sentence Classification of [Kim (2014)](https://arxiv.org/abs/1408.5882) with PyTorch.

## Project Structure

```
text-classification-cnn
  ├── config
  │   └── classification.cfg
  ├── etc
  │   ├── kmeans.py
  │   └── utils.py
  ├── model
  │   └── cnnText
  │       └── cnntext.py
  ├── network
  │   └── cnnTextNetwork.py
  ├── README.md
  ├── bucket.py
  ├── configurable.py
  ├── dataset.py
  ├── example.py
  ├── vocab.py
  └── main.py

```

## Model Type

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are randomly initialized -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.
- multichannel: A model with two sets of word vectors. Each set of vectors is treated as a 'channel' and each filter is applied to both channels, but gradients are back-propagated only through one of the channels. Hence the model is able to fine-tune one set of vectors while keeping the other static. Both channels are initialized with word2vec.



## Quick Start

To run the model on [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/) dataset on [rand](## Model Type)

```
python main.py --config_file config/classification.cfg --model_type CNNText --train
```
rand 92.16
static 88.94
non-static 92.62
multichannel 93.45

rand 42.59
static 46.33
non-static 44.32
multichannel 43.63

rand 82.20
static 86.42
non-static 85.43
multichannel 83.82
