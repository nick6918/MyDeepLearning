# As usual, a bit of setup
from __future__ import print_function
import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from lib.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from lib.layers.rnn_layers import *
from lib.solvers.captioning_solver import CaptioningSolver
from lib.classifiers.rnn import CaptioningRNN
from lib.utils.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from lib.utils.image_utils import image_from_url

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

#Two type of data to use
data = load_coco_data(pca_features=True)

med_data = load_coco_data(max_train=1000)

for k, v in med_data.items():
    if type(v) == np.ndarray:
        print(k, type(v), v.shape, v.dtype)
    else:
        print(k, type(v), len(v))

# U can use either lstm or vanilla rnn
med_lstm_model = CaptioningRNN(
          cell_type='lstm',
          word_to_idx=med_data['word_to_idx'],
          input_dim=med_data['train_features'].shape[1],
          hidden_dim=512,
          wordvec_dim=256,
          dtype=np.float32,
        )

med_lstm_solver = CaptioningSolver(med_lstm_model, med_data,
           update_rule='adam',
           num_epochs=50,
           batch_size=50,
           optim_config={
             'learning_rate': 5e-3,
           },
           lr_decay=0.995,
           verbose=True, print_every=10,
         )

#achieved (Epoch 22 / 50) train acc: 0.992441; val_acc: 0.226580
#Overfitting, try to feed more data and increase regularization to improve accuracy
med_lstm_solver.train()

# Plot the training losses
plt.plot(med_lstm_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()

for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(med_data, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = med_lstm_model.sample(features)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()
