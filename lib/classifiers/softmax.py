import numpy as np
from random import shuffle
from past.builtins import xrange
from math import log

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  num_train, dim = X.shape
  num_classes = W.shape[1]

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  scores = X.dot(W) #(N, C)
  scores -= np.max(scores, axis=1, keepdims=True)
  exp_scores = np.exp(scores)
  scores_normalized = exp_scores/exp_scores.dot(np.ones(num_classes))[..., np.newaxis]
  loss_i = -np.log(scores_normalized[xrange(num_train), y])
  loss += np.sum(loss_i) + 0.5*reg*np.sum(W**2)

  for i in xrange(num_train):
    scores_normalized[y[i]] -= 1
  dW = X.T.dot(scores_normalized) + reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  exp_scores = np.exp(scores)
  scores_normalized = exp_scores/exp_scores.dot(np.ones(num_classes))[..., np.newaxis]
  loss_i = -np.log(scores_normalized[xrange(num_train), y])
  loss += np.sum(loss_i)/num_train + 0.5*reg*np.sum(W**2)

  dMark = scores_normalized
  dMark[xrange(num_train), y] -= 1
  dW = X.T.dot(scores_normalized) + reg*W

  return loss, dW

