import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  errorCount = 0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[..., j] += X[i]
        errorCount += 1
  #np.sum(a, axis=1) <==> a.dot(ones(num_line))
  dW[..., y[i]] -= dW.dot(np.ones(num_classes))
  dW = dW/num_train + reg * W

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.

  W: (D, T)
  X: (N, D)
  y: (N, )
  margin:
  """
  loss = 0.0
  num_train, dim = X.shape
  num_classes = W.shape[1]
  dW = np.zeros(W.shape) # initialize the gradient as zero

  scores = X.dot(W) # (N, T)
  correct_class_scores = scores[range(num_train), y].reshape(num_train, -1)
  margin = scores - correct_class_scores + 1 
  margin[range(num_train), y] = 0
  margin_max = margin * (margin > 0)
  loss = np.sum(margin_max)/num_train + 0.5*reg*np.sum(W*W)


  margin_mark = (margin >0).astype(float, copy=False)
  margin_mark[range(num_train), y] -= margin_mark.dot(np.ones(num_classes))
  dW = X.T.dot(margin_mark)
  dW = dW/num_train + reg * W

  return loss, dW
