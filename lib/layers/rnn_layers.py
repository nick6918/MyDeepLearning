from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
    cache = Wx, x, Wh, prev_h, next_h
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    Wx, x, Wh, prev_h, next_h = cache
    N, H = dnext_h.shape
    dLoss_dH = (1 - next_h**2) * dnext_h #(N, H)
    dx = dLoss_dH.dot(Wx.T)
    dWx = x.T.dot(dLoss_dH)
    dprev_h = dLoss_dH.dot(Wh.T)
    dWh = prev_h.T.dot(dLoss_dH)
    db = np.ones(N).dot(dLoss_dH)
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    N, T, D = x.shape
    N, H = h0.shape
    h = np.zeros((N, T, H))
    prev_h = h0
    for i in range(T):
        current_x = x[:, i, :] #(N, D)
        prev_h, _ = rnn_step_forward(current_x, prev_h, Wx, Wh, b)
        h[:, i, :] = prev_h
    cache = Wx, x, Wh, h, h0

    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    Wx, x, Wh, h, h0 = cache
    N, T, H = dh.shape
    dx = np.zeros_like(x)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros(H)
    dprev_h = 0
    for i in range(T-1, -1, -1):
        next_h = h[:, i, :]
        if i > 0:
            prev_h = h[:, i-1, :]
        else:
            prev_h = h0
        dnext_h = dh[:, i, :] + dprev_h
        current_cache = Wx, x[:, i, :], Wh, prev_h, next_h
        dx_i, dprev_h, dWx_i, dWh_i, db_i = rnn_step_backward(dnext_h, current_cache)
        dWx += dWx_i
        dWh += dWh_i
        db += db_i
        dx[:, i, :] = dx_i
    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out = W[x, :]
    cache = x, W

    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)

    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """

    N, H = prev_h.shape
    middle_state = prev_h.dot(Wh) + x.dot(Wx) + b       #(N, 4H)
    i0, f0, o0, g0 = middle_state.T.reshape(4, H, -1)       #(H, N)
    i = sigmoid(i0.T)
    f = sigmoid(f0.T)
    g = np.tanh(g0.T)
    o = sigmoid(o0.T)
    next_c = prev_c * f + i * g
    next_h = o * np.tanh(next_c)
    cache = f, g, i, o, prev_h, prev_c, next_c, x, Wh, Wx

    return next_h, next_c, cache

def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """

    #Implement the backward pass for a single timestep of an LSTM.       

    f, g, i, o, prev_h, prev_c, next_c, x, Wh, Wx = cache
    N, H = dnext_h.shape

    do = dnext_h * np.tanh(next_c)
    dnext_c += dnext_h * o * (1 - np.tanh(next_c)**2)
    dprev_c = dnext_c * f
    di = dnext_c * g
    dg = dnext_c * i
    df = dnext_c * prev_c
    #Gredient of tanh and sigmoid
    dg0 = (1 - g**2) * dg 
    df0 = df * (f*(1-f))   
    di0 = di * (i*(1-i))
    do0 = do * (o*(1-o)) 
    dmiddle_state = np.hstack((di0, df0, do0, dg0))   #(N, 4H)
    dWx = x.T.dot(dmiddle_state)
    dWh = prev_h.T.dot(dmiddle_state) 
    dprev_h = dmiddle_state.dot(Wh.T)
    dx = dmiddle_state.dot(Wx.T)
    db = np.ones(N).dot(dmiddle_state)

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """

    # Implement the forward pass for an LSTM over an entire timeseries.   

    N, T, D = x.shape
    N, H = h0.shape
    next_c = np.zeros((N, H))
    next_h = h0
    h = np.zeros((N, T, H))
    caches = []
    for index in range(T):
        current_x = x[:, index, :]
        next_h, next_c, cache = lstm_step_forward(current_x, next_h, next_c, Wx, Wh, b)    
        caches.append(cache[:-2])
        h[:, index, :] = next_h
    cache = (caches, Wx, Wh)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """

    #Implement the backward pass for an LSTM over an entire timeseries.  
    caches, Wx, Wh = cache
    N, T, H = dh.shape
    D, _ = Wx.shape
    dWx = np.zeros(Wx.shape)
    dWh = np.zeros(Wh.shape)
    dx = np.zeros((N, T, D))
    db = np.zeros(4*H)
    dnext_c = np.zeros((N, H))
    dnext_h = np.zeros((N, H))
    for index in range(T-1, -1, -1):
        dnext_h += dh[:, index, :]
        current_cache = caches[index] + (Wh, Wx)
        dx_i, dnext_h, dnext_c, dWx_i, dWh_i, db_i = lstm_step_backward(dnext_h, dnext_c, current_cache)
        dx[:, index, :] += dx_i
        dWh += dWh_i
        dWx += dWx_i
        db += db_i
    dh0 = dnext_h

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
