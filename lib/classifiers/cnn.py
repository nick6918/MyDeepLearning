from builtins import object
import numpy as np

from lib.layers.layers import *
from lib.layers.fast_layers import *
from lib.layers.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - bn - relu - 2x2 max pool - affine - bn - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, dropout=0, use_batchnorm=False, pool_param={'pool_height': 2, 'pool_width': 2, 'stride': 2}, conv_param=None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.pool_param = pool_param
        if conv_param:
            self.conv_param = conv_param
        else:
            self.conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        if self.use_batchnorm:
            self.bn_param1 = {'mode': 'train'}
            self.bn_param2 = {'mode': 'train'}

        self.use_dropout = False
        if dropout > 0:
            self.use_dropout = True
            self.dropout_param = {'mode': 'train', 'p': dropout}

        C, H, W = input_dim
        Hn = (H - filter_size + 2 * self.conv_param['pad']) // self.conv_param['stride'] + 1
        Wn = (W - filter_size + 2 * self.conv_param['pad']) // self.conv_param['stride'] + 1
        Hp = (Hn - self.pool_param["pool_height"]) // self.pool_param["stride"] + 1
        Wp = (Wn - self.pool_param["pool_width"]) // self.pool_param["stride"] + 1

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        
        # Conv layer initialization
        #(N, C, HW, WW)
        self.params["W1"] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params["b1"] = np.zeros(num_filters)
        # FC layer initialization
        self.params["W2"] = weight_scale * np.random.randn(num_filters * Hp * Wp, hidden_dim)
        self.params["b2"] = np.zeros(hidden_dim)
        #Loss layer initialization
        self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes)
        if self.use_batchnorm:
            #Spatial BN
            self.params["gamma1"] = np.ones(num_filters)
            self.params["beta1"] = np.zeros(num_filters)
            #Affine BN
            self.params["gamma2"] = np.ones(hidden_dim)
            self.params["beta2"] = np.zeros(hidden_dim)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            self.bn_param1['mode'] = mode
            self.bn_param2['mode'] = mode

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        if self.use_batchnorm:
            gamma1, beta1 = self.params['gamma1'], self.params['beta1']
            gamma2, beta2 = self.params['gamma2'], self.params['beta2']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        #conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        #pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        caches = []
        if self.use_batchnorm:
            H1, cache = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, self.conv_param, self.bn_param1, self.pool_param)
            caches.append(cache)
            if self.use_dropout:
                H1, cache = dropout_forward(H1, self.dropout_param)
                caches.append(cache) 
            H1, cache = flatten_forward(H1)
            caches.append(cache)
            H2, cache = affine_bn_relu_forward(H1, W2, b2, gamma2, beta2, self.bn_param2)
            caches.append(cache)
            if self.use_dropout:
                H2, cache = dropout_forward(H2, self.dropout_param)
                caches.append(cache) 
        else:
            H1, cache = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
            caches.append(cache)
            if self.use_dropout:
                H1, cache = dropout_forward(H1, self.dropout_param)
                caches.append(cache)
            H1, cache = flatten_forward(H1)
            caches.append(cache)
            H2, cache = affine_relu_forward(H1, W2, b2)
            caches.append(cache)
            if self.use_dropout:
                H2, cache = dropout_forward(H2, self.dropout_param)
                caches.append(cache)
        scores, cache = affine_forward(H2, W3, b3)
        caches.append(cache)
    
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dLoss_dscores = softmax_loss(scores, y)
        loss += 0.5*self.reg*np.sum(self.params["W3"]**2)
        loss += 0.5*self.reg*np.sum(self.params["W2"]**2)
        loss += 0.5*self.reg*np.sum(self.params["W1"]**2)
        dLoss_dH2, grads["W3"], grads["b3"] = affine_backward(dLoss_dscores, caches.pop())
        if self.use_dropout:
                dLoss_dH2 = dropout_backward(dLoss_dH2, caches.pop())
        if self.use_batchnorm:
            dLoss_dH1, grads["W2"], grads["b2"], grads["gamma2"], grads["beta2"] = affine_bn_relu_backward(dLoss_dH2, caches.pop())
            dLoss_dH1 = flatten_backward(dLoss_dH1, caches.pop())
            if self.use_dropout:
                dLoss_dH1 = dropout_backward(dLoss_dH1, caches.pop())
            dLoss_dX, grads["W1"], grads["b1"], grads["gamma1"], grads["beta1"] = conv_bn_relu_pool_backward(dLoss_dH1, caches.pop())
        else:
            dLoss_dH1, grads["W2"], grads["b2"] = affine_relu_backward(dLoss_dH2, caches.pop())
            dLoss_dH1 = flatten_backward(dLoss_dH1, caches.pop())
            if self.use_dropout:
                dLoss_dH1 = dropout_backward(dLoss_dH1, caches.pop())
            dLoss_dX, grads["W1"], grads["b1"] = conv_relu_pool_backward(dLoss_dH1, caches.pop())
        grads["W3"] += self.reg*self.params["W3"]
        grads["W2"] += self.reg*self.params["W2"]
        grads["W1"] += self.reg*self.params["W1"]

        return loss, grads
