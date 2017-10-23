import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from lib.solvers.tfSolver import run_model
from lib.utils.data_utils import get_CIFAR10_data

#data prep
data = get_CIFAR10_data()
X_train, y_train, X_val, y_val, X_test, y_test = data["X_train"], data["y_train"], data["X_val"], data["y_val"], data["X_test"], data["y_test"]

# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# define model
def complex_model(X,y,is_training):

    N, H, W, C = X.shape

    #initialization
    Wconv1 = tf.get_variable("Wconv1", [7, 7, 3, 32], initializer = tf.contrib.layers.xavier_initializer())
    bconv1 = tf.get_variable("bconv1", [32, ], initializer = tf.zeros_initializer())
    gamma1 = tf.get_variable("gamma1", [32, ], initializer = tf.ones_initializer())
    beta1 = tf.get_variable("beta1", [32, ], initializer = tf.zeros_initializer())
    running_mean = tf.get_variable("running_mean", [32, ], initializer = tf.zeros_initializer())
    running_variance = tf.get_variable("running_variance", [32, ], initializer = tf.ones_initializer())
    W1 = tf.get_variable("W1", [8192, 10], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [10, ], initializer = tf.zeros_initializer())

    #construct CG
    A1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1 ,1], padding='SAME') + bconv1
    A1b = tf.layers.batch_normalization(A1, training=is_training)
    H1 = tf.nn.relu(A1b)
    #tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    H1P = tf.nn.max_pool(H1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    #H1D = tf.layers.dropout(H1P, 0.25, training=is_training)
    H1_reshaped = tf.reshape(H1P, [-1, 8192])
    y_out = tf.matmul(H1_reshaped, W1) + b1
    return y_out

y_out = complex_model(X,y,is_training)

# define our loss
total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

## Train model
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print('Training')
	run_model(sess, X, y, is_training, y_out, mean_loss,X_train,y_train,1,64,100,train_step)
	print('Validation')
	run_model(sess,X, y, is_training, y_out,mean_loss,X_val,y_val,1,64)
# check correctness
# Now we're going to feed a random batch into the model 
# and make sure the output is the right size

# x = np.random.randn(64, 32, 32,3)
# with tf.Session() as sess:
#     with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
#         tf.global_variables_initializer().run()

#         ans = sess.run(y_out,feed_dict={X:x,is_training:True})
#         print(ans.shape)
#         print(np.array_equal(ans.shape, np.array([64, 10])))

#check forward pass:
# try:
#     with tf.Session() as sess:
#         with tf.device("/cpu:0") as dev: #"/cpu:0" or "/gpu:0"
#         	x = np.random.randn(64, 32, 32,3)
#             tf.global_variables_initializer().run()

#             ans = sess.run(y_out,feed_dict={X:x,is_training:True})
# except tf.errors.InvalidArgumentError:
#     print("no gpu found, please use Google Cloud if you want GPU acceleration")    
#     # rebuild the graph
#     # trying to start a GPU throws an exception 
#     # and also trashes the original graph
#     tf.reset_default_graph()
#     X = tf.placeholder(tf.float32, [None, 32, 32, 3])
#     y = tf.placeholder(tf.int64, [None])
#     is_training = tf.placeholder(tf.bool)
#     y_out = complex_model(X,y,is_training)
