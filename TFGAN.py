from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# A bunch of utility functions

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./cs231n/datasets/MNIST_data', one_hot=False)

# show a batch
# print("get here")
# show_images(mnist.train.next_batch(16)[0])
# plt.show()

def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function."""
    return tf.maximum(alpha*x, x)

def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    noise = tf.random_uniform([batch_size, dim], -1, 1)
    return noise

def basic_fc_discriminator(x):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """

    with tf.variable_scope("bfcdiscriminator"):

        W1 = tf.get_variable("W1", (784, 256))
        b1 = tf.get_variable("b1", (256, ), initializer=tf.zeros_initializer())
        W2 = tf.get_variable("W2", (256, 256))
        b2 = tf.get_variable("b2", (256, ), initializer=tf.zeros_initializer())
        W3 = tf.get_variable("W3", (256, 1), )
        b3 = tf.get_variable("b3", (1, ), initializer=tf.zeros_initializer())

        H1 = tf.matmul(x, W1) + b1
        H1L = leaky_relu(H1)
        H2 = tf.matmul(H1L, W2) + b2
        H2L = leaky_relu(H2)
        logits = tf.matmul(H2L, W3) + b3

        return logits

def fc_discriminator(x):
    #use layers, you don't have to initialize variables yourself
    with tf.variable_scope("fcdiscriminator"):
        H1 = tf.layers.dense(x, units = 256, activation = None, use_bias = True)
        H1L = leaky_relu(H1)
        H2 = tf.layers.dense(H1L, units = 256, activation = None, use_bias = True)
        H2L = leaky_relu(H2)
        logits = tf.layers.dense(H2L, units = 1, activation = None, use_bias = True)

        return logits

def discriminator(x):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """

    with tf.variable_scope("discriminator"):
        #implement architecture
        X_reshaped = tf.reshape(x, shape=[-1, 28, 28, 1])
        H1 = tf.layers.conv2d(inputs=X_reshaped, filters=32, kernel_size=5, strides=1,activation=None, padding='VALID', use_bias=True)
        H1D = leaky_relu(H1)
        H1_pooled = tf.layers.max_pooling2d(inputs = H1D, strides=2, pool_size=2)
        H2 = tf.layers.conv2d(inputs=H1_pooled, filters=64, kernel_size=5, strides=1,activation=None, padding='VALID', use_bias=True)
        H2D = leaky_relu(H2)
        H3 = tf.layers.max_pooling2d(inputs = H2D, strides=2, pool_size=2)
        H3_flattened = tf.reshape(H3, shape=[-1, 4*4*64])
        H4 = tf.layers.dense(inputs=H3_flattened, units=4*4*64, activation = None, use_bias = True)
        H4D = leaky_relu(H4)
        logits = tf.layers.dense(inputs=H4D, units=1, activation = None, use_bias = True)
        return logits

def fc_generator(z):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("fcgenerator"):
        H1 = tf.layers.dense(z, units = 1024, activation = tf.nn.relu, use_bias = True)
        H2 = tf.layers.dense(H1, units = 1024, activation = tf.nn.relu, use_bias = True)
        img = tf.layers.dense(H2, units = 784, activation = tf.nn.tanh, use_bias = True)
        
        return img

def generator(z):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    with tf.variable_scope("generator"):
        #implement architecture
        H1 = tf.layers.dense(inputs = z, units = 1024, activation = tf.nn.relu, use_bias = True)
        H1_BN = tf.layers.batch_normalization(inputs=H1, axis=1)
        H2 = tf.layers.dense(inputs = H1_BN, units = 7*7*128, activation = tf.nn.relu, use_bias = True)
        H2_BN = tf.layers.batch_normalization(inputs=H2, axis=1)
        H2_reshaped = tf.reshape(H2_BN, shape = [-1, 7, 7, 128])
        H3 = tf.layers.conv2d_transpose(inputs = H2_reshaped, strides = 2, filters = 64, kernel_size = 4, padding = 'SAME', activation =tf.nn.relu, use_bias = True)
        H3_BN = tf.layers.batch_normalization(inputs = H3, axis = 3)
        img = tf.layers.conv2d_transpose(inputs = H3_BN, strides = 2, filters = 1, kernel_size = 4, padding = 'SAME', activation =tf.nn.tanh, use_bias = True)
        return img
        
def wgangp_loss(logits_real, logits_fake, batch_size, x, G_sample):
    """Compute the WGAN-GP loss.
    
    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image
    - batch_size: The number of examples in this batch
    - x: the input (real) images for this batch
    - G_sample: the generated (fake) images for this batch
    
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    
    #compute D_loss and G_loss    
    D_loss = - tf.reduce_mean(logits_real) + tf.reduce_mean(logits_fake)
    G_loss = - tf.reduce_mean(logits_fake)

    # lambda from the paper
    lam = 10
    
    # random sample of batch_size (tf.random_uniform)
    eps = tf.random_uniform([batch_size,1], minval=0.0, maxval=1.0)
    x_hat = eps*x+(1-eps)*G_sample
    #diff = G_sample - x
    #interp = x + (eps * diff)
    
    # Gradients of Gradients is kind of tricky!
    with tf.variable_scope('',reuse=True) as scope:
        grad_D_x_hat = tf.gradients(discriminator(x_hat), x_hat)
    
    grad_norm = tf.norm(grad_D_x_hat[0], axis=1, ord='euclidean')
    grad_pen = tf.reduce_mean(tf.square(grad_norm-1))
    #slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_x_hat), reduction_indices=[1]))
    #grad_pen = tf.reduce_mean((slopes - 1.) ** 2)
    
    
    D_loss += lam*grad_pen

    return D_loss, G_loss

def lsgan_loss(score_real, score_fake):
    """Compute the Least Squares GAN loss.
    
    Inputs:
    - score_real: Tensor, shape [batch_size, 1], output of discriminator
        score for each real image
    - score_fake: Tensor, shape[batch_size, 1], output of discriminator
        score for each fake image    
          
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """

    #compute D_loss and G_loss
    G_loss = 0.5 * tf.reduce_mean((score_fake-1)**2)
    D_loss = 0.5 * tf.reduce_mean((score_real-1)**2)\
             + 0.5 * tf.reduce_mean(score_fake**2)
    return D_loss, G_loss

def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.
    
    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image
    
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """

    #compute D_loss and G_loss
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_real), logits=logits_real)) +  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(logits_fake), logits=logits_fake))
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_fake), logits=logits_fake))
    return D_loss, G_loss   

#create an AdamOptimizer for D_solver and G_solver
def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    
    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)
    
    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    """
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    return D_solver, G_solver

def construct_GAN_CG():

    tf.reset_default_graph()

    # number of images for each batch
    batch_size = 128
    # our noise dimension
    noise_dim = 96

    # placeholder for images from the training dataset
    x = tf.placeholder(tf.float32, [None, 784])
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_dim)
    # generated images
    G_sample = generator(z)

    with tf.variable_scope("") as scope:
        #scale images to be -1 to 1
        logits_real = discriminator(preprocess_img(x))
        # Re-use discriminator weights on new inputs
        scope.reuse_variables()
        logits_fake = discriminator(G_sample)

    # Get the list of variables for the discriminator and generator
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

    # get our solver
    D_solver, G_solver = get_solvers()

    # get our loss
    D_loss, G_loss = lsgan_loss(logits_real, logits_fake)

    # setup training steps
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
    D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
    G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
    return x, G_sample, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step

# a giant helper function
def gan_train(sess, x, G_sample, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, show_every=250, print_every=50, batch_size=128, num_epoch=3, dataset=mnist):
    """Train a GAN for a certain number of epochs.
    
    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    Returns:
        Nothing
    """
    # compute the number of iterations we need
    max_iter = int(dataset.train.num_examples*num_epoch/batch_size)

    imgs_in_process = []

    for it in range(max_iter):
        # every show often, show a sample result
        if it % show_every == 0:
            samples = sess.run(G_sample)
            # fig = show_images(samples[:16])
            # plt.show()
            imgs_in_process.append(samples[:16])
            print("Saved images in iter %d" % it)
        # run a batch of data through the network
        minibatch, minbatch_y = dataset.train.next_batch(batch_size)
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
        _, G_loss_curr = sess.run([G_train_step, G_loss])

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if it % print_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    return imgs_in_process, G_sample

show_every = 500

x, G_sample, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step= construct_GAN_CG()
with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    imgs_in_process, G_sample = gan_train(sess,  x, G_sample, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, show_every=show_every)
    print('Samples during training')
    f, axarr = plt.subplots(1,len(imgs_in_process))
    for i in range(len(imgs_in_process)):
        current_step = i * show_every + 1
        current_img = imgs_in_process[i]
        axarr[i].axis('off')
        axarr[i].set_title("Iteration %d" % current_step)
        show_images(current_img)
    plt.show()
    print('Final images')
    samples = sess.run(G_sample)
    fig = show_images(samples[:16])
    plt.show()