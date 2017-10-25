# As usual, a bit of setup
from __future__ import print_function
import time, os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from lib.classifiers.squeezenet import SqueezeNet
from lib.utils.data_utils import load_tiny_imagenet
from lib.utils.image_utils import preprocess_image, deprocess_image
from lib.utils.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD

from lib.utils.data_utils import load_imagenet_val

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

tf.reset_default_graph()
sess = get_session()

SAVE_PATH = 'lib/datasets/squeezenet.ckpt'
# if not os.path.exists(SAVE_PATH):
#     raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet(save_path=SAVE_PATH, sess=sess)

X_raw, y, class_names = load_imagenet_val(num=5)
X = np.array([preprocess_image(img) for img in X_raw])

#----------------------------Finish Setup----------------------------

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, numpy array of shape (N, H, W, 3)
    - y: Labels for X, numpy of shape (N,)
    - model: A SqueezeNet model that will be used to compute the saliency map.

    Returns:
    - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
    input images.
    """
    saliency = None
    # Compute the score of the correct class for each example.
    # This gives a Tensor with shape [N], the number of examples.
    #
    # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
    # for computing vectorized losses.
    correct_scores = tf.gather_nd(model.classifier,
                                  tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
    losses = tf.square(1 - correct_scores)
    #losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(model.labels, model.classifier.shape[1]), logits=model.classifier)
    grad_img = tf.gradients(losses,model.image)
    grad_img_val = sess.run(grad_img,feed_dict={model.image:X,model.labels:y})[0]
    saliency = np.sum(np.maximum(grad_img_val,0),axis=3)

    return saliency

def show_saliency_maps(X, y, mask):
    mask = np.asarray(mask)
    Xm = X[mask]
    ym = y[mask]
        
    saliency = compute_saliency_maps(Xm, ym, model)
    for i in range(mask.size):
        plt.subplot(2, mask.size, i + 1)
        plt.imshow(deprocess_image(Xm[i]))
        plt.axis('off')
        plt.title(class_names[ym[i]])
        plt.subplot(2, mask.size, mask.size + i + 1)
        plt.title(mask[i])
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(10, 4)
    plt.show()

mask = np.arange(5)
show_saliency_maps(X, y, mask)