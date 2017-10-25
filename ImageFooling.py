# As usual, a bit of setup
from __future__ import print_function
import time, os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from lib.classifiers.squeezenet import SqueezeNet
from lib.utils.data_utils import load_tiny_imagenet, load_imagenet_val
from lib.utils.image_utils import preprocess_image, deprocess_image
from lib.utils.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD

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

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image, of shape (1, 224, 224, 3)
    - target_y: An integer in the range [0, 1000)
    - model: Pretrained SqueezeNet model

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    X_fooling = X.copy()
    learning_rate = 1

    for i in range(100):
    	scores = sess.run(model.classifier, feed_dict = {model.image: X_fooling})
    	print('step:%d,current_label_score:%f,target_label_score:%f' % \
              (i,scores[0].max(),scores[0][target_y]))
    	predict_y = np.argmax(scores[0])
    	if predict_y == target_y:
    		break
    	losses = scores[0, target_y]
    	
    	grad_img = tf.gradients(model.classifier[0,target_y],model.image)[0]
    	grad_img_val = sess.run(grad_img, feed_dict = {model.image: X_fooling})
    	grad_img_val = grad_img_val[0]
    	dX = learning_rate * grad_img_val / np.sum(grad_img_val)
    	X_fooling += dX
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling

idx = 0
Xi = X[idx][None]
target_y = 6
X_fooling = make_fooling_image(Xi, target_y, model)

# Make sure that X_fooling is classified as y_target
scores = sess.run(model.classifier, {model.image: X_fooling})
assert scores[0].argmax() == target_y, 'The network is not fooled!'

def show_image_fooling(X, X_fooling):
	# Show original image, fooling image, and difference
	orig_img = deprocess_image(X[0])
	fool_img = deprocess_image(X_fooling[0])
	# Rescale 
	plt.subplot(1, 4, 1)
	plt.imshow(orig_img)
	plt.axis('off')
	plt.title(class_names[y[idx]])
	plt.subplot(1, 4, 2)
	plt.imshow(fool_img)
	plt.title(class_names[target_y])
	plt.axis('off')
	plt.subplot(1, 4, 3)
	plt.title('Difference')
	plt.imshow(deprocess_image((Xi-X_fooling)[0]))
	plt.axis('off')
	plt.subplot(1, 4, 4)
	plt.title('Magnified difference (10x)')
	plt.imshow(deprocess_image(10 * (Xi-X_fooling)[0]))
	plt.axis('off')
	plt.gcf().tight_layout()

	plt.show()

show_image_fooling(Xi, X_fooling)