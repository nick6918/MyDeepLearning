from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver
from cs231n.vis_utils import visualize_grid

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data(num_training=5000)
for k, v in data.items():
  print('%s: ' % k, v.shape)

model = ThreeLayerConvNet()

#1, Using random data to get intuition on loss
# N = 50
# X = np.random.randn(N, 3, 32, 32)
# y = np.random.randint(10, size=N)

# loss, grads = model.loss(X, y)
# print('Initial loss (no regularization): ', loss)

# model.reg = 0.5
# loss, grads = model.loss(X, y)
# print('Initial loss (with regularization): ', loss)

#2, Use numeric gradient check to check gredient
# num_inputs = 2
# input_dim = (3, 16, 16)
# reg = 0.0
# num_classes = 10
# np.random.seed(231)
# X = np.random.randn(num_inputs, *input_dim)
# y = np.random.randint(num_classes, size=num_inputs)

# model = ThreeLayerConvNet(num_filters=3, filter_size=3,
#                           input_dim=input_dim, hidden_dim=7,
#                           dtype=np.float64)
# loss, grads = model.loss(X, y)
# for param_name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
#     e = rel_error(param_grad_num, grads[param_name])
#     print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

#3, Overfit small data to check model correctness

# np.random.seed(231)

# num_train = 100
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }

# model = ThreeLayerConvNet(weight_scale=1e-2, dropout=0.75, use_batchnorm=True)

# solver = Solver(model, small_data,
#                 num_epochs=15, batch_size=50,
#                 update_rule='adam',
#                 optim_config={
#                   'learning_rate': 1e-3,
#                 },
#                 verbose=True, print_every=1)
# solver.train()

# plt.subplot(2, 1, 1)
# plt.plot(solver.loss_history, 'o')
# plt.xlabel('iteration')
# plt.ylabel('loss')

# plt.subplot(2, 1, 2)
# plt.plot(solver.train_acc_history, '-o')
# plt.plot(solver.val_acc_history, '-o')
# plt.legend(['train', 'val'], loc='upper left')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()




#4, train the model first time, get raw data
model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, dropout=0.55, use_batchnorm=True)

solver = Solver(model, data,
                num_epochs=10, batch_size=100,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()

plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

#visualize first layer wirght
grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()
