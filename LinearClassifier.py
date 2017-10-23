import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from lib.features import *

from lib.classifiers.linear_classifier import LinearSVM, Softmax
from lib.utils.data_utils import get_CIFAR10_data

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data = get_CIFAR10_data()
X_train, y_train, X_val, y_val, X_test, y_test = data["X_train"].transpose(0, 2, 3, 1), data["y_train"], data["X_val"].transpose(0, 2, 3, 1), data["y_val"], data["X_test"].transpose(0, 2, 3, 1), data["y_test"]

# Extract features instean of original pixels
num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])

# Use the validation set to tune the learning rate and regularization strength
learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]


results = {}
best_val = -1
best_svm = None

#SVM or Softmax linear classifer
validationSet = product(set(learning_rates), set(regularization_strengths))
for (lr, reg) in validationSet:
    SVM = Softmax()
    loss_hist = SVM.train(X_train_feats, y_train, lr, reg)
    y_train_pred = SVM.predict(X_train_feats)
    current_train_accuracy = np.mean((y_train == y_train_pred).astype(int, copy=False))
    y_val_pred = SVM.predict(X_val_feats)
    current_val_accuracy = np.mean((y_val_pred == y_val).astype(int, copy=False))
    results[(lr, reg)] = (current_train_accuracy, current_val_accuracy)
    if current_val_accuracy > best_val:
        best_val = current_val_accuracy
        best_svm = SVM

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)

#tuned the best hyperparameters
#get best result: lr:e-07 reg: 5e06

#train SVM using cross_validation
#too easy to continue...

#get final SVM
# lr = 1e-7
# reg = 5e5
# SVM = Softmax()   ##Choose classifier here
# loss_hist = SVM.train(X_train_feats, y_train, lr, reg)
# print(loss_hist)

# y_test_pred = SVM.predict(X_test_feats)
# current_test_accuracy = np.mean((y_test_pred == y_test).astype(int, copy=False))
# print("Final accuracy: %f" % current_test_accuracy)

#way1, print false cases
# examples_per_class = 8
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# for cls, cls_name in enumerate(classes):
#     idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
#     idxs = np.random.choice(idxs, examples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
#         plt.imshow(X_test[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls_name)
# plt.show()

#way2, check histogram
# plt.plot(loss_hist)
# plt.show()