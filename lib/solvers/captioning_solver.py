from __future__ import print_function, division
from builtins import range
from builtins import object
import numpy as np
import nltk

from lib import optim
from lib.utils.coco_utils import sample_coco_minibatch, decode_captions


class CaptioningSolver(object):
    """
    A CaptioningSolver encapsulates all the logic necessary for training
    image captioning models. The CaptioningSolver performs stochastic gradient
    descent using different update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a CaptioningSolver instance,
    passing the model, dataset, and various options (learning rate, batch size,
    etc) to the constructor. You will then call the train() method to run the
    optimization procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists containing
    the accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    solver = CaptioningSolver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A CaptioningSolver works on a model object that must conform to the following
    API:

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.

    - model.loss(features, captions) must be a function that computes
      training-time loss and gradients, with the following inputs and outputs:

      Inputs:
      - features: Array giving a minibatch of features for images, of shape (N, D
      - captions: Array of captions for those images, of shape (N, T) where
        each element is in the range (0, V].

      Returns:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new CaptioningSolver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data from load_coco_data

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the learning
          rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient during
          training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        """
        self.model = model
        self.data = data

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()


    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        minibatch = sample_coco_minibatch(self.data,
                      batch_size=self.batch_size,
                      split='train')
        captions, features, urls = minibatch

        # Compute loss and gradient
        loss, grads = self.model.loss(features, captions)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
          much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        
        N, T = y.shape
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.sample(X, max_length=T)
            y_pred.append(scores)
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc


    def train(self):
        """
        Run optimization to train the model.
        """
        data = self.data
        num_train = data['train_captions'].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        best_val_acc = 0.0

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))

            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            first_iter = t==0
            last_iter = t == num_iterations -1
            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            # Implement some logic to check Bleu on validation set periodically
            if first_iter or last_iter or epoch_end:

                # Raw way to test
                # minibatch = sample_coco_minibatch(data, split="train", batch_size=50)
                # captions, features, _ = minibatch 
                # train_acc = self.check_accuracy(features, captions)
                # minibatch = sample_coco_minibatch(data, split="val", batch_size=50)
                # captions, features, _ = minibatch
                # val_acc = self.check_accuracy(features, captions)

                # self.train_acc_history.append(train_acc)
                # self.val_acc_history.append(val_acc)

                BLEUscores = self.evaluate_model()
                train_acc = BLEUscores["train"]
                val_acc = BLEUscores["val"]
                
                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params

    def BLEU_score(self, gt_caption, sample_caption):
        """
        gt_caption: string, ground-truth caption
        sample_caption: string, your model's predicted caption
        Returns unigram BLEU score.
        """
        reference = [x for x in gt_caption.split(' ') 
                     if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
        hypothesis = [x for x in sample_caption.split(' ') 
                      if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
        return BLEUscore

    def evaluate_model(self):
        """
        model: CaptioningRNN model
        Prints unigram BLEU score averaged over 1000 training and val examples.
        """
        model = self.model
        data = self.data
        BLEUscores = {}
        for split in ['train', 'val']:
            minibatch = sample_coco_minibatch(self.data, split=split, batch_size=1000)
            gt_captions, features, urls = minibatch
            gt_captions = decode_captions(gt_captions, data['idx_to_word'])

            sample_captions = model.sample(features)
            sample_captions = decode_captions(sample_captions, data['idx_to_word'])

            total_score = 0.0
            for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
                total_score += self.BLEU_score(gt_caption, sample_caption)

            BLEUscores[split] = total_score / len(sample_captions)

        for split in BLEUscores:
            print('Average BLEU score for %s: %f' % (split, BLEUscores[split]))

        return BLEUscores
