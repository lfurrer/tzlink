
""" An early-stopping callback based on ranking accuracy
evaluator = Evaluator()
1. evaluator returns accuracy by evaluator.accuracy
2. the callback function somehow
	stores the accuracy at the beginning of each epoch?
	terminates early if accuracy does not improve in a
        given number of epochs
3. what else do we need? do we need loss?
"""

import keras

class ranking_accuracy_callback(keras.callbacks.Callback):
    '''A callback function based on ranking accuracy.
    Modified from keras EarlyStopping callback

    #Arguments
        evaluator: the evaluator object whose accuracy is
            measured
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement.
        patience: number of epochs with no improvement,
            default = 0
        after whichtraining will be stopped.
        verbose: verbosity mode.
        baseline: Baseline value for the monitored quantity
            to reach. Training will stop if the model does
            not show improvement over the baseline.
    '''
    def __init__(self,
                 evaluator,
                 patience=0,
                 verbose=0,
                 baseline=None):
    super(EarlyStopping, self).__init__()

    self.evaluator = evaluator
    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = 1
    self.wait = 0
    self.stopped_epoch = 0

    def on_train_begin(self, logs={}):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = -np.Inf
        self.accuracy = []

    #def on_epoch_begin(self, logs={}):
    #    return
 
    def on_epoch_end(self, epoch, logs={}):
        self.accuracy.append(getattr(evaluator, accuracy))
        current = logs.get(self.accuracy)
        if np.greater(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    #def on_batch_begin(self, batch, logs={}):
    #    return

    #def on_batch_end(self, batch, logs={}):
    #    return

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
