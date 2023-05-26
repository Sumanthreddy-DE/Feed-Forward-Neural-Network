from Layers.Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self._optimizer = []
        self._gradient_weights = []

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        bias = np.expand_dims(np.ones(batch_size), axis=1)  #add weights to bias
        self.input_tensor = np.hstack((input_tensor, bias))  # (batch_size, input_size + 1)
        output = np.dot(self.input_tensor, self.weights)
        return output

    def backward(self, error_tensor):
        loss_grad = np.dot(error_tensor, self.weights.T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        output = loss_grad[:, 0:loss_grad.shape[1] - 1]
        return output

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizerType):
        self._optimizer = optimizerType

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, grad_weight):
        self._gradient_weights = grad_weight

    @gradient_weights.deleter
    def gradient_weights(self):
        del self._gradient_weights
