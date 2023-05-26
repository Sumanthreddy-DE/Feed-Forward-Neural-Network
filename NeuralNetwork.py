from copy import deepcopy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.loss = []
        self.layers = []
        self.optimizer = optimizer
        self.data_layer = []
        self.loss_layer = []
        self.label_tensor = []
    
    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        
        loss = self.loss_layer.forward(input_tensor, self.label_tensor)
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if (layer.trainable):
            layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for iter in range(iterations):
            l = self.forward()
            self.backward()
            self.loss.append(l)
    
    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        # this final input_tensor is actually the output tensor now
        return input_tensor