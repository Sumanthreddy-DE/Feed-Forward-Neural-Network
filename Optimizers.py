class Sgd:
    def __init__(self,  learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor): 
        prev_weights = weight_tensor
        new_weights = prev_weights - self.learning_rate*gradient_tensor
        return new_weights




