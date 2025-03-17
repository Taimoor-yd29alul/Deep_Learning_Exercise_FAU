class Sgd:
    def __init__(self, learning_rate: float):
        self._learning_rate = learning_rate

    def calculate_update(self,  weight_tensor, gradient_tensor):
        new_weights_tensor = weight_tensor - (self._learning_rate * gradient_tensor)
        return new_weights_tensor
