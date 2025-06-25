from copy import deepcopy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.layers = []
        self.loss = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        #从数据层获取输入张量和标签张量。
        self.label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return self.loss_layer.forward(input_tensor, label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()
            self.update_weights()

    def update_weights(self):
        for layer in self.layers:
            if layer.trainable:
                layer.update_weights()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    def use_data_layer(self, data_layer):
        self.data_layer = data_layer

    def use_loss_layer(self, loss_layer):
        self.loss_layer = loss_layer
