import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3)


print("nnfs_3")


class layer_Dense:

    def __init__(self, input_shape, n_neurons):
        self.weights = 0.01 * np.random.randn(input_shape, n_neurons)
        self.biases = np.zeros(n_neurons)

        self.weight_momentum = np.zeros_like(self.weights)
        self.biase_momentum = np.zeros_like(self.biases)
    
    def forward(self, inputs):
        self.layer_input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dinputs):
        self.dw = np.dot(self.layer_input.T, dinputs)
        self.db = np.sum(dinputs, axis=0)

        self.di = np.dot(dinputs, self.weights.T)
    
    def updateWeights(self, learning_rate):
        
        momentum = 0.889

        weight_updates = momentum * self.weight_momentum + (learning_rate * self.dw)
        biase_updates = momentum * self.biase_momentum + (learning_rate * self.db)
        
        self.weight_momentum = weight_updates
        self.biase_momentum = biase_updates
        
        self.weights += weight_updates
        self.biases += biase_updates

        '''self.weights += learning_rate * self.dw
        self.biases += learning_rate * self.db'''



class activation_Relu:

    def forward(self, inputs):
        self.output =  np.maximum(0, inputs)

    def backward(self, dinputs):
        self.di = dinputs.copy()
        self.di[self.output == 0] = 0


class activation_softmax:
    
    def forward(self, inputs):
        self.inputs = inputs
        self.exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = self.exp_values / np.sum(self.exp_values, axis=1, keepdims=True)
    
    def backward(self, y_true):
        
        samples = len(self.output)
        ds = self.output.copy()

        if y_true.shape == 2:
            y_true = np.argmax(y_true, axis=1)

        ds[range(samples), y_true] -= 1

        self.di = ds / samples

        

class Loss:
     def calculate(self, y_pred, y_true):
         
         sample_Losses = self.forward(y_pred, y_true)
         batch_Loss = np.mean(sample_Losses)

         return batch_Loss
         


class loss_CategoricalCrossEntropy(Loss):
    
    def forward(self, y_pred, y_true):
        self.samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-07, 1-1e-07)

        if len(y_true.shape) == 1:
            self.correct_confidences = y_pred_clipped[range(self.samples), y_true]

        elif len(y_true.shape) == 2:
            self.correct_confidences = np.max(y_pred_clipped * y_true, axis=1)
        
        losslikelyhood = -np.log(self.correct_confidences)

        return losslikelyhood
    

    def backward(self):
        self.di = -(1 / self.samples) * (1 / self.correct_confidences)
        self.di = self.di.reshape(self.samples, 1)


layer1 = layer_Dense(2, 64)
activation1 = activation_Relu()

layer2 = layer_Dense(64, 3)
activation2 = activation_softmax()

l1 = loss_CategoricalCrossEntropy()


starting_learning_rate = -1.0
learning_rate_decay = 1e-3
step = 0


for epochs in range(10000):

    learning_rate = starting_learning_rate * \
        (1./ (1. + learning_rate_decay * step))

    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss = l1.calculate(activation2.output, y)

    # accuracy
    predictions = np.argmax(activation2.output, axis=1)
    if y.shape == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    print(f'epochs:{epochs}, ' +
          f'accuracy:{accuracy:.3f}, ' + 
          f'loss:{loss:.6f}, ' + 
          f'lr: {learning_rate} ')


    l1.backward()

    activation2.backward(y)
    layer2.backward(activation2.di)

    activation1.backward(layer2.di)
    layer1.backward(activation1.di)

    layer1.updateWeights(learning_rate)
    layer2.updateWeights(learning_rate)

    step += 1











