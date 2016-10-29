#Neural Network, Brendan Cordy 2016
import numpy as np
from random import shuffle

#Numpy Array Normalization-------------------------------------------------------------------------------

#Find the max of each column in a numpy array.
def get_maxs(M):
    cols = np.hsplit(M, len(M[0]))
    maxs = []
    for c in cols:
        maxs.append(np.max(c))
    return maxs

#Divide entries in the columns of a numpy array by values given in the list maxs.
def normalize(M, maxs):
    cols = np.hsplit(M, len(M[0]))
    z_cols = []
    for i, c in enumerate(cols):
        z_cols.append(np.multiply(1/(float(maxs[i])), c))
    return np.concatenate(z_cols, axis=1)

#Mutliply entries in the columns of a numpy array by values given in the list maxs.
def denormalize(M, maxs):
    z_cols = np.hsplit(M, len(M[0]))
    cols = []
    for i, c in enumerate(z_cols):
        cols.append(np.multiply(maxs[i],c))
    return np.concatenate(cols, axis=1)

#Datasets------------------------------------------------------------------------------------------------

#This is resued code from a random forest classifier module I wrote, modified to work for
#regression instead of classification.
class DataCSV(object):
    def __init__(self, str_data, vrs, cls, ins, outs):
        self.str_data = str_data
        self.variables = vrs
        self.classes = cls
        self.inputs = ins
        self.outputs = outs

    #Input data with outputs given, to be used for training/testing.
    @classmethod
    def regression_training(cls, filename, num_outputs):
        with open(filename,'r') as input_file:
            input_data = [line.rstrip('\n').rstrip('\r').split(',') for line in input_file]

        #Ignore empty rows (an extra newline at the end of the file will trigger this).
        no_empty_rows = [x for x in input_data if x != ['']]
        str_data = no_empty_rows

        #Extract variable names from the top row.
        variables = str_data[0]
        #Convert strings representing values of quantitative variables to floats.
        examples = [[float(val) for val in line] for line in str_data[1:]]

        #Divide each example into input variable values and output variable values.
        inputs = np.array([ex[:-num_outputs] for ex in examples])
        outputs = np.array([ex[-num_outputs:] for ex in examples])

        print "Examples: " + str(len(examples)) + ", Input variables: " + \
        str(len(variables) - num_outputs) + ", Output variables: " + str(num_outputs)

        return cls(str_data, variables, [], inputs, outputs)

    #Input data without outputs.
    @classmethod
    def regression_testing(cls, filename):
        with open(filename,'r') as input_file:
            input_data = [line.rstrip('\n').rstrip('\r').split(',') for line in input_file]

        #Ignore empty rows (an extra newline at the end of the file will trigger this).
        no_empty_rows = [x for x in input_data if x != ['']]
        str_data = no_empty_rows

        #Extract variable names from the top row.
        variables = str_data[0]
        #Convert strings representing values of quantitative variables to floats.
        inputs = np.array([[float(val) for val in line] for line in str_data[1:]])

        return cls(str_data, variables, [], inputs, [])

    def get_input_maxs(self):
        return get_maxs(self.inputs)

    def get_output_maxs(self):
        return get_maxs(self.outputs)

    def get_maxs(self):
        return get_maxs(self.inputs) + get_maxs(self.outputs)

    def normalize_inputs(self, maxs):
        return normalize(self.inputs, maxs)

    def normalize_outputs(self, maxs):
        return normalize(self.outputs, maxs)

#Neural Network------------------------------------------------------------------------------------------

#Neural network (multilayer perceptron) with any number of layers, and any layer sizes.
class NeuralNet(object):

    def __init__(self, layer_sizes):
        self.layers = len(layer_sizes)
        self.inLayerSize = layer_sizes[0]
        self.outLayerSize = layer_sizes[-1]
        self.Ws = []

        #Initialize weight matrices with values from a std normal dist.
        for i in range(len(layer_sizes)-1):
            self.Ws.append(np.random.randn(layer_sizes[i],layer_sizes[i+1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def deriv_sigmoid(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2

    #Feed examples through the network and return the final layer activation.
    def feedforward(self, Xo):
        X = Xo
        for W in self.Ws:
            #Calculuate the weighted sums at each node in this layer.
            Y = np.dot(X, W)
            #Apply the sigmoid function.
            X = self.sigmoid(Y)
        return X

    #Feed examples through the network and return the activation and activity at each layer.
    #Xs are post-sigmoid, Ys are pre-sigmoid.
    def get_activity(self, Xo):
        X = Xo
        Ys, Xs = [], [Xo]
        #Run the inputs through the network.
        for W in self.Ws:
            #Calculuate the weighted sums at each node in this layer.
            Y = np.dot(X,W)
            Ys.append(Y)
            #Apply the sigmoid function.
            X = self.sigmoid(Y)
            Xs.append(X)
        return Ys, Xs

    def cost(self, Xo, D):
        #Feed examples through the network and compute sum of squared errors.
        Xn = self.feedforward(Xo)
        return np.sum(np.multiply((Xn-D),(Xn-D)))

    def deriv_cost(self, Xo, D):
        #Build lists of activities and activations.
        Ys, Xs = self.get_activity(Xo)

        #Compute derivatives.
        dEdYs, dEdWs = [], []
        dEdYs = [np.multiply(-(D-Xs[-1]), self.deriv_sigmoid(Ys[-1]))]
        dEdWs = [np.dot(Xs[-2].T, dEdYs[-1])]

        #Chain rule y'all.
        for i in range(2, self.layers):
            dEdYs = [np.dot(dEdYs[0], self.Ws[-i+1].T) * self.deriv_sigmoid(Ys[-i])] + dEdYs
            dEdWs = [np.dot(Xs[-i-1].T, dEdYs[0])] + dEdWs
        return dEdWs

    #Change the weight matrices by adding epsilon times the gradient.
    def adjust_weights(self, Xo, D, epsilon):
        #Compute the gradient.
        dEdWs = self.deriv_cost(Xo, D)

        #Move in the direction of the gradient by epsilon.
        new_Ws = []
        for i, M in enumerate(self.Ws):
            new_Ws.append(M + epsilon * dEdWs[i])
        self.Ws = new_Ws

    #Train the network on the data in Xo and D by using gradient descent with a step size of
    #epsilon, using a fixed number of steps.
    def train(self, Xo, D, steps, epsilon):
        #Print out initial cost and weight matrices for funzies.
        print 'Initial cost: ' + str(self.cost(Xo, D)) + '\n'
        print 'Initial weight matrices: \n' + str(self.Ws) + '\n'

        #Gradient descent. That'll learn ya.
        for i in range(steps):
            self.adjust_weights(Xo, D, -epsilon)

        #Report new cost and weight matrices.
        print '...Training complete... \n'
        print 'Final cost: ' + str(self.cost(Xo, D)) + '\n'
        print 'Final weight matrices: \n' + str(self.Ws) + '\n'

    def show_predictions(self, Xo, in_maxs, out_maxs):
        print 'Testing inputs: \n' + str(denormalize(Xo, in_maxs)) + '\n'
        print 'Testing outputs: \n' + str(denormalize(self.feedforward(Xo), out_maxs)) + '\n'
