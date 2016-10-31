neural_networker
===============

Feedforward Neural Network with Gradient Descent in Python.

How about we make a neural network learn to add numbers between one and twenty? Let's use two nodes in the input layer (the two terms to be summed), two hidden layers with five and four nodes each, and a single node in the output layer.

```
>>>N = NeuralNet([2,5,4,1])
>>>sums_training_data = DataCSV.regression_training('sums_train.csv', 1)
Examples: 10, Input variables: 2, Output variables: 1
```

The `NeuralNet` constructor takes a list of sizes of the layers. The `DataCSV.regression_training` call parses the data in a CSV file. The first line of this file must contain the variable names, while the other lines are examples we'll use to train the neural network. The second argument to this call is the number of output variables, which are assumend to be furthest right in the CSV. In this case, there is one output variable, the sum.

Next, the data will have to be normalized before the training starts. The normalization is done by simply dividing all values of each variable by the maximum value of that variable present in the dataset. We'll need to store these maximum values in a list so we can reuse them later. Then we can normalize the inputs and outputs.

```
>>>in_maxs = sums_training_data.get_input_maxs()
>>>out_maxs = sums_training_data.get_output_maxs()
>>>Xo = sums_training_data.normalize_inputs(in_maxs)
>>>D = sums_training_data.normalize_outputs(out_maxs)
```

The matrix `Xo` contains rows of example inputs to be fed into to the first layer, while the matrix `D` contains the desired outputs when the examples come out the other end (I'm roughly following the notation in [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf). Matrices are represented as numpy arrays throughout.

Now we can train the neural network with gradient descent. Let's move in the direction of the gradient 10000 times, each time adjusting the weight matrices by adding 5 times the gradient. This will produce a bunch of output, reporting the cost before training and the cost after training, likewise for the weights in the network. The cost function is the sum of squared errors, and the activation function at each node is the logistic curve.

```
>>>N.train(Xo, D, 10000, 5)
Initial cost: 0.963366366666

Initial weight matrices:
    [array([[-1.08201359,  0.08666151,  1.11983888,  0.37455837,  1.23274228],
    [ 1.17904517, -0.80306934,  1.45694537,  0.33968163,  0.57140519]]),
    array([[ 1.28963611,  1.02902573, -0.80575728, -0.76089612],
    [-1.16258948, -0.74541203, -0.53218717,  1.05019559],
    [-0.45872475, -0.41637176, -0.33426049, -2.21596128],
    [ 0.29860438,  0.06925221,  1.18882199,  0.65278895],
    [ 0.50749625,  0.16538508,  0.03953266, -0.40324495]]),
    array([[ 1.3725006 ],
    [ 0.81335753],
    [ 0.18541208],
    [ 0.67246648]])]

...Training complete...

Final cost: 0.000185084629701

Final weight matrices:
    [array([[-3.17673514, -1.4325484 ,  1.50707634, -1.33849055, -0.79677038],
    [ 2.81463828, -1.5140025 ,  0.87547773, -1.49344654, -1.47811967]]),
    array([[  2.824761  ,  -0.82676358,  -1.73836325,  -0.59950383],
    [-11.06837213,  -1.88536573,   0.42552252,   4.90211698],
    [  0.95067092,  -0.78872699,  -3.0366897 ,  -5.22436022],
    [ -9.18448941,  -1.27700817,   1.6545084 ,   4.00082943],
    [ -4.1022713 ,  -1.30362083,  -1.68233432,  -1.25603596]]),
    array([[ 8.15714398],
    [ 1.79631575],
    [-0.82980745],
    [-3.84603807]])]

```

Now the neural network should be able to do sums of small positive numbers. We can check by using the `sums_predict.csv` file. The call to `DataCSV.regression_testing` parses a csv file where the values of the output variables are absent. There's no need to specify the number of output varibles this time, since that information is in the structure of the neural network that's already been built.

Any inputs/outputs will need to be normalized in the same way as the training data, so we'll use the `in_maxs` and `out_maxs` lists that were made from the training data. Calling `N.show_predictions` with these new normalized inputs will run them through the network, and print the (denormalized) inputs as well as the (denormalized) resulting outputs.

```
>>>sums_prediction = DataCSV.regression_prediction('sums_predict.csv')
>>>To = sums_prediction.normalize_inputs(in_maxs)
>>>N.show_predictions(To, in_maxs, out_maxs)
Inputs:
[[  2.   6.]
 [  8.   8.]
 [  1.   5.]
 [ 11.   4.]
 [ 17.   2.]
 [  5.   9.]
 [  1.   3.]
 [  4.   6.]]

Outputs:
[[  7.9405537 ]
 [ 16.36732913]
 [  5.64545299]
 [ 15.223159  ]
 [ 18.85156674]
 [ 15.06215616]
 [  3.82348212]
 [ 10.12259362]]

```

Hooray! It's pretty good at adding.
