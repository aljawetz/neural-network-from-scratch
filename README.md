# Neural Network from Scratch
Creating a Neural Network library from scratch in Python

## Installation

Clone this repo and install the numpy library using pip
```sh
pip install numpy
```

## Usage example

```sh
# Importing
from network import *

# Creating a Network
net = Network()

# Adding a Fully Connected Layer with input size 3, output size 1 and tanh activation function to the network
# sigmoid, tanh, relu and none activation functions available
net.add(FCLayer(input_size=3, output_size=1, activation=tanh))

# Using the mean squared error as loss function
# mse and sse loss functions available
net.use(loss=mse)

# Training the neural network
net.fit(x_train, y_train, epochs=100, learning_rate=0.1)

# Making predictions
predictions = net.predict(x_pred)
```

## Contributing

1. Fork it (<https://github.com/aljawetz/neural-network-from-scratch/fork>)
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
