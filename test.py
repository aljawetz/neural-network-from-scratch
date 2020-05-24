from network import *

x_train = np.array([
    [[0, 0, 0]],
    [[0, 0, 1]],
    [[0, 1, 0]],
    [[0, 1, 1]],
    [[1, 1, 0]]
])

y_train = np.array([
    [[0]],
    [[1]],
    [[1]],
    [[0]],
    [[0]]
])
# network
net = Network()
net.add(FCLayer(3, 2, tanh))
net.add(FCLayer(2, 1, sigmoid))

# train
net.use(mse)
net.fit(x_train, y_train, epochs=500, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
