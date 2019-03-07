### Neural Networks
# keras is a Tensorflow API
# Dense - layers of connection
# Units - Number of Neurons
# Sequential - Successive layers are defined in Sequence
model = keras.sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Loss functions - Measure the outcome
# Optimizer - First NN starts with a guess and then optimizer is used to improve upon it (sgd - stochastic gradient descent)
model.compile(optimizer='sgd', loss='mean_squared_error')



















