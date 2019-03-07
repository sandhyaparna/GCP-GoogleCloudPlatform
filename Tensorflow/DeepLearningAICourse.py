### Neural Networks
# keras is a Tensorflow API
# Dense - layers of connection
# Units - Number of Neurons
# Sequential - Successive layers are defined in Sequence
model = keras.sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Loss functions - Measure how good the current guess is
# Optimizer - First NN starts with a guess and then optimizer is used to improve upon it. It generates a new and improved guess
# (sgd - stochastic gradient descent)
model.compile(optimizer='sgd', loss='mean_squared_error')

# xs - Input data
# ys - Target var
# epochs - Number of training loops 
model.fit(xs, ys, epochs=500)
model.predict([y])
















