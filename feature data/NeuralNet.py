import tensorflow as tf

# Set hyperparameters
learning_rate = 0.1
epochs = 10
batch_size = 100

layersizes = [784, 300, 100, 10]
# Training data/label placeholders
x = tf.placeholder(tf.float32, [None, layersizes[0]])
y = tf.placeholder(tf.float32, [None, layersizes[-1]])
#print(y)

# Weights to first layer
W1 = tf.Variable(tf.random_normal([layersizes[0], layersizes[1]], stddev=0.05), name='W1')
b1 = tf.Variable(tf.random_normal([layersizes[1]]), name='b1')
# Weights to second layer
W2 = tf.Variable(tf.random_normal([layersizes[1], layersizes[2]], stddev=0.05), name='W2')
b2 = tf.Variable(tf.random_normal([layersizes[2]]), name='b2')
# Weights to third layer
W3 = tf.Variable(tf.random_normal([layersizes[2], layersizes[3]], stddev=0.05), name='W3')
b3 = tf.Variable(tf.random_normal([layersizes[3]]), name='b3')

# Output of hidden layers
hidden1 = tf.add(tf.matmul(x, W1), b1)
hidden1 = tf.nn.relu(hidden1)
hidden2 = tf.add(tf.matmul(hidden1, W2), b2)
hidden2 = tf.nn.relu(hidden2)

# Output of output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden2, W3), b3))
# Clip output to avoid log(0) error
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
# Cross entropy loss
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Initialization operator
init_op = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize training Session
with tf.Session() as sess:
   # Initialize variables
   sess.run(init_op)
   # Change len(mnist.train.labels) to length of the labels of our data, i.e. number of examples
   batch = int(#size of training data / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(batch):
            batch_x, batch_y = # Get batch x and labels batch_y
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / batch
        print("Epoch", (epoch + 1), ": cross-entropy cost =", "{:.3f}".format(avg_cost))
   print("Accuracy:", sess.run(accuracy, feed_dict={x: # put x here, y: # put y here}))