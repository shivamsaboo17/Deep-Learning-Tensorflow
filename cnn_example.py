import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""

input > weight > hidden layer 1 (activation function) > weight > hidden layer 2 (activation function)>
weight > output layer

compare output to the intended output > cost function (cross entropy)
optimizer function > minimize the cost (Adam, SGD, AdaGrad etc)

backpropagate!

feed forward + back propagate = epoch

repeat for 10-12 epoch

"""

# one hot = True gives vectorized output for multi-classification i.e 0 is represented as:
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Define the number of nodes in each layer


n_classes = 10
batch_Size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# The computational graph for the neural network
def convolutional_network(x):

    # Define the weights and biases of hidden layers and output layer

    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([ 32])),
               'b_conv2': tf.Variable(tf.random_normal([64])),
               'b_fc': tf.Variable(tf.random_normal([1024])),
               'out': tf.Variable(tf.random_normal([n_classes]))}

    # Reshape input to a 4D tensor
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution Layer, using our function
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    # Convolution Layer
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output


def train_neural_network(x):
    prediction = convolutional_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_Size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_Size)
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss += c

            print("Epoch", epoch, "completed out of", hm_epochs, "loss:", epoch_loss)

        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print("Accuracy", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)
