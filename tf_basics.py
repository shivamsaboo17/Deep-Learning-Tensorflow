"""
Tensorflow is basically a matrix operation library written in C++ for high performance
A Tensor is just a value, i.e it may be any dimensional matrix or an array.
In mathematics, tensors are geometric objects that describe linear relations between geometric vectors, scalars, and
other tensors. Elementary examples of such relations include the dot product, the cross product, and linear maps.
Tensorflow provides with many operations as well as helper functions used for Deep Learning
The advantage of using tensorflow is that it makes model beforehand, does all the pre-processing and hence
is more efficient than Python's way of interpreting the code line by line.

"""

import tensorflow as tf

"""
A constant in Tensorflow is value which doesn't change just as in C++ or Java

"""

# First define the model of the computational graph

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)

# This won't print 30 but just the tensor 'result' is created
print(result)

# To actually print the result we need to run the session as follow:
with tf.Session() as sess:
    print(sess.run(result))