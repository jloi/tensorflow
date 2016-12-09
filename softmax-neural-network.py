import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def initialize():
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.matmul(x, W) + b
	return x, y_, W, b, y

def performConvolution(image, weight_shape, bias_shape):
	W_conv = weight_variable(weight_shape)
	b_conv = bias_variable(bias_shape)
	h_conv = tf.nn.relu(conv2d(image, W_conv) + b_conv)
	h_pool = max_pool_2x2(h_conv)
	return h_pool

def setupTensorsForTraining(y_conv, y_):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy, train_step

def createFullyConnectedLayer(h_pool):
	size = 7 * 7 * 64
	W_fc = weight_variable([size, 1024])
	b_fc = bias_variable([1024])
	h_pool_flat = tf.reshape(h_pool, [-1, size])
	h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)
	return h_fc

def createDropOuts(h_fc):
	keep_prob = tf.placeholder(tf.float32)
	h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
	return keep_prob, h_fc_drop

def createReadoutLayer(h_fc_drop):
	W_fc = weight_variable([1024, 10])
	b_fc = bias_variable([10])
	y_conv = tf.matmul(h_fc_drop, W_fc) + b_fc
	return y_conv


x, y_, W, b, y = initialize()
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_pool = performConvolution(x_image, [5, 5, 1, 32], [32])
h_pool = performConvolution(h_pool, [5, 5, 32, 64], [64])
h_fc = createFullyConnectedLayer(h_pool)
keep_prob, h_fc_drop = createDropOuts(h_fc)
y_conv = createReadoutLayer(h_fc_drop)

accuracy, train_step = setupTensorsForTraining(y_conv, y_)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={
			x: batch[0], y_: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

sess.close()