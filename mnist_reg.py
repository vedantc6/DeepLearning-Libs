from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 784 is the one dimension vector of the image. The images are 28x28. They have been flattened
# 10 is the number of digits one digit can be. 0-9

def prog():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])

    W = tf.Variable(tf.zeros([784, 10]))    # Weights
    b = tf.Variable(tf.zeros([10]))     # Bias

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])     # Correct Answers
    # Cross-entropy equation
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    # Using gradient algorithm with a learning rate of 0.5.
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

    # Evaluating the model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))    # Gives a list of booleans
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

if __name__ == '__main__':
    prog()