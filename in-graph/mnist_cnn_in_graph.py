
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


FLAGS = None


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


def main():

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # placeholder
    tower1_x = tf.placeholder(tf.float32, [None, 784])
    tower1_y_ = tf.placeholder(tf.float32, [None, 10])
    tower2_x = tf.placeholder(tf.float32, [None, 784])
    tower2_y_ = tf.placeholder(tf.float32, [None, 10])
    tower1_x_image = tf.reshape(tower1_x, [-1, 28, 28, 1])
    tower2_x_image = tf.reshape(tower2_x, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)

    # variable
    with tf.device('/job:ps/task:0'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

    # tower1
    with tf.device('/job:worker/task:0'):
        tower1_h_conv1 = tf.nn.relu(conv2d(tower1_x_image, W_conv1) + b_conv1)
        tower1_h_pool1 = max_pool_2x2(tower1_h_conv1)
        tower1_h_conv2 = tf.nn.relu(conv2d(tower1_h_pool1, W_conv2) + b_conv2)
        tower1_h_pool2 = max_pool_2x2(tower1_h_conv2)
        tower1_h_pool2_flat = tf.reshape(tower1_h_pool2, [-1, 7 * 7 * 64])
        tower1_h_fc1 = tf.nn.relu(
            tf.matmul(tower1_h_pool2_flat, W_fc1) + b_fc1)
        tower1_h_fc1_drop = tf.nn.dropout(tower1_h_fc1, keep_prob)
        tower1_y_conv = tf.matmul(tower1_h_fc1_drop, W_fc2) + b_fc2
        tower1_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tower1_y_, logits=tower1_y_conv))
        tower1_train_step = tf.train.AdamOptimizer(
            1e-4).minimize(tower1_cross_entropy)

    # tower2
    with tf.device('/job:worker/task:1'):
        tower2_h_conv1 = tf.nn.relu(conv2d(tower2_x_image, W_conv1) + b_conv1)
        tower2_h_pool1 = max_pool_2x2(tower2_h_conv1)
        tower2_h_conv2 = tf.nn.relu(conv2d(tower2_h_pool1, W_conv2) + b_conv2)
        tower2_h_pool2 = max_pool_2x2(tower2_h_conv2)
        tower2_h_pool2_flat = tf.reshape(tower2_h_pool2, [-1, 7 * 7 * 64])
        tower2_h_fc1 = tf.nn.relu(
            tf.matmul(tower2_h_pool2_flat, W_fc1) + b_fc1)
        tower2_h_fc1_drop = tf.nn.dropout(tower2_h_fc1, keep_prob)
        tower2_y_conv = tf.matmul(tower2_h_fc1_drop, W_fc2) + b_fc2
        tower2_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tower2_y_, logits=tower2_y_conv))
        tower2_train_step = tf.train.AdamOptimizer(
            1e-4).minimize(tower2_cross_entropy)
    
    #cluster = tf.train.ClusterSpec({"worker": ["hades02:5222", "hades03:5223"] , "ps": ["hades02:6222"]})
    #server = tf.train.Server(cluster)
    print(1)
    sess = tf.Session(target="grpc://hades02:5222")
    print(2)
    sess.run(tf.global_variables_initializer())
    print(3)
    # Train
    for _ in range(500):

        batch_xs1, batch_ys1 = mnist.train.next_batch(50)
        batch_xs2, batch_ys2 = mnist.train.next_batch(50)
        sess.run([tower1_train_step, tower2_train_step], feed_dict={tower1_x: batch_xs1,
                                                                    tower1_y_: batch_ys1,
                                                                    tower2_x: batch_xs2,
                                                                    tower2_y_: batch_ys2,
                                                                    keep_prob: 0.5})

    # Test trained model
    correct_prediction = tf.equal(
        tf.argmax(tower1_y_conv, 1), tf.argmax(tower1_y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={tower1_x: mnist.test.images,
                                        tower1_y_: mnist.test.labels,
                                        keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='~/MNIST_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    # tf.app.run(maiin=main, argv=[sys.argv[0]] + unparsed)
    main()
