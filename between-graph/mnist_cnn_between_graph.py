
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


def main(_):

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Import data
        mnist = input_data.read_data_sets('~/', one_hot=True)

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            # placeholder
            tower1_x = tf.placeholder(tf.float32, [None, 784])
            tower1_y_ = tf.placeholder(tf.float32, [None, 10])

            tower1_x_image = tf.reshape(tower1_x, [-1, 28, 28, 1])

            keep_prob = tf.placeholder(tf.float32)

            # variable

            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])

            # tower1

            tower1_h_conv1 = tf.nn.relu(
                conv2d(tower1_x_image, W_conv1) + b_conv1)
            tower1_h_pool1 = max_pool_2x2(tower1_h_conv1)
            tower1_h_conv2 = tf.nn.relu(
                conv2d(tower1_h_pool1, W_conv2) + b_conv2)
            tower1_h_pool2 = max_pool_2x2(tower1_h_conv2)
            tower1_h_pool2_flat = tf.reshape(tower1_h_pool2, [-1, 7 * 7 * 64])
            tower1_h_fc1 = tf.nn.relu(
                tf.matmul(tower1_h_pool2_flat, W_fc1) + b_fc1)
            tower1_h_fc1_drop = tf.nn.dropout(tower1_h_fc1, keep_prob)
            tower1_y_conv = tf.matmul(tower1_h_fc1_drop, W_fc2) + b_fc2
            tower1_cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tower1_y_, logits=tower1_y_conv))
            global_step = tf.contrib.framework.get_or_create_global_step()
            tower1_train_step = tf.train.AdamOptimizer(
                1e-4).minimize(tower1_cross_entropy, global_step=global_step)

            # calculate accuracy
            correct_prediction = tf.equal(
                tf.argmax(tower1_y_conv, 1), tf.argmax(tower1_y_, 1))
            accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

            # The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=1)]

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.

            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(
                                                       FLAGS.task_index == 0),
                                                   checkpoint_dir= None,
                                                   hooks=hooks) as mon_sess:
                while not mon_sess.should_stop():
                    # Run a training step asynchronously.
                    # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                    # perform *synchronous* training.
                    # mon_sess.run handles AbortedError in case of preempted PS.
                    batch_xs1, batch_ys1 = mnist.train.next_batch(50)
                    mon_sess.run(tower1_train_step, feed_dict={
                        tower1_x: batch_xs1, tower1_y_: batch_ys1, keep_prob: 0.5})

        with tf.Session(target=server.target
                        ) as sess:
            # Test trained model
            print(sess.run(accuracy, feed_dict={tower1_x: mnist.test.images,
                                                tower1_y_: mnist.test.labels,
                                                keep_prob: 1.0}))

        print('yay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
