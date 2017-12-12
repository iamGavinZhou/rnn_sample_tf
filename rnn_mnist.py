#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def initialize_weight_bias(in_size, out_size):
    weight = tf.truncated_normal(shape=(in_size, out_size), stddev=0.01, mean=0.0)
    bias = tf.constant(0.1, shape=[out_size])
    return tf.Variable(weight), tf.Variable(bias)


def model(data, target, dropout, num_hidden=200, num_layers=3):
    """
    RNN model for mnist classification.
    Args:
        data: input data with shape (batch_size, max_time_steps, cell_size).
        target : label of input data with shape (batch_size, num_classes).
        dropout: dropout rate.
        num_hidden: the number of hidden units.
        num_layers: the number of RNN layers.

    Returns:

    """
    # establish RNN model
    cells = list()
    for _ in range(num_layers):
        cell = tf.nn.rnn_cell.GRUCell(num_units=num_hidden)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=1.0-dropout)
        cells.append(cell)
    network = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
    outputs, last_state = tf.nn.dynamic_rnn(cell=network, inputs=data, dtype=tf.float32)

    # get last output
    outputs = tf.transpose(outputs, (1, 0, 2))
    last_output = tf.gather(outputs, int(outputs.get_shape()[0])-1)

    # add softmax layer
    out_size = int(target.get_shape()[1])
    weight, bias = initialize_weight_bias(in_size=num_hidden, out_size=out_size)
    logits = tf.add(tf.matmul(last_output, weight), bias)

    return logits


def main():
    # define some parameters
    default_epochs = 10
    default_batch_size = 64
    default_dropout = 0.5
    test_freq = 150  # every 150 batches
    logs_path = 'data/log'

    # get train and test data
    mnist_data = input_data.read_data_sets('data/mnist', one_hot=True)
    total_steps = int(mnist_data.train.num_examples/default_batch_size)
    total_test_steps = int(mnist_data.test.num_examples/default_batch_size)
    print('number of training examples: %d' % mnist_data.train.num_examples)  # 55000
    print('number of test examples: %d' % mnist_data.test.num_examples)  # 10000

    # fit RNN model
    input_x = tf.placeholder(tf.float32, shape=(None, 28, 28))
    input_y = tf.placeholder(tf.float32, shape=(None, 10))
    dropout = tf.placeholder(tf.float32)
    input_logits = model(input_x, input_y, dropout)

    # define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=input_logits, labels=input_y))
    train_op = tf.train.RMSPropOptimizer(0.001).minimize(loss)
    input_prob = tf.nn.softmax(input_logits)
    error_count = tf.not_equal(tf.arg_max(input_prob, 1), tf.arg_max(input_y, 1))
    error_rate_op = tf.reduce_mean(tf.cast(error_count, tf.float32))

    # add summary
    tf.summary.scalar('error_rate', error_rate_op)
    tf.summary.scalar('loss', loss)
    merge_summary_op = tf.summary.merge_all()

    # train and test
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(logdir=logs_path, graph=tf.get_default_graph())
        # train
        for epoch in range(default_epochs):
            for step in range(total_steps):
                train_x, train_y = mnist_data.train.next_batch(default_batch_size)
                train_x = train_x.reshape(-1, 28, 28)
                feed_dict = {input_x: train_x,
                             input_y: train_y,
                             dropout: default_dropout}
                _, summary = session.run([train_op, merge_summary_op], feed_dict=feed_dict)
                # write logs
                summary_writer.add_summary(summary, global_step=epoch*total_steps+step)

                # test
                if step > 0 and (step % test_freq == 0):
                    avg_error = 0
                    for test_step in range(total_test_steps):
                        test_x, test_y = mnist_data.test.next_batch(default_batch_size)
                        test_x = test_x.reshape(-1, 28, 28)
                        feed_dict = {input_x: test_x,
                                     input_y: test_y,
                                     dropout: 0}
                        test_error = session.run(error_rate_op, feed_dict=feed_dict)
                        avg_error += test_error / total_test_steps
                    print('epoch: %d, steps: %d, avg_test_error: %.4f' % (epoch, step, avg_error))

if __name__ == '__main__':
    main()
