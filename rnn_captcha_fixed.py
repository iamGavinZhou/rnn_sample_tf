#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function
import os
import string
import random
import tensorflow as tf
import numpy as np
from captcha.image import ImageCaptcha
from scipy.misc import imread, imresize
from rnn_mnist import initialize_weight_bias

chars_length = 4
img_width = 160
img_height = 60
captcha_save_dir = 'data/captcha'
chars_list = list(string.ascii_lowercase + string.digits)
image = ImageCaptcha(width=img_height, height=img_width)


def gen_captcha(captcha_num=50000):
    """
    Generate Captcha images.
    Returns:

    """
    if not os.path.exists(captcha_save_dir):
        os.makedirs(captcha_save_dir)
    random.seed(100)
    for i in range(captcha_num):
        # generate captcha content
        chars_index = [random.randint(0, len(chars_list)-1) for _ in range(chars_length)]
        chars = ''
        for index in chars_index:
            chars += chars_list[index]
        save_path = os.path.join(captcha_save_dir, chars+'.png')
        image.write(chars, save_path)


def read_img(path):
    im = imread(path, mode='RGB')
    width, height = im.shape[0], im.shape[1]
    if (width, height) != (img_height, img_width):
        im = imresize(im, (img_height, img_width))
    im = im / 255.
    im = np.clip(im, 0, 1)
    im = im.reshape((1, img_height, img_width, 3))
    im = im.astype(np.float32)

    return im


def get_label(path):
    label = np.zeros((chars_length, len(chars_list)), dtype=np.uint8)
    file_name = os.path.basename(path)
    label_str = file_name.split('.')[0].strip().lower()
    assert len(label_str) == chars_length
    for char_index, char_seq in enumerate(label_str):
        label[char_index, chars_list.index(char_seq)] = 1
    return label


def batch_generator(data_dir, batch_size=16, shuffle=False):
    files = os.listdir(data_dir)

    def load_imgs(names):
        batch_imgs = []
        for name in names:
            batch_imgs.append(read_img(os.path.join(data_dir, name)))
        return np.concatenate(batch_imgs, 0)

    def load_label(names):
        batch_labels = []
        for name in names:
            batch_labels.append(get_label(os.path.join(data_dir, name)))
        return np.concatenate(batch_labels, 0)

    indices = np.arange(len(files))
    if shuffle is True:
        np.random.shuffle(indices)
    for start_idx in range(0, len(files) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        temp_names = []
        for x in excerpt:
            temp_names.append(files[x])
        yield load_imgs(temp_names), load_label(temp_names)


def model(data, target, dropout, is_training=True, num_hidden=200, num_layers=3):
    """
    RNN model for sequence labeling.
    Args:
        data: input data with shape (batch_size, img_width, img_height, img_channel).
        target: data label with shape (batch_size, max_length, num_classes).
        dropout: dropout rate.
        is_training: indicate training or test stage for using batch normalization.
        num_hidden: number of hidden units.
        num_layers: number of rnn layer.

    Returns:

    """
    # establish CNN model
    conv1 = tf.layers.conv2d(data, filters=4, kernel_size=3, padding='same', activation=None, use_bias=True)
    conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=is_training))
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=1)

    conv2 = tf.layers.conv2d(pool1, filters=8, kernel_size=3, padding='same', activation=None, use_bias=True)
    conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=is_training))
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=1)

    conv3 = tf.layers.conv2d(pool2, filters=8, kernel_size=3, padding='same', activation=None, use_bias=True)
    conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=is_training))
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=1)

    logits = tf.contrib.layers.fully_connected(pool3, num_outputs=1024, activation_fn=None)  # (batch_size, 1024)
    logits = tf.reshape(logits, (-1, chars_length, 1024//chars_length))  # (batch_size, max_length, unit_length)

    # establish RNN model
    cells = list()
    for _ in range(num_layers):
        cell = tf.nn.rnn_cell.GRUCell(num_units=num_hidden)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=1.0 - dropout)
        cells.append(cell)
    network = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
    outputs, last_state = tf.nn.dynamic_rnn(cell=network, inputs=logits, dtype=tf.float32)

    max_length = int(target.get_shape()[1])
    num_classes = int(target.get_shape()[2])
    weight, bias = initialize_weight_bias(in_size=num_hidden, out_size=num_classes)

    # flatten to apply same weights to all time steps
    outputs = tf.reshape(outputs, (-1, num_hidden))  # (batch_size*max_length, num_hidden)
    logits = tf.add(tf.matmul(outputs, weight), bias)  # (batch_size*max_length, num_classes)
    logits = tf.reshape(logits, (-1, max_length, num_classes))  # (batch_size, max_length, num_classes)

    return logits


def main():
    # generate training captcha images
    # gen_captcha()

    # training parameters
    batch_size = 32
    epochs = 20
    train_dropout = 0.5
    model_save_freq = 2
    total_steps = len(os.listdir('data/captcha')) // batch_size
    logs_path = 'data/logs/2'
    model_path = 'data/model/2'

    # fit model
    input_x = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, 3))
    input_y = tf.placeholder(tf.float32, shape=(batch_size, chars_length, len(chars_list)))
    dropout = tf.placeholder(tf.float32)
    input_logits = model(input_x, input_y, dropout, is_training=True)

    # define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(input_logits, input_y))
    train_op = tf.train.RMSPropOptimizer(0.001).minimize(loss)
    input_prob = tf.nn.softmax(input_logits)
    error_count = tf.not_equal(tf.arg_max(input_prob, 1), tf.argmax(input_y, 1))
    error_rate_op = tf.reduce_mean(tf.cast(error_count, tf.float32))

    # add summary
    tf.summary.scalar('error_rate', error_rate_op)
    tf.summary.scalar('loss', loss)
    merge_summary_op = tf.summary.merge_all()

    # training
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(logdir=logs_path, graph=session.graph, flush_secs=5)
        saver = tf.train.Saver(tf.all_variables())

        for epoch in range(epochs):
            batch_gen = batch_generator('data/captcha', batch_size=batch_size, shuffle=True)
            step = 0
            # save model
            if epoch > 0 and epoch % model_save_freq == 0:
                saver.save(session, save_path=os.path.join(model_path, 'sl_epoch%d.ckpt' % epoch))

            for train_x, train_y in batch_gen:
                feed_dict = {input_x: train_x,
                             input_y: train_y,
                             dropout: train_dropout}
                _, summary = session.run([train_op, merge_summary_op], feed_dict=feed_dict)
                # write logs
                summary_writer.add_summary(summary, global_step=epoch * total_steps + step)
                step += 1

if __name__ == '__main__':
    main()
