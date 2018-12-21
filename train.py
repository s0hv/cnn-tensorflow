import argparse
import os
import time
from math import ceil

import tensorflow as tf
import numpy as np
import dataset
from six.moves import cPickle
import re


# https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
    tf.nn.zero_fraction(x))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    # We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    # Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    # Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    # Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer, weights, biases


def train(args):
    batch_size = args.batch_size

    # Prepare input data
    classes = os.listdir(args.data_dir)

    # 20% of the data will automatically be used for validation
    validation_size = 0.2
    # Multiples of 2 please
    img_size = args.image_size
    num_channels = 3
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.init_from:
        with open(os.path.join(args.init_from, 'labels.cpkl'), 'rb') as f:
            labels = cPickle.load(f)

            for label in classes:
                if label not in labels:
                    labels.append(label)

            classes = labels

        with open(os.path.join(args.init_from, 'config.cpkl'), 'rb') as f:
            old_args = cPickle.load(f)
            need_to_be_same = ['img_size']
            assert all([getattr(args, x) == getattr(old_args, x) for x in need_to_be_same])

    num_classes = len(classes)

    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    train_data, valid_data, filecount = dataset.read_train_sets(args.data_dir, img_size, classes, validation_size=validation_size, max_size=10000)

    with open(os.path.join(args.save_dir, 'labels.cpkl'), 'wb') as f:
        cPickle.dump(classes, f)

    with open(os.path.join(args.save_dir, 'config.cpkl'), 'wb') as f:
        cPickle.dump(args, f)

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels],
                           name='x')

        # labels
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        y_true_cls = tf.argmax(y_true, axis=1)

        # Network graph params
        filter_size_conv1 = 3
        num_filters_conv1 = 32

        filter_size_conv2 = 3
        num_filters_conv2 = 64

        filter_size_conv3 = 3
        num_filters_conv3 = 128

        fc_layer_size = img_size

        layer_conv1 = create_convolutional_layer(input=x,
                                                 num_input_channels=num_channels,
                                                 conv_filter_size=filter_size_conv1,
                                                 num_filters=num_filters_conv1)

        _activation_summary(layer_conv1)

        layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                                 num_input_channels=num_filters_conv1,
                                                 conv_filter_size=filter_size_conv2,
                                                 num_filters=num_filters_conv2)

        layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                                 num_input_channels=num_filters_conv2,
                                                 conv_filter_size=filter_size_conv3,
                                                 num_filters=num_filters_conv3)

        layer_flat = create_flatten_layer(layer_conv3)

        layer_fc1, a, b = create_fc_layer(input=layer_flat,
                                          num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                          num_outputs=fc_layer_size,
                                          use_relu=True)
        _activation_summary(layer_fc1)

        layer_fc2, wei, bia = create_fc_layer(input=layer_fc1,
                                              num_inputs=fc_layer_size,
                                              num_outputs=num_classes,
                                              use_relu=False)
        _activation_summary(layer_fc2)

        y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

        y_pred_cls = tf.argmax(y_pred, axis=1)
        sess.run(tf.global_variables_initializer())
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                                   labels=y_true)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(cost)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())

        tf.summary.scalar('train_loss', cost)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        saver = tf.train.Saver(tf.global_variables())

        if args.init_from:
            print(f'Restoring save from folder {args.init_from}')
            saver.restore(sess, tf.train.latest_checkpoint(args.init_from))

        num_batches = ceil(filecount/batch_size)

        def show_progress(epoch, loss, batch, duration):
            print(f"Training Epoch {epoch}/{args.num_epochs} Batch: {batch}/{num_batches} Loss: {loss:.3f}, Time: {duration:.02f}s")

        train_data = train_data.batch(batch_size)
        iterator = train_data.make_initializable_iterator()
        next_element = iterator.get_next()

        valid_data = valid_data.batch(batch_size)
        valid_data = valid_data.repeat()
        valid_iterator = valid_data.make_initializable_iterator()
        next_valid = valid_iterator.get_next()
        sess.run(valid_iterator.initializer)

        for epoch in range(args.num_epochs):
            sess.run(iterator.initializer)
            batch = 1
            while True:
                start = time.time()
                try:
                    x_batch, y_true_batch = sess.run(next_element)
                except tf.errors.OutOfRangeError:
                    acc = sess.run(accuracy, feed_dict=feed_dict_tr)
                    val_acc = sess.run(accuracy, feed_dict=feed_dict_val)
                    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}"
                    print(msg.format(epoch + 1, acc, val_acc))

                    break

                x_valid_batch, y_valid_batch = sess.run(next_valid)

                feed_dict_tr = {x: x_batch,
                                y_true: y_true_batch}
                feed_dict_val = {x: x_valid_batch,
                                 y_true: y_valid_batch}

                sess.run(optimizer, feed_dict=feed_dict_tr)
                val_loss = sess.run(cost, feed_dict=feed_dict_val)
                # This speeds up training by a lot
                #writer.add_summary(summ, epoch*num_batches + batch)
                show_progress(epoch, val_loss, batch, time.time() - start)

                if (epoch * num_batches + batch) % args.save_every == 0\
                        or (epoch == args.num_epochs-1 and batch == num_batches):
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    checkpoint_path = saver.save(sess, checkpoint_path, global_step=epoch*num_batches+batch)
                    print(f'Saved checkpoint {checkpoint_path}')

                batch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=70,
                        help='Size of each batch')

    parser.add_argument('--image-size', type=int, default=128,
                        help='Size that all images will be resized to')

    parser.add_argument('--save-dir', type=str, default='model',
                        help='The directory where the models are saved')

    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for tensorboard logs')

    parser.add_argument('--save-every', type=int, default=1000,
                        help="How many passes have to be between each checkpoint")

    parser.add_argument('--init-from', type=str, default=None,
                        help="How many passes have to be between each checkpoint")

    parser.add_argument('--continue', action='store_true', dest='continue_',
                        help='If this is set will continue from last save and restore all parameters.\n'
                             'This means every setting you use with this will be ignored except init-from.')

    parser.add_argument('--data-dir', type=str, default='data2',
                        help='Directory where the data is located in.\n'
                             'Directory must have subdirectories that include the training pics')

    parser.add_argument('--num-epochs', type=int, default=50,
                        help='number of epochs. Number of full passes through the training examples.')

    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='The learning rate for the optimizer')

    args = parser.parse_args()

    train(args)
