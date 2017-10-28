import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
slim = tf.contrib.slim


def fcn_8s(input, num_classes=21, is_training=False, scope='fcn_8s'):

    with tf.variable_scope(scope):
        end_points_collection = scope + '_endpoints'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu
                            ):
            with slim.arg_scope([slim.conv2d,
                                 slim.fully_connected,
                                 slim.max_pool2d,
                                 slim.conv2d_transpose],
                                outputs_collections=end_points_collection,
                                padding='SAME'):
                with slim.arg_scope([slim.conv2d_transpose], padding='VALID', biases_initializer=None):
                    net = tf.pad(input, [[0, 0], [100, 100], [100, 100], [0, 0]], mode="CONSTANT", constant_values=0.0)
                    net = slim.conv2d(net, 64, [3, 3], padding='VALID', scope='conv1_1')
                    net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    pool3 = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    pool4 = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')
                    net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = slim.conv2d(net, num_classes, [1, 1], scope='score_fr', activation_fn=None)
                    net = slim.conv2d_transpose(net, num_classes, [4, 4], 2, scope='upscore2', activation_fn=None)
                    pool4_fcn = slim.conv2d(pool4, num_classes, [1, 1], scope='score_pool4', activation_fn=None)
                    size_to_slice = net.get_shape().as_list()
                    size_to_slice[0] = pool4_fcn.get_shape().as_list()[0]
                    pool4_fcn = tf.slice(pool4_fcn, [0, 5, 5, 0], net.get_shape().as_list())
                    net = tf.add(net, pool4_fcn, name='fuse_pool4')
                    net = slim.conv2d_transpose(net, num_classes, [4, 4], 2, scope='upscore_pool4', activation_fn=None)
                    pool3_fcn = slim.conv2d(pool3, num_classes, [1, 1], scope='score_pool3', activation_fn=None)
                    pool3_fcn = tf.slice(pool3_fcn, [0, 9, 9, 0], net.get_shape().as_list())
                    net = tf.add(net, pool3_fcn, name='fuse_pool3')
                    net = slim.conv2d_transpose(net, num_classes, [16, 16], 8, scope='upscore8', activation_fn=None)
                    size_to_slice = input.get_shape().as_list()
                    size_to_slice[0], size_to_slice[3] = net.get_shape().as_list()[0], net.get_shape().as_list()[3]
                    net = tf.slice(net, [0, 31, 31, 0], size_to_slice, name='score')
                    tf.add_to_collection(end_points_collection, net)
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                return net, end_points


def fcn_16s(input, num_classes=21, is_training=False, scope='fcn_16s'):
    with tf.variable_scope(scope):
        end_points_collection = scope + '_endpoints'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu
                            ):
            with slim.arg_scope([slim.conv2d,
                                 slim.fully_connected,
                                 slim.max_pool2d,
                                 slim.conv2d_transpose],
                                outputs_collections=end_points_collection,
                                padding='SAME'):
                with slim.arg_scope([slim.conv2d_transpose], padding='VALID', biases_initializer=None):
                    net = tf.pad(input, [[0, 0], [100, 100], [100, 100], [0, 0]], mode="CONSTANT", constant_values=0.0)
                    net = slim.conv2d(net, 64, [3, 3], padding='VALID', scope='conv1_1')
                    net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    pool3 = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    pool4 = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')
                    net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = slim.conv2d(net, num_classes, [1, 1], scope='score_fr', activation_fn=None)
                    net = slim.conv2d_transpose(net, num_classes, [4, 4], 2, scope='upscore2', activation_fn=None)
                    pool4_fcn = slim.conv2d(pool4, num_classes, [1, 1], scope='score_pool4', activation_fn=None)
                    size_to_slice = net.get_shape().as_list()
                    size_to_slice[0] = pool4_fcn.get_shape().as_list()[0]
                    pool4_fcn = tf.slice(pool4_fcn, [0, 5, 5, 0], net.get_shape().as_list())
                    net = tf.add(net, pool4_fcn, name='fuse_pool4')
                    net = slim.conv2d_transpose(net, num_classes, [32, 32], 16, scope='upscore16', activation_fn=None)
                    size_to_slice = input.get_shape().as_list()
                    size_to_slice[0], size_to_slice[3] = net.get_shape().as_list()[0], net.get_shape().as_list()[3]
                    net = tf.slice(net, [0, 27, 27, 0], size_to_slice, name='score')
                    tf.add_to_collection(end_points_collection, net)
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                return net, end_points

def fcn_32s(input, num_classes=21, is_training=False, scope='fcn_32s'):
    with tf.variable_scope(scope):
        end_points_collection = scope + '_endpoints'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu
                            ):
            with slim.arg_scope([slim.conv2d,
                                 slim.fully_connected,
                                 slim.max_pool2d,
                                 slim.conv2d_transpose],
                                outputs_collections=end_points_collection,
                                padding='SAME'):
                with slim.arg_scope([slim.conv2d_transpose], padding='VALID', biases_initializer=None):
                    net = tf.pad(input, [[0, 0], [100, 100], [100, 100], [0, 0]], mode="CONSTANT", constant_values=0.0)
                    net = slim.conv2d(net, 64, [3, 3], padding='VALID', scope='conv1_1')
                    net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    pool3 = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    pool4 = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')
                    net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = slim.conv2d(net, num_classes, [1, 1], scope='score_fr', activation_fn=None)
                    net = slim.conv2d_transpose(net, num_classes, [64, 64], 32, scope='upscore', activation_fn=None)
                    size_to_slice = input.get_shape().as_list()
                    size_to_slice[0], size_to_slice[3] = net.get_shape().as_list()[0], net.get_shape().as_list()[3]
                    net = tf.slice(net, [0, 19, 19, 0], size_to_slice, name='score')
                    tf.add_to_collection(end_points_collection, net)
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                return net, end_points