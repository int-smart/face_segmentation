import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
slim = tf.contrib.slim
import os
import urllib
import matplotlib.image as mpimg
from PIL import Image


IMG_PATH = '/home/divyansh/PycharmProjects/545-Project/PASCAL_VOC_2012_EA/IMG'
GT_PATH = '/home/divyansh/PycharmProjects/545-Project/PASCAL_VOC_2012_EA/GT'
SEG_IMG_PATH = '/home/divyansh/PycharmProjects/545-Project/fcn.berkeleyvision.org/data/pascal/seg11valid.txt'


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

def load_weights(session, weights_url='https://umich.box.com/shared/static/81kicsu5t0u2pybjik0d7v13brlualxi.npy',
                 weights_download_dir='../data'):
    try:
        os.mkdir(weights_download_dir)
    except OSError:
        pass
    if not os.path.isfile(weights_download_dir + '/fcn8s_weights.npy'):
        print('Downloading weights for FCN-8s network')
        urllib.request.urlretrieve(weights_url, weights_download_dir + '/fcn8s_weights.npy')

    weights = np.load(weights_download_dir + '/fcn8s_weights.npy').item()
    print('Updating the weights:')
    for key, value in weights.items():
        tensor = slim.get_variables_by_name(key)
        if tensor:
            tf.assign(tensor[0], value).op.run(session=session)
'''
def intersection_over_union(ground_truth, prediction):
    iou = ((np.logical_and(ground_truth,prediction)).astype(int)).sum()/((np.logical_or(ground_truth,prediction)).astype(int)).sum()
    return iou
'''
def pixelAccuracy(y_pred, y_true):
    y_pred = np.argmax(np.reshape(y_pred), axis=2)
    y_pred = y_pred
    return 1.0 * np.sum((y_pred == y_true)) / np.sum(y_true != 255)

def compute_MIoU(y_pred_batch, y_true_batch):
    return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))]))


def load_data(image, IMAGE_PATH = IMG_PATH, GROUNDT_PATH = GT_PATH):
    img = Image.open(IMAGE_PATH + '/' + image.rstrip('\n') + ".jpg")
    img = img.resize((500, 500))
    input_img = np.array(img, dtype=np.float32)
    input_img = input_img[:, :, ::-1]
    input_img -= np.array((104.00698793, 116.66876762, 122.67891434))
    input_img = np.expand_dims(input_img, axis=0)

    # gt = np.expand_dims(mpimg.imread(GT_PATH + '/' + image.rstrip('\n') + ".png"), axis= 0)
    gt = Image.open(GROUNDT_PATH + '/' + image.rstrip('\n') + ".png")
    gt = gt.resize((500, 500))
    gt = np.array(gt, dtype=np.uint8)
    gt = gt[np.newaxis, ...]
    return input_img,gt

def test(list_image, sess, prediction, endpoints, input):
    batch_size = len(list_image)
    list_iou = np.zeros([batch_size,1])
    fin_prediction = []
    # fin_prediction = np.zeros([batch_size, 500,500])
    # fin_gt = np.zeros([batch_size, 500, 500])
    # inp_images = np.zeros([batch_size,500,500,3])
    # miou = 0
    i = 0

    for each in list_image:
        inp_img, gt = load_data(each)
        image_pred = sess.run(prediction, feed_dict={input: inp_img})
        output = np.argmax(image_pred, axis=3).reshape((500, 500))
        fin_prediction.append(output)
        list_iou[i,:] = compute_MIoU(output,gt)
        i += 1
    mIoU = sum(list_iou)/len(list_iou)
    return list_iou,mIoU,fin_prediction


def main():
    train_log_dir = "./train_log_dir/"
    if not tf.gfile.Exists(train_log_dir):
        tf.gfile.MakeDirs(train_log_dir)

    #Creating a list of validation images
    seg_img = open(SEG_IMG_PATH, 'r')
    list_valimg = []
    for each in seg_img:
        list_valimg.append(each)

    # Start session, setup placeholders and load weights.
    sess1 = tf.Session()
    inp = tf.placeholder(dtype=tf.float32, shape=[1, 500, 500, 3])
    pred, endpoint = fcn_8s(inp)
    load_weights(sess1)

    # Call the test function to load images, run it through the net and calculate IoU values.
    list_iou, miou, list_pred = test(list_valimg, sess= sess1, prediction= pred, endpoints=endpoint, input = inp)
    print(list_iou, miou, list_pred)

if __name__=='__main__':
    main()
