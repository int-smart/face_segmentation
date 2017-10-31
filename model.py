import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
slim = tf.contrib.slim
import os
import urllib
import matplotlib.pyplot as plt
from PIL import Image

IMG_PATH = './data/VOC2012/JPEGImages'
GT_PATH = './data/VOC2012/SegmentationClass'
SEG_IMG_PATH = './data/VOC2012/seg11valid.txt'
error_analysis_path = './data/error_analysis'

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
                 weights_download_dir='./model_files/'):
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


def pixelAccuracy(y_pred, y_true):
    return 1.0 * np.sum((y_pred == y_true) * (y_true > 0)) / np.sum(((y_true+y_pred) < 255) * ((y_true+y_pred) > 0))


def pixelInterPerClass(y_pred, y_true, sc_dict):
    classes = np.unique(y_pred)
    inter_k = np.zeros(classes.shape[0],)
    union_k = np.zeros(classes.shape[0],)
    pred_pixels_k = np.zeros(classes.shape[0],)
    for i, val in enumerate(classes):
        if val != 0:
            inter_k[i] = np.sum(np.logical_and(y_pred == y_true, y_pred == val))
            union_k[i] = np.sum(np.logical_or(y_pred == val, y_true == val))
            pred_pixels_k[i] = np.sum(y_pred == val)

    y_pred_sc = np.copy(y_pred.flatten())
    y_true_sc = np.copy(y_true.flatten())
    y_pred_sc = np.array([sc_dict[i] for i in y_pred_sc])
    y_true_sc = np.array([sc_dict[i] for i in y_true_sc])

    inter_sc = np.zeros(classes.shape[0],)
    inter_dc = np.zeros(classes.shape[0],)

    for i, val in enumerate(classes):
        if val != 0:
            inter_sc[i] = np.sum(np.logical_and((y_pred_sc == y_true_sc), (y_pred == val).flatten()))
            inter_dc[i] = np.sum(np.logical_and((y_pred == val).flatten(), (y_true_sc != sc_dict[val]) *
                                             (y_true_sc != 255) * (y_true_sc != 0)))
    return classes, inter_k, union_k, pred_pixels_k, inter_sc, inter_dc


def compute_MIoU(y_pred_batch, y_true_batch):
    return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))]))


def load_data(image, IMAGE_PATH = IMG_PATH, GROUNDT_PATH = GT_PATH):
    img = Image.open(IMAGE_PATH + '/' + image.rstrip('\n') + ".jpg")
    img = img.resize((500, 500))
    input_img = np.array(img, dtype=np.float32)
    input_img = input_img[:, :, ::-1]
    input_img -= np.array((104.00698793, 116.66876762, 122.67891434))
    input_img = np.expand_dims(input_img, axis=0)
    gt = Image.open(GROUNDT_PATH + '/' + image.rstrip('\n') + ".png")
    gt = gt.resize((500, 500))
    gt = np.array(gt, dtype=np.uint8)
    # gt = gt[np.newaxis, ...]
    return input_img,gt


def test_per_class(list_image, write_out=True, tf_use=False, prediction=None, input=None, sess=None):
    batch_size = len(list_image)
    list_iou = np.zeros((batch_size,))
    i = 0
    fin_prediction = np.zeros((batch_size, 500, 500), dtype=np.uint8)
    if not tf_use:
        fin_prediction = np.load('./results/fin_prediction.npy')

    if write_out:
        outfile = open('./results/false_positive_outputs.txt', 'w+')
        outfile.write('filename class inter_k union_k pred_pixels_k inter_sc inter_dc\n')

    sc_dict = create_sc_dict()
    interk, union_k, pred_pixels_k, inter_sc, inter_dc = [], [], [], [], []
    for index, each in enumerate(list_image):
        inp_img, gt = load_data(each)
        if tf_use:
            image_pred = sess.run(prediction, feed_dict={input: inp_img})
            output = np.argmax(image_pred, axis=3).reshape((500, 500))
            fin_prediction[index] = output

        classes, interk_temp, unionk_temp, pred_pixels_k_temp, inter_sc_temp, inter_dc_temp = pixelInterPerClass(fin_prediction[index], gt, sc_dict)
        interk.append(interk_temp)
        union_k.append(unionk_temp)
        pred_pixels_k.append(pred_pixels_k_temp)
        inter_sc.append(inter_sc_temp)
        inter_dc.append(inter_dc_temp)
        if write_out:
            for ind, interk_val in enumerate(interk_temp):
                if classes[ind] != 0:
                    outfile.write('{}.jpg {} {} {} {} {} {}\n'.format(each[:-1],
                                                                    classes[ind],
                                                                    interk_val,
                                                                    unionk_temp[ind],
                                                                    pred_pixels_k_temp[ind],
                                                                    inter_sc_temp[ind],
                                                                    inter_dc_temp[ind]))
    outfile.close()
    return interk, union_k, pred_pixels_k, inter_sc, inter_dc


def load_data(image_path, ground_truth_path):
    img = Image.open(image_path)
    img = img.resize((500, 500))
    input_img = np.array(img, dtype=np.float32)
    input_img = input_img[:, :, ::-1]
    input_img -= np.array((104.00698793, 116.66876762, 122.67891434))
    input_img = np.expand_dims(input_img, axis=0)
    gt = Image.open(ground_truth_path)
    gt = gt.resize((500, 500))
    gt = np.array(gt, dtype=np.uint8)
    return input_img, gt


def test(list_image_paths, list_gt_paths, write_out=True, iou_path=None, pred_path=None,
         prediction=None, input=None, sess=None):
    batch_size = len(list_image_paths)
    list_iou = np.zeros((batch_size,))
    fin_prediction = np.zeros((batch_size, 500, 500), dtype=np.uint8)
    for index, each in enumerate(list_image_paths):
        inp_img, gt = load_data(each, list_gt_paths[index])
        image_pred = sess.run(prediction, feed_dict={input: inp_img})
        output = np.argmax(image_pred, axis=3).reshape((500, 500))
        fin_prediction[index] = output
        list_iou[index] = pixelAccuracy(output, gt)

    if write_out:
        np.save(iou_path + '/iou.npy', list_iou)
        np.save(pred_path + '/prediction.npy', fin_prediction)
    return np.mean(list_iou)


def main(image_dir, gt_dir, sess, pred, inp):
    list_valimg = [os.path.join(image_dir, i) for i in os.listdir(image_dir)]
    list_gt_paths = [os.path.join(gt_dir, i) for i in os.listdir(gt_dir)]
    val = test(list_valimg, list_gt_paths, True, image_dir, image_dir, pred, inp, sess)
    return val

def create_outputFile():
    list_valimg = open(SEG_IMG_PATH, 'r').readlines()
    iou_temp = np.load('results/updated_iou_list.npy')
    outfile = open('./results/iou_outputs.txt', 'w+')
    outfile.write('filename mean_iou \n')
    for i, val in enumerate(iou_temp):
        outfile.write('{}.jpg {}\n'.format(list_valimg[i][:-1], iou_temp[i]))
    outfile.close()


def create_sc_dict():
    keys = np.arange(21)
    values = [0, 1, 2, 1, 3, 7, 3,
              3, 5, 6, 5, 6, 5, 5,
              2, 5, 7, 5, 6, 3, 21]
    sc_dict = dict(zip(keys, values))
    sc_dict[255] = 255
    return sc_dict


if __name__=='__main__':
    # Start session, setup placeholders and load weights.
    sess1 = tf.Session()
    inp = tf.placeholder(dtype=tf.float32, shape=[1, 500, 500, 3])
    pred, endpoint = fcn_8s(inp)
    load_weights(sess1)

    error_analysis_path = './data/error_analysis'

    # Original images
    val = main(error_analysis_path+'/orig_image', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for original_images is {}'.format(val))


    # Blurring images
    blurring_path = error_analysis_path+'/blurring'
    # 1_gaussian_blur
    val = main(blurring_path+'/1_gaussian_blur', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for 1_gaussian_blur is {}'.format(val))

    # 2_gaussian_blur
    val = main(blurring_path+'/2_gaussian_blur', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for 2_gaussian_blur is {}'.format(val))

    # half_gaussian_blur
    val = main(blurring_path+'/half_gaussian_blur', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for half_gaussian_blur is {}'.format(val))

    # Effect of lens flare
    flare_path = error_analysis_path + '/lens_flare'
    # 100
    val = main(flare_path+'/100', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for lens flare 100 is {}'.format(val))

    # 125
    val = main(flare_path+'/125', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for lens flare 125 is {}'.format(val))

    # 150
    val = main(flare_path+'/150', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for lens flare 150 is {}'.format(val))

    # 175
    val = main(flare_path+'/175', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for lens flare 175 is {}'.format(val))

    # Effect of rotation
    rotation_path = error_analysis_path+'/rotation'
    # 90
    val = main(rotation_path+'/90', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for rotation 90 is {}'.format(val))

    # 180
    val = main(rotation_path+'/180', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for rotation 180 is {}'.format(val))

    # 270
    val = main(rotation_path+'/270', error_analysis_path+'/gt', sess1, pred, inp)
    print('mean_iou for rotation 270 is {}'.format(val))

    # Effect of vignetting
    val = main(error_analysis_path + '/vignetting', error_analysis_path + '/gt', sess1, pred, inp)
    print('mean_iou for vignetting is {}'.format(val))