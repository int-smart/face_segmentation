import tensorflow as tf
from PIL import Image
import numpy as np
from scipy import misc
import urllib
import os
import _pickle as cPickle
import gzip


def get_test_data(test_label_dir, test_image_dir):
    images_labels = []
    for file in os.listdir(test_label_dir):
        path_to_label = test_label_dir + file
        if os.path.isfile(path_to_label):
            if '.ppm' in file:
                file = file.replace('.ppm', '')
                file = file.split('_')
                file_number = file[-1]
                file_name = file[:-1]
                file_name = '_'.join(file_name)
                dir_name = file_name
                file_name = file_name + '_' + str(file_number) + image_type
            label = Image.open(path_to_label)
            label = label.resize((500, 500))
            label = np.array(label, dtype=np.int32)
            label = label[:, :, 1]

            path_to_image = test_image_dir + dir_name + '/' + file_name
            image = Image.open(path_to_image)
            image = image.resize((500, 500))
            image = np.array(image, dtype=np.float32)
            # image_scipy = misc.imread('/media/abhishek/B67A61587A611701/Users/Second/Desktop/Research/FCN/parts_lfw_funneled_gt_images/Aaron_Peirsol_0001.ppm')
            # print(image_scipy.shape)
            # plt.imshow(image_scipy[:,:,1])
            # plt.show()
            images_labels.append([image, label])
        break

def get_test_data_pascal_voc(test_label_dir, test_image_dir):
    images_labels = []
    for file in os.listdir(test_label_dir):
        path_to_label = test_label_dir + file
        if os.path.isfile(path_to_label):
            if '.png' in file:
                file_name = file.replace('.png', '')
            label = Image.open(path_to_label)
            label = label.resize((500, 500))
            label = np.array(label, dtype=np.int32)
            print(label.shape)
#            label = label[:, :, 1]

            path_to_image = test_image_dir + file_name +".jpg"
            image = Image.open(path_to_image)
            image = image.resize((500, 500))
            image = np.array(image, dtype=np.float32)
            # image_scipy = misc.imread('/media/abhishek/B67A61587A611701/Users/Second/Desktop/Research/FCN/parts_lfw_funneled_gt_images/Aaron_Peirsol_0001.ppm')
            # print(image_scipy.shape)
            # plt.imshow(image_scipy[:,:,1])
            # plt.show()
            images_labels.append([image, label])
        break


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
                 weights_download_dir='./data'):
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

def compute_MIoU(y_pred_batch, y_true_batch):
    return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))]))

def pixelAccuracy(y_pred, y_true):
    y_pred = np.argmax(np.reshape(y_pred), axis=2)
    y_pred = y_pred
    return 1.0 * np.sum((y_pred == y_true)) / np.sum(y_true != 255)


if __name__ == '__main__':
    IOU = []
    filename = '2007_000033'
    img = Image.open('./pascal_voc/'+filename+'.jpg')
    img = img.resize((500, 500))
    input_img = np.array(img, dtype=np.float32)
    input_img = input_img[:, :, ::-1]
    input_img = np.expand_dims(input_img, axis=0)
    input_img -= np.array((104.00698793, 116.66876762, 122.67891434))

    sess = tf.Session()
    input = tf.placeholder(dtype=tf.float32, shape=[1, 500, 500, 3])
    pred, endpoints = fcn_8s(input)
    load_weights(sess)


    image_pred = sess.run(pred, feed_dict={input: input_img})
    output = np.argmax(image_pred, axis=3).reshape((500, 500))


    for file in os.listdir(test_label_dir):
        if filename in file:
            label = Image.open(test_label_dir + filename + '.png')
            label = label.resize((500, 500))
            # input_label = np.array(img, dtype=np.float32)
            # input_label = input_label[:, :, ::-1]
            # input_label = np.expand_dims(input_img, axis=0)

    print("Done with the image")

    iou = intersection_over_union(label, output)
    a = np.load('./data/VOC2012/SegmentationClass/2007_000032.png')
    print("The iou is ", iou)
    print(image_pred.shape)
    plt.figure(1)
    plt.imshow(output)
    plt.show()
    # k=0
    # for example in data:
    #     k=k+1
    #     input_img = example[0]
    #     #print(example[0])
    #     input_label = example[1]
    #     input_img -= np.array((104.00698793, 116.66876762, 122.67891434))
    #     input_img = np.expand_dims(input_img, axis=0)
    #
    #     # Creating synthetic data to test the network
    #     # synthetic_input = np.random.uniform(0, 255, [1, 500, 500, 3])
    #     # synthetic_input = synthetic_input.astype(np.int32)
    #     # synthetic_input = synthetic_input.astype(np.float32)
    #
    #     image_pred = sess.run(pred, feed_dict={input: input_img})
    #
    #     # Saving and plotting the output results
    #     # np.save('../data/image_prediction.npy', image_pred)
    #     output = np.argmax(image_pred, axis=3).reshape((500, 500))
    #     print(np.unique(output))
    #     iou = intersection_over_union(input_label, output)
    #     IOU.append(iou)
    #     print("Done with the image", k)
    #     plt.figure(11)
    #     plt.imshow(output)
    #     plt.figure(12)
    #     plt.imshow(input_label)
    #     plt.show()
    #     break
        #np.save('../data/iou.npy', IOU)

        # f, axarr = plt.subplots()
        # axarr.imshow(im)
        # axarr.imshow(output, alpha=0.7)
        # plt.savefig('../data/output.png')
    # print("The IOU list is ",IOU)
    # print("The average IOU is ",sum(IOU)/len(IOU))
