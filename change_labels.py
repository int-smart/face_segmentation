import numpy as np
from PIL import Image
import tensorflow as tf
import _pickle as cPickle
import gzip
import os
from scipy import misc
from matplotlib import pyplot as plt

test_label_dir = '/media/abhishek/B67A61587A611701/Users/Second/Desktop/Research/FCN/parts_lfw_funneled_gt_images/'
test_image_dir = '/media/abhishek/B67A61587A611701/Users/Second/Desktop/Research/FCN/lfw_funneled/'
image_type = '.jpg'
cPickle_file = 'saved_test'

def get_test_data(test_label_dir,test_image_dir):
    images_labels = []
    for file in os.listdir(test_label_dir):
        path_to_label = test_label_dir+file
        if os.path.isfile(path_to_label):
            if '.ppm' in file:
                file = file.replace('.ppm','')
                file = file.split('_')
                file_number = file[-1]
                file_name = file[:-1]
                file_name = '_'.join(file_name)
                dir_name = file_name
                file_name = file_name+'_'+str(file_number)+image_type
            label = misc.imread(path_to_label)
            label = label[:,:,1]

            path_to_image = test_image_dir+dir_name+'/'+file_name
            image = misc.imread(path_to_image)
            #image_scipy = misc.imread('/media/abhishek/B67A61587A611701/Users/Second/Desktop/Research/FCN/parts_lfw_funneled_gt_images/Aaron_Peirsol_0001.ppm')
            #print(image_scipy.shape)
            # plt.imshow(image_scipy[:,:,1])
            # plt.show()
            images_labels.append([image,label])
    return images_labels

def save(test_image_dir, cPickle_file, images_labels):
    cPickle_file = test_image_dir+cPickle_file
    stream = gzip.open(cPickle_file, 'wb')
    cPickle.dump(images_labels, stream)
    stream.close()

def load(cPickle_file):
    stream = gzip.open(cPickle_file, "rb")
    model = cPickle.load(stream)
    stream.close()
    return model

if __name__=='__main__':
    if cPickle_file not in os.listdir(test_image_dir):
        data = get_test_data(test_label_dir,test_image_dir)
        save(test_image_dir, cPickle_file, data)
        print("I am saving the file")
    else:
        data = load(cPickle_file)
        print("I am loading the file")
    image = data[0][0]
    label = data[0][1]
    label = tf.convert_to_tensor(label)
    iou = tf.metrics.mean_iou(label,label,2)
    print(iou)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        iou = sess.run([iou])
    print("The mean iou is ",iou)
    #print("The length of the list is ",len(iou[0]))

    #print("The elements are ", iou[0][1])
    #plt.figure(1)
    #plt.imshow(image[:,:,1])
    #plt.figure(2)
    #plt.imshow(label)
    #plt.show()
