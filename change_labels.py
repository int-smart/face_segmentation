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
cPickle_file_dir = '/media/abhishek/New Volume/'
image_type = '.jpg'
cPickle_file = 'saved_test_00'

def get_test_data(test_label_dir, test_image_dir):
    i=0
    images_labels = []
    for file in os.listdir(test_label_dir):
        i=i+1
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
        if i%50==0:
            print("Done processing 1000 images and labels")
            save(cPickle_file_dir, cPickle_file+"_0{}".format(i%2), images_labels)
            # images_labels = []
            break
        print("I am done with file {}, {}".format(file,i))
    print(len(images_labels))
    return images_labels

def save(cPickle_file_dir, cPickle_file, images_labels):
    print("Saving file......")
    cPickle_file = cPickle_file_dir+cPickle_file
    stream = gzip.open(cPickle_file, 'wb')
    cPickle.dump(images_labels, stream)
    stream.close()

def load(cPickle_file):
    print("Loading the saved file.....")
    stream = gzip.open(cPickle_file_dir+cPickle_file, "rb")
    model = cPickle.load(stream)
    stream.close()
    return model

def intersection_over_union(ground_truth, prediction):
    iou = ((np.logical_and(ground_truth,prediction)).astype(int)).sum()/((np.logical_or(ground_truth,prediction)).astype(int)).sum()
    return iou
#Important a==b, numpy.astype(int),np.logical_or(a,c)
#


if __name__=='__main__':
    if cPickle_file not in os.listdir(cPickle_file_dir):
        data = get_test_data(test_label_dir,test_image_dir)
        print("I am saving the file")
    else:
        data = load(cPickle_file)
        print("I am loading the file")
    #print(data)
    image = data[0][0]
    label = data[0][1]
    #label = tf.convert_to_tensor(label)
    iou = intersection_over_union(label,label)

    print(iou)
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     sess.run(tf.local_variables_initializer())
    #     iou = sess.run([iou])
    print("The mean iou is ",iou)
    #print("The length of the list is ",len(iou[0]))

    #print("The elements are ", iou[0][1])
    # plt.figure(1)
    # plt.imshow(image[:,:,1])
    # plt.figure(2)
    # plt.imshow(label)
    # plt.show()
