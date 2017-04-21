import tensorflow as tf
import os
from glob import glob
from numpy import *
import numpy as np
from scipy.misc import imread, imresize,imshow
from sys import stdout
import re
from scipy import ndimage
import matplotlib.pyplot as plt
from lxml import etree

vgg_weights = load('vgg16.npy', encoding='latin1').item()
numbers = re.compile(r'(\d+)')

class bbox_property:
    def __init__(self, xmin, xmax, ymin, ymax, label):
        self.label = label
        self.xmin = str(xmin)
        self.xmax = str(xmax)
        self.ymin = str(ymin)
        self.ymax = str(ymax)


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_images(pattern):
    fn = sorted(glob(pattern))
    if 'images' in pattern:
        img = zeros((len(fn), 480, 854, 3), dtype=uint8)
        for k in range(len(fn)):
            img1 = imread(fn[k])
            img1 = imresize(img1, (480, 854,3))
            img[k,...] = img1
    else:
        img = zeros((len(fn), 480, 854), dtype=uint8)
        for k in range(len(fn)):
            pimg = imread(fn[k])
            if len(pimg.shape) == 3:
                img1 = pimg[:,:,0]
                img1 = imresize(img1, (480, 854))
                img[k, ...] = img1
            else:
                img1 = imresize(pimg, (480, 854))
                img[k,...] = img1

    return img

def load_edge_image(label_pattern, image_pattern):
    list_of_label = sorted(glob(label_pattern+'/*.png'))
    list_of_image = sorted(glob(image_pattern+'/*.jpg'))
    len_label = 100#len(list_of_label)
    label = zeros((len_label, 448, 448), dtype=uint8)
    img = zeros((len_label, 448, 448, 3), dtype=uint8)
    print 'loading the data....'
    for k in range(len_label):
        label1 = imread(list_of_label[k])
        #label1 = imresize(label1, (448, 448))
        label1 = label1/255
        label[k,...] = label1
        base = os.path.basename(list_of_label[k])
        base = os.path.splitext(base)[0]
        matching = [s for s in list_of_image if base in s]
        img1 = imread(matching[0])
        img1 = imresize(img1, (448, 448, 3))
        img[k,...] = img1
        rate = float(k)/float(len_label)*100.0
        stdout.write("\r completing... %.2f %%" % rate)
        stdout.flush()
       
    stdout.write("\n")
    print 'finish loading data!' 
    return img, label

def input_pipeline(fn_seg, fn_img, batch_size, training = True):
    reader = tf.WholeFileReader()
       
    #print fn_img
    if not len(fn_seg) == len(fn_img):
            raise ValueError('Number of images and segmentations do not match!')
    with tf.variable_scope('image'):
        fn_img_queue = tf.train.string_input_producer(fn_img, shuffle=False)
        _, value = reader.read(fn_img_queue)
        img = tf.image.decode_jpeg(value, channels=3)
        img = tf.image.resize_images(img, [480, 640], method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, dtype = tf.float32)
    with tf.variable_scope('segmentation'):
        fn_seg_queue = tf.train.string_input_producer(fn_seg, shuffle=False)
        _, value = reader.read(fn_seg_queue)
        seg = tf.image.decode_png(value, channels=1, dtype=tf.uint8)
        seg = tf.image.resize_images(seg, [480, 640], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        seg = tf.reshape(seg, [480, 640])
        
    if training is True:
        #print 'shuffle!!!!!!!!!!!!!!!!!!!!!!!!!'
        with tf.variable_scope('shuffle'):
            seg, img = tf.train.shuffle_batch([seg, img], batch_size=batch_size,
                                                num_threads=4,
                                                capacity=1000 + 3 * batch_size,
                                                min_after_dequeue=1000)
    return seg/255, img


def conv_relu_vgg(x, reuse=None, name='conv_vgg', training = True):
    kernel = vgg_weights[name][0]
    bias = vgg_weights[name][1]
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[0],
                padding='same', use_bias=True, reuse=reuse,
                kernel_initializer=tf.constant_initializer(kernel),
                bias_initializer=tf.constant_initializer(bias),
                name='conv2d', trainable = training)
        return tf.nn.relu(x, name='relu')

def upconv_relu(x, num_filters, ksize=3, stride=2, reuse=None, name='upconv', training = True):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, num_filters, ksize, stride,
                padding='same', use_bias=False, reuse=reuse,
                name='conv2d_transpose', trainable = training)
        return tf.nn.relu(x, name='relu')

def build_model(x, y, reuse=None, training=True, threshold = 0.9):
    with tf.variable_scope('OSVOS'):
        
        x = x[..., ::-1] - [103.939, 116.779, 123.68]

        # 224 448
        conv1 = conv_relu_vgg(x, reuse=reuse, name='conv1_1', training = training)
        conv1 = conv_relu_vgg(conv1, reuse=reuse, name='conv1_2', training = training)

        # 112 224
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='pool1')
        conv2 = conv_relu_vgg(pool1, reuse=reuse, name='conv2_1', training = training)
        conv2 = conv_relu_vgg(conv2, reuse=reuse, name='conv2_2', training = training)

        # 56 112
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool2')
        conv3 = conv_relu_vgg(pool2, reuse=reuse, name='conv3_1', training = training)
        conv3 = conv_relu_vgg(conv3, reuse=reuse, name='conv3_2', training = training)
        conv3 = conv_relu_vgg(conv3, reuse=reuse, name='conv3_3', training = training)

        # 28 56
        pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='pool3')
        conv4 = conv_relu_vgg(pool3, reuse=reuse, name='conv4_1', training = training)
        conv4 = conv_relu_vgg(conv4, reuse=reuse, name='conv4_2', training = training)
        conv4 = conv_relu_vgg(conv4, reuse=reuse, name='conv4_3', training = training)

        # 14 28
        pool4 = tf.layers.max_pooling2d(conv4, 2, 2, name='pool4')
        conv5 = conv_relu_vgg(pool4, reuse=reuse, name='conv5_1', training = training)
        conv5 = conv_relu_vgg(conv5, reuse=reuse, name='conv5_2', training = training)
        conv5 = conv_relu_vgg(conv5, reuse=reuse, name='conv5_3', training = training)

        # 7 14
        #pool5 = tf.layers.max_pooling2d(conv5, 2, 2, name='pool5')
        #(a)for segmentation
        #prepare 
        prep2 = tf.layers.conv2d(inputs = conv2, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse,
                name='prep2', trainable = training)
        prep3 = tf.layers.conv2d(inputs = conv3, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse,
                name='prep3', trainable = training)
        prep4 = tf.layers.conv2d(inputs = conv4, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse,
                name='prep4', trainable = training)              
        prep5 = tf.layers.conv2d(inputs = conv5, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse,
                name='prep5', trainable = training)       
        #upsampling
        up2 = tf.layers.conv2d_transpose(prep2, filters=16, kernel_size = 4, strides = 2,
                padding='same', use_bias=False, reuse=reuse,
                name='up2', trainable = training)
        start1 = (up2.shape[1]-480)/2
        start2 = (up2.shape[2]-640)/2
        end1 = up2.shape[1]-start1
        end2 = up2.shape[2]-start2 
        up2c = up2[:,start1:end1,start2:end2,0:16]#tf.image.resize_image_with_crop_or_pad(up2, 480, 854)
        up3 = tf.layers.conv2d_transpose(prep3, filters=16, kernel_size = 8, strides = 4,
                padding='valid', use_bias=False, reuse=reuse,
                name='up3', trainable = training)
        start1 = (up3.shape[1]-480)/2
        start2 = (up3.shape[2]-640)/2
        end1 = up3.shape[1]-start1
        end2 = up3.shape[2]-start2 
        up3c = up3[:,start1:end1,start2:end2,0:16]#tf.image.resize_image_with_crop_or_pad(up3, 480, 854)
        up4 = tf.layers.conv2d_transpose(prep4, filters=16, kernel_size = 16, strides = 8,
                padding='valid', use_bias=False, reuse=reuse,
                name='up4', trainable = training)
        start1 = (up4.shape[1]-480)/2
        start2 = (up4.shape[2]-640)/2
        end1 = up4.shape[1]-start1
        end2 = up4.shape[2]-start2 
        up4c = up4[:,start1:end1,start2:end2,0:16]#tf.image.resize_image_with_crop_or_pad(up4, 480, 854)
        up5 = tf.layers.conv2d_transpose(prep5, filters=16, kernel_size = 32, strides = 16,
                padding='valid', use_bias=False, reuse=reuse,
                name='up5', trainable = training)
        start1 = (up5.shape[1]-480)/2
        start2 = (up5.shape[2]-640)/2
        end1 = up5.shape[1]-start1
        end2 = up5.shape[2]-start2 
        up5c = up5[:,start1:end1,start2:end2,0:16]#tf.image.resize_image_with_crop_or_pad(up5, 480, 854)
        
    
        concat_score = tf.concat([up2c, up3c,up4c,up5c], axis=3, name='concat_score')
        out_prep = tf.layers.conv2d(inputs = concat_score, filters = 1, kernel_size = 1, strides = 1,
                padding='same', use_bias=False, reuse=reuse,
                name='out_prep', trainable = training)  
        
        threshold = tf.constant(threshold, dtype = float32)

        #filter based on threshold
        out1 = tf.floordiv(tf.sigmoid(out_prep), threshold, name=None)

        #out1 = tf.round(tf.sigmoid(out_prep))
        #out1 = tf.sigmoid(out_prep)
        
        out = tf.reshape(out1,[-1,480,640],name='out')
        #loss = -tf.reduce_mean(y*tf.log(out)+(1-y)*tf.log(1-out))
        logits = tf.reshape(out_prep, [-1, 480, 640])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                               logits=logits, labels=tf.to_float(y)),name = "loss")
        return out,loss
     
        
def str2bool(parameter):
    if parameter.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if parameter.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')            

def bbox_generate(image):

    mask = image>0

    label_im, nb_labels = ndimage.label(mask)

    # Find the largest connect component
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))#sizes of connected component. a lists
    mask_size = sizes < 100
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = unique(label_im)
    label_im = searchsorted(labels, label_im)

    # Now that we have only one connect component, extract it's bounding box
    slice_x, slice_y = ndimage.find_objects(label_im==(len(labels)-1))[0] #find the largest one

    return slice_x, slice_y 

def write_txt(datapath, writepath, set_name, label):

    if not os.path.exists(writepath):
            os.makedirs(writepath)
    
    im_list = sorted(glob(datapath+'/*'+label+'*.jpg'), key=numericalSort)
    
    with open(writepath+'/'+set_name+'.txt', 'w') as f:
        for im in im_list:
            im = os.path.basename(im)
            im = os.path.splitext(im)[0]
            f.write(im+'\n')

def write_xml(file_name, writepath, bbox):

    if not os.path.exists(writepath):
            os.makedirs(writepath)

    label = bbox.label

    xml_file_name = os.path.basename(file_name)
    xml_file_name = os.path.splitext(xml_file_name)[0]+'.xml'
    with open(writepath +'/'+ xml_file_name, 'w') as out:

        img = imread(file_name)
        print img.shape

        root = etree.Element('annotation')
        chd_folder = etree.Element('folder')
        chd_folder.text = 'progress'
        root.append(chd_folder)
        chd_fname = etree.Element('filename')
        chd_fname.text = os.path.basename(file_name)
        root.append(chd_fname)

        chd_size = etree.Element('size')
        chd_size_width = etree.Element('width')
        chd_size_width.text = str(img.shape[1]) 
        chd_size_height = etree.Element('height')
        chd_size_height.text = str(img.shape[0])
        chd_size_depth = etree.Element('depth')
        chd_size_depth.text = str(img.shape[2])
        chd_size.append(chd_size_width)
        chd_size.append(chd_size_height)
        chd_size.append(chd_size_depth)         
        root.append(chd_size)

        chd_obj = etree.Element('object')
        chd_obj_name = etree.Element('name')
        chd_obj_name.text = label
        chd_obj.append(chd_obj_name)
        chd_obj_bbox = etree.Element('bndbox')
        chd_obj_bbox_xmin = etree.Element('xmin')   
        chd_obj_bbox_xmin.text = bbox.xmin
        chd_obj_bbox.append(chd_obj_bbox_xmin)
        chd_obj_bbox_ymin = etree.Element('ymin')   
        chd_obj_bbox_ymin.text = bbox.ymin
        chd_obj_bbox.append(chd_obj_bbox_ymin)
        chd_obj_bbox_xmax = etree.Element('xmax')   
        chd_obj_bbox_xmax.text = bbox.xmax 
        chd_obj_bbox.append(chd_obj_bbox_xmax)
        chd_obj_bbox_ymax = etree.Element('ymax')   
        chd_obj_bbox_ymax.text = bbox.ymax 
        chd_obj_bbox.append(chd_obj_bbox_ymax)
        chd_obj.append(chd_obj_bbox)
        root.append(chd_obj)


        s = etree.tostring(root, pretty_print=True)

        print s
        out.write(s)

        print('Writing Done!')