from util import *
import tensorflow as tf  
import time, os
import argparse
import glob
from numpy import *
from scipy.misc import imsave,imshow,imresize
from scipy.signal import medfilt
from PIL import Image
import subprocess
from scipy.ndimage.filters import median_filter
import cv2
#default resolution is 640*480
def main(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    data_dir = './table/table_9'
    result_dir = './results'
    train_result_dir = './train_results'
    resize_image_dir = './progress/JPEGImages'
    xml_path = './progress/Annotations'
    txt_path = './progress/ImageSets/Main'
    learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')
    t0 = time.time()
    if config.training:
        if not config.edge_training:
            print('\nLoading data from '+data_dir)
            fn_img = []
            fn_seg = []
            if config.stage is 1:
                with open(data_dir+'/ImageSets/1080p/train.txt', 'r') as f:
                    for line in f:
                        
                            i,s = line.split(' ')
                            fn_img.append(data_dir+i)
                            fn_seg.append(data_dir+s[:-1])
                    
            elif config.stage is 2:

                fn_img = [data_dir+'/001.jpg']
                fn_seg = [data_dir+'/gt/001.png']
                config.batch_size = len(fn_img)

            if not os.path.exists(train_result_dir):
                os.makedirs(train_result_dir)

            y, x = input_pipeline(fn_seg, fn_img, config.batch_size)
            logits, loss = build_model(x, y, reuse = None, training = config.training)
            tf.summary.scalar('loss', loss)
            y = tf.to_int64(y, name = 'y')
            pred_train = tf.to_int64(logits, name = 'pred_train')
            result_train = tf.concat([y, pred_train], axis=2)
            result_train = tf.cast(255 * tf.reshape(result_train, [-1, 480, 640*2, 1]), tf.uint8)
            tf.summary.image('result_train', result_train, max_outputs=config.batch_size)
            num_param = 0
            vars_trainable = tf.trainable_variables()
            for var in vars_trainable:
                num_param += prod(var.get_shape()).value
                tf.summary.histogram(var.name, var)

            print('\nTotal nummber of parameters = %d' % num_param)

            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=vars_trainable)
            
            
        else:
            label_pattern = './Data/edge_image'
            image_pattern = './Data/VOC2010/JPEGImages'
            images, labels = load_edge_image(label_pattern, image_pattern)
            images_val, labels_val = load_edge_image(label_pattern, image_pattern)
        
    else:
        print('\nLoading data from '+data_dir)
        #data_dir = './progress'
        fn_img = []
        fn_seg = []
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(resize_image_dir):
            os.makedirs(resize_image_dir)

        fn_img = sorted(glob.glob(data_dir+'/*.jpg'), key=numericalSort)
        fn_seg = sorted(glob.glob(data_dir+'/gt/*.png'), key=numericalSort)

        fn_seg = [fn_seg[0]]*len(fn_img)
        #print fn_seg

        y, x = input_pipeline(fn_seg, fn_img, 1,  training = config.training)
        y = tf.reshape(y, [1, 480, 640])
        x = tf.reshape(x, [1, 480, 640, 3])
        logits, loss = build_model(x, y, reuse = None, training = config.training)
        y = tf.to_int64(y, name = 'y')
        val_result = tf.to_int64(logits, name = 'val_result')
        val_result = tf.cast(tf.reshape(val_result, [-1, 480, 640, 1]), tf.uint8)
        input_image = tf.cast(x, tf.uint8)

    print('Finished loading in %.2f seconds.' % (time.time() - t0))
    
    tf.summary.scalar('learning_rate', learning_rate)
    
    sum_all = tf.summary.merge_all()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=2)

        if config.training:
            ckpt = tf.train.get_checkpoint_state('./pretrained_checkpoint')
        else:
            ckpt = tf.train.get_checkpoint_state('./checkpoint')

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if config.training:
                saver.restore(sess, os.path.join('./pretrained_checkpoint', ckpt_name))
            else:
                saver.restore(sess, os.path.join('./checkpoint', ckpt_name))
            print('[*] Success to read {}'.format(ckpt_name))
        else:
            if config.training:
                print('[*] Failed to find a checkpoint. Start training from scratch ...')
            else:
                raise ValueError('[*] Failed to find a checkpoint.')

        if config.training:
            writer = tf.summary.FileWriter("./logs", sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            total_count = 0
            t0 = time.time()
            for epoch in range(config.num_epoch):

                lr = config.init_learning_rate * config.learning_rate_decay**epoch

                for k in range(len(fn_seg) // config.batch_size):
                    
                    l_train, train_result, _ = sess.run([loss, pred_train, train_step], feed_dict={learning_rate: lr})
                    
                    if total_count<= 200:
                        train_result = reshape(train_result[0], (480, 640))
                        train_result = around(median_filter(train_result, 9))
                        train_result = 255*train_result
                        train_result[(train_result>0)] = 255
                        
                        gt_seg = imresize(imread(data_dir+'/gt/001.png'),(480, 640))
                        
                        train_result = concatenate((gt_seg, train_result), axis = 0)
                        train_result[(train_result>0)] = 255
                        imsave(train_result_dir+'/'+str(total_count)+'.png', train_result)

                    writer.add_summary(sess.run(sum_all, feed_dict={learning_rate: lr}), total_count)
                    total_count += 1                
                    m, s = divmod(time.time() - t0, 60)
                    h, m = divmod(m, 60)
                    print('Epoch: [%4d/%4d], [%4d/%4d], Time: [%02d:%02d:%02d], loss: %.4f'
                    % (epoch+1, config.num_epoch, k+1, len(fn_seg) // config.batch_size, h, m, s, l_train))


                if epoch % 100 == 0:
                    print('Saving checkpoint ...')
                    saver.save(sess, './checkpoint/Davis.ckpt')
            coord.request_stop()         
            coord.join(threads)
        else:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            count = 1
            for k in range(len(fn_img)):
                result, image = sess.run([val_result, x])

                result = reshape(result, (480, 640))
                result = around(median_filter(result, 9))
                result = 255*result

                slice_x, slice_y = bbox_generate(result)
                height = slice_x.stop-slice_x.start+1
                width = slice_y.stop-slice_y.start+1
                if (width == 640 or height == 480) or (width <=10 or height <= 10):
                    print('Evaluate picture %d/%d !!!!Not meet requirement' % (k+1, len(fn_img)))
                    pass
                else:
                    img_name = os.path.basename(fn_img[k])
                    img_name = os.path.splitext(img_name)[0]
                    print('Evaluate picture %d/%d :: BBox size is height:[%d, %d] width:[%d, %d]' % (k+1, len(fn_img), 
                        slice_x.start, slice_x.stop, slice_y.start, slice_y.stop))
                    
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

                    vis = concatenate((image[0], result), axis=0)

                    cv2.rectangle(vis,(slice_y.start,slice_x.start),(slice_y.stop,slice_x.stop),(0,255,0),3)

                    imsave(result_dir+'/'+img_name+'.jpg', vis)

                    img_path = resize_image_dir+'/'+config.object+time.strftime("%d_%m_%Y")+'_'+str(count)+'.jpg'

                    imsave(img_path, image[0])
                    count = count+1
                    bbox = bbox_property(slice_y.start, slice_y.stop, slice_x.start, slice_x.stop, config.label)
                    print('Writing .XML File!')
                    write_xml(img_path, xml_path, bbox)

            coord.request_stop()         
            coord.join(threads)
            print('Writing .txt File!')
            write_txt(resize_image_dir, txt_path, 'train', config.object)
            print('Writing Done!')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='OSVOS_demo')
    parser.add_argument('--edge', dest='edge_training', help='set edge_flag, default is False',
                        default=False, type=str2bool)
    parser.add_argument('--train', dest='training', help='set train_flag, default is True',
                        default=True, type=str2bool)
    parser.add_argument('--stage', dest='stage', help='set train_stage, default is 1',
                        default=2, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Number of images in each batch',
                        default=1, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', help='Total number of epochs to run for training',
                        default=1000, type=int)
    parser.add_argument('--init_learning_rate', dest='init_learning_rate', help='Initial learning rate',
                        default=1e-5, type=float)
    parser.add_argument('--learning_rate_decay', dest='learning_rate_decay', help='Ratio for decaying the learning rate after each epoch',
                        default=1, type=float)
    parser.add_argument('--gpu', dest='gpu', help='GPU to be used',
                        default='0', type=str)
    parser.add_argument('--threshold', dest='threshold', help='threshold to display',
                        default=0, type=float)

    parser.add_argument('--object', dest='object', help='object for data collection',
                        default='table_9_', type=str)

    parser.add_argument('--label', dest='label', help='object label for data collection',
                        default='table', type=str)


    config = parser.parse_args()


    return config

if __name__ == '__main__':
    config = parse_args()
    main(config)

    




