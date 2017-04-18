from util import *
import tensorflow as tf  
import time, os
import argparse
import glob
from numpy import *
from scipy.misc import imsave,imshow 
from scipy.signal import medfilt
from PIL import Image
import subprocess
from scipy.ndimage.filters import median_filter
def main(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    data_dir = './chair/01'
    result_dir = './results'
 
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
                '''
                with open(data_dir+'/ImageSets/1080p/val.txt', 'r') as f:
                    for line in f:
                        i,s = line.split(' ')
                        if ('00000.jpg' in i) and ('00000.png' in s):
                            fn_img.append(data_dir+i)
                            fn_seg.append(data_dir+s[:-1])'''
                fn_img = [data_dir+'/001.jpg', data_dir+'/056.jpg']
                fn_seg = [data_dir+'/gt/001.png', data_dir+'/gt/056.png']

            y, x = input_pipeline(fn_seg, fn_img, config.batch_size)
            logits, loss = build_model(x, y, reuse = None, training = config.training)
            tf.summary.scalar('loss', loss)
            y = tf.to_int64(y, name = 'y')
            pred_train = tf.to_int64(logits, name = 'pred_train')
            result_train = tf.concat([y, pred_train], axis=2)
            result_train = tf.cast(255 * tf.reshape(result_train, [-1, 480, 854*2, 1]), tf.uint8)
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
        '''
        with open(data_dir+'/ImageSets/1080p/val.txt', 'r') as f:
            for line in f:
                i,s = line.split(' ')
                fn_img.append(data_dir+i)
                fn_seg.append(data_dir+s[:-1])'''
        fn_img = sorted(glob.glob(data_dir+'/*.jpg'), key=numericalSort)
        fn_seg = sorted(glob.glob(data_dir+'/gt/*.png'), key=numericalSort)

        fn_seg = [fn_seg[0]]*len(fn_img)
        print fn_seg

        y, x = input_pipeline(fn_seg, fn_img, 1,  training = config.training)
        y = tf.reshape(y, [1, 480, 854])
        x = tf.reshape(x, [1, 480, 854, 3])
        logits, loss = build_model(x, y, reuse = None, training = config.training)
        y = tf.to_int64(y, name = 'y')
        val_result = tf.to_int64(logits, name = 'val_result')
        #val_result = tf.concat([y, val_result], axis=2)
        val_result = tf.cast(tf.reshape(val_result, [-1, 480, 854, 1]), tf.uint8)
        input_image = tf.cast(x, tf.uint8)
        #tf.summary.image('val_result', val_result, max_outputs=8)
        #tf.summary.image('input_image', input_image, max_outputs=8)

    print('Finished loading in %.2f seconds.' % (time.time() - t0))
    
    tf.summary.scalar('learning_rate', learning_rate)
    
    sum_all = tf.summary.merge_all()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=10)
        #ckpt = tf.train.get_checkpoint_state('./pretrained_checkpoint')
        ckpt = tf.train.get_checkpoint_state('./checkpoint')
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join('./checkpoint', ckpt_name))
            #saver.restore(sess, os.path.join('./pretrained_checkpoint', ckpt_name))
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
                    
                    l_train, _ = sess.run([loss, train_step], feed_dict={learning_rate: lr})
                    
                    writer.add_summary(sess.run(sum_all, feed_dict={learning_rate: lr}), total_count)
                    total_count += 1                
                    m, s = divmod(time.time() - t0, 60)
                    h, m = divmod(m, 60)
                    print('Epoch: [%4d/%4d], [%4d/%4d], Time: [%02d:%02d:%02d], loss: %.4f'
                    % (epoch, config.num_epoch, k, len(fn_seg) // config.batch_size, h, m, s, l_train))

                if epoch % 50 == 0:
                    print('Saving checkpoint ...')
                    saver.save(sess, './checkpoint/Davis.ckpt')
        else:
            writer = tf.summary.FileWriter("./logs", sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            lr = config.init_learning_rate * config.learning_rate_decay    

            for k in range(len(fn_img)):
                result = sess.run(val_result)
                #result = result.astype(float)
                #index = result > config.threshold
                #print result
                #result[index] = 1
                

                result = 255*result
                #result = medfilt(result, 3)
                writer.add_summary(sess.run(sum_all, feed_dict={learning_rate: lr}), k)
                print('Evaluate picture %d/%d' % (k+1, len(fn_img)))
                result = reshape(result, (480, 854))
                result = median_filter(result, 9)
                #result = uint8(result)
                #im = Image.fromarray(result)
                img_name = os.path.basename(fn_img[k])
                img_name = os.path.splitext(img_name)[0]

                imsave(result_dir+'/'+img_name+'.png', result)
                #p = subprocess.Popen(["display", result_dir+'/'+img_name+'.png'])
                #time.sleep(1)
                #p.kill()
                    
                


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



    config = parser.parse_args()


    return config

if __name__ == '__main__':
    config = parse_args()
    main(config)

    




