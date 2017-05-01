from util import *

class OSVOS_model():

    vgg_weights = load('vgg16.npy', encoding='latin1').item()
    learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')
    pretrained_weight_path = './pretrained_checkpoint'
    weight_path = './checkpoint'

    def __init__(self, config, img_width, img_height, reuse = None):
        self.threshold = config.threshold
        self.training = config.training
        self.reuse = reuse
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = config.batch_size

    def input_pipeline(self, fn_seg, fn_img):
        reader = tf.WholeFileReader()      
        
        if not len(fn_seg) == len(fn_img):
                raise ValueError('Number of images and segmentations do not match!')

        with tf.variable_scope('image'):
            fn_img_queue = tf.train.string_input_producer(fn_img, shuffle=False)
            _, value = reader.read(fn_img_queue)
            img = tf.image.decode_jpeg(value, channels=3)
            img = tf.image.resize_images(img, [self.img_height, self.img_width], method=tf.image.ResizeMethod.BILINEAR)
            img = tf.cast(img, dtype = tf.float32)
        with tf.variable_scope('segmentation'):
            fn_seg_queue = tf.train.string_input_producer(fn_seg, shuffle=False)
            _, value = reader.read(fn_seg_queue)
            seg = tf.image.decode_png(value, channels=1, dtype=tf.uint8)
            seg = tf.image.resize_images(seg, [self.img_height, self.img_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            seg = tf.reshape(seg, [self.img_height, self.img_width])
            
        if self.training is True:
            with tf.variable_scope('shuffle'):
                seg, img = tf.train.shuffle_batch([seg, img], batch_size=self.batch_size,
                                                    num_threads=4,
                                                    capacity=1000 + 3 * self.batch_size,
                                                    min_after_dequeue=1000)
        return seg/255, img

    def conv_relu_vgg(self, x, name='conv_vgg'):
        kernel = self.vgg_weights[name][0]
        bias = self.vgg_weights[name][1]
        with tf.variable_scope(name):
            x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[0],
                    padding='same', use_bias=True, reuse=self.reuse,
                    kernel_initializer=tf.constant_initializer(kernel),
                    bias_initializer=tf.constant_initializer(bias),
                    name='conv2d', trainable = self.training)
            return tf.nn.relu(x, name='relu')

    def upconv_relu(self, x, num_filters, ksize=3, stride=2, name='upconv'):
        with tf.variable_scope(name):
            x = tf.layers.conv2d_transpose(x, num_filters, ksize, stride,
                    padding='same', use_bias=False, reuse=self.reuse,
                    name='conv2d_transpose', trainable = self.training)
            return tf.nn.relu(x, name='relu')

    def img_croping(self, x):
        start1 = (x.shape[1]-self.img_height)/2
        start2 = (x.shape[2]-self.img_width)/2
        end1 = x.shape[1]-start1
        end2 = x.shape[2]-start2 
        return x[:,start1:end1,start2:end2, 0:16]

    def chk_point_restore(self, sess):
    	saver = tf.train.Saver(max_to_keep=2)
    	if self.training:
            ckpt = tf.train.get_checkpoint_state(self.pretrained_weight_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.weight_path)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if self.training:
                saver.restore(sess, os.path.join(self.pretrained_weight_path, ckpt_name))
            else:
                saver.restore(sess, os.path.join(self.weight_path, ckpt_name))
            print('[*] Success to read {}'.format(ckpt_name))
        else:
            if self.training:
                print('[*] Failed to find a checkpoint. Start training from scratch ...')
            else:
                raise ValueError('[*] Failed to find a checkpoint.')

        return saver

    def build_model(self, x, y):

        with tf.variable_scope('OSVOS'):
            
            x = x[..., ::-1] - [103.939, 116.779, 123.68]

            # 224 448
            conv1 = self.conv_relu_vgg(x, name='conv1_1')
            conv1 = self.conv_relu_vgg(conv1, name='conv1_2')

            # 112 224
            pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='pool1')
            conv2 = self.conv_relu_vgg(pool1, name='conv2_1')
            conv2 = self.conv_relu_vgg(conv2, name='conv2_2')

            # 56 112
            pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool2')
            conv3 = self.conv_relu_vgg(pool2, name='conv3_1')
            conv3 = self.conv_relu_vgg(conv3, name='conv3_2')
            conv3 = self.conv_relu_vgg(conv3, name='conv3_3')

            # 28 56
            pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='pool3')
            conv4 = self.conv_relu_vgg(pool3, name='conv4_1')
            conv4 = self.conv_relu_vgg(conv4, name='conv4_2')
            conv4 = self.conv_relu_vgg(conv4, name='conv4_3')

            # 14 28
            pool4 = tf.layers.max_pooling2d(conv4, 2, 2, name='pool4')
            conv5 = self.conv_relu_vgg(pool4, name='conv5_1')
            conv5 = self.conv_relu_vgg(conv5, name='conv5_2')
            conv5 = self.conv_relu_vgg(conv5, name='conv5_3')
     
            prep2 = tf.layers.conv2d(inputs = conv2, filters = 16, kernel_size = 3, strides = 1,
                    padding='same', use_bias=True, reuse=self.reuse,
                    name='prep2', trainable = self.training)
            prep3 = tf.layers.conv2d(inputs = conv3, filters = 16, kernel_size = 3, strides = 1,
                    padding='same', use_bias=True, reuse=self.reuse,
                    name='prep3', trainable = self.training)
            prep4 = tf.layers.conv2d(inputs = conv4, filters = 16, kernel_size = 3, strides = 1,
                    padding='same', use_bias=True, reuse=self.reuse,
                    name='prep4', trainable = self.training)              
            prep5 = tf.layers.conv2d(inputs = conv5, filters = 16, kernel_size = 3, strides = 1,
                    padding='same', use_bias=True, reuse=self.reuse,
                    name='prep5', trainable = self.training)       

            #upsampling
            up2 = tf.layers.conv2d_transpose(prep2, filters=16, kernel_size = 4, strides = 2,
                    padding='valid', use_bias=False, reuse=self.reuse,
                    name='up2', trainable = self.training)
            up2c = self.img_croping(up2)

            up3 = tf.layers.conv2d_transpose(prep3, filters=16, kernel_size = 8, strides = 4,
                    padding='valid', use_bias=False, reuse=self.reuse,
                    name='up3', trainable = self.training)
            up3c = self.img_croping(up3)

            up4 = tf.layers.conv2d_transpose(prep4, filters=16, kernel_size = 16, strides = 8,
                    padding='valid', use_bias=False, reuse=self.reuse,
                    name='up4', trainable = self.training)
            up4c = self.img_croping(up4)

            up5 = tf.layers.conv2d_transpose(prep5, filters=16, kernel_size = 32, strides = 16,
                    padding='valid', use_bias=False, reuse=self.reuse,
                    name='up5', trainable = self.training)
            up5c = self.img_croping(up5)

                
            concat_score = tf.concat([up2c, up3c,up4c,up5c], axis=3, name='concat_score')

            out_prep = tf.layers.conv2d(inputs = concat_score, filters = 1, kernel_size = 1, strides = 1,
                    padding='same', use_bias=False, reuse=self.reuse,
                    name='out_prep', trainable = self.training)  
            
            threshold = tf.constant(self.threshold, dtype = float32)

            #filter based on threshold
            out1 = tf.floordiv(tf.sigmoid(out_prep), threshold, name=None)
            
            out = tf.reshape(out1,[-1, self.img_height, self.img_width],name='out')
            
            logits = tf.reshape(out_prep, [-1, self.img_height, self.img_width])

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   logits=logits, labels=tf.to_float(y)),name = "loss")
            return out,loss

    def OSVOS_training_setup(self, fn_seg, fn_img):

        self.batch_size = len(fn_img)

        y, x = self.input_pipeline(fn_seg, fn_img)
        logits, loss = self.build_model(x, y)

        y = tf.to_int64(y, name = 'y')
        pred_train = tf.to_int64(logits, name = 'pred_train')
        result_train = tf.concat([y, pred_train], axis=2)
        result_train = tf.cast(255 * tf.reshape(result_train, [-1, self.img_height, self.img_width*2, 1]), tf.uint8)

        tf.summary.scalar('loss', loss)
        tf.summary.image('result_train', result_train, max_outputs=self.batch_size)
        tf.summary.scalar('learning_rate', self.learning_rate)
    
        sum_all = tf.summary.merge_all()
        
        vars_trainable = tf.trainable_variables()

        num_param = 0

        for var in vars_trainable:
            num_param += prod(var.get_shape()).value
            tf.summary.histogram(var.name, var)

        print('\nTotal nummber of parameters = %d' % num_param)

        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, var_list=vars_trainable)
        
        return loss, sum_all, train_step, pred_train

    def OSVOS_training(self, sess, config, path_pack, loss, sum_all, train_step, pred_train, saver):

        total_count = 0
        t0 = time.time()

        writer = tf.summary.FileWriter("./logs", sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(config.num_epoch):

            lr = config.init_learning_rate * config.learning_rate_decay**epoch

            for k in range(self.batch_size // config.batch_size):
                
                l_train, train_result, _ = sess.run([loss, pred_train, train_step], feed_dict={self.learning_rate: lr})
                
                if total_count<= 200:
                    train_result = reshape(train_result[0], (480, 640))
                    train_result = around(median_filter(train_result, 9))
                    train_result = 255*train_result
                    train_result[(train_result>0)] = 255
                    
                    gt_seg = imresize(imread(path_pack.data_dir+'/gt/001.png'),(480, 640))
                    
                    train_result = concatenate((gt_seg, train_result), axis = 0)
                    train_result[(train_result>0)] = 255
                    
                    imsave(path_pack.train_result_dir+'/'+str(total_count)+'.png', train_result)

                writer.add_summary(sess.run(sum_all, feed_dict={self.learning_rate: lr}), total_count)
                total_count += 1                
                m, s = divmod(time.time() - t0, 60)
                h, m = divmod(m, 60)
                print('Epoch: [%4d/%4d], [%4d/%4d], Time: [%02d:%02d:%02d], loss: %.4f'
                % (epoch+1, config.num_epoch, k+1, self.batch_size // config.batch_size, h, m, s, l_train))


            if epoch % 100 == 0:
                print('Saving checkpoint ...')
                saver.save(sess, self.weight_path + '/Davis.ckpt')

        coord.request_stop()         
        coord.join(threads)

    def OSVOS_testing_setup(self, fn_seg, fn_img):

        fn_seg = [fn_seg[0]]*len(fn_img)
        y, x = self.input_pipeline(fn_seg, fn_img)
        y = tf.reshape(y, [1, self.img_height, self.img_width])
        x = tf.reshape(x, [1, self.img_height, self.img_width, 3])
        logits, loss = self.build_model(x, y)
        y = tf.to_int64(y, name = 'y')
        val_result = tf.to_int64(logits, name = 'val_result')
        val_result = tf.cast(tf.reshape(val_result, [-1, self.img_height, self.img_width, 1]), tf.uint8)
        input_image = tf.cast(x, tf.uint8)

    	return val_result, x

    def OSVOS_testing(self, sess, config, path_pack, fn_img, val_result, x):

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        total_count = 1
        for k in range(len(fn_img)):
            result, image = sess.run([val_result, x])

            result = reshape(result, (self.img_height, self.img_width))
            result = around(median_filter(result, 9))
            result = 255*result
            print result.shape
            slice_x, slice_y = bbox_generate(result)
            height = slice_x.stop-slice_x.start+1
            width = slice_y.stop-slice_y.start+1
            if (width == self.img_width or height == self.img_height) or (width <=10 or height <= 10):
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

                imsave(path_pack.result_dir+'/'+img_name+'.jpg', vis)

                img_path = path_pack.resize_image_dir+'/'+config.object+time.strftime("%d_%m_%Y")+'_'+str(total_count)+'.jpg'

                imsave(img_path, image[0])
                total_count += 1
                bbox = bbox_property(slice_y.start, slice_y.stop, slice_x.start, slice_x.stop, config.label)
                print('Writing .XML File!')
                write_xml(img_path, path_pack.xml_path, bbox)

        coord.request_stop()         
        coord.join(threads)
        print('Writing .txt File!')
        write_txt(resize_image_dir, txt_path, 'train', config.object)
        print('Writing Done!')


