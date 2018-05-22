import os
import shutil
import tensorflow as tf
import scipy.io
import tools
import numpy as np
import time
import test
import SimpleITK as ST
from dicom_read import read_dicoms
import gc

# global variables
###############################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 2
ori_lr = 0.0001
power = 0.9
# GPU0 = '1'
input_shape = [64,64,128]
output_shape = [64,64,128]
epoch_walked = 0
step_walked = 0
upper_threshold = 0.6
MAX_EPOCH = 2000
re_example_epoch = 2
total_test_epoch = 1
show_step = 10
block_test_step = 20
model_save_step = 50
output_epoch = total_test_epoch * 20
test_extra_threshold = 0.20
edge_thickness = 20
original_g = 24
growth_d = 10
layer_num_d = 12
test_dir = './WU_XIAO_YING/'
config={}
config['batch_size'] = batch_size
config['meta_path'] = '/opt/artery_extraction/data_meta_airway.pkl'
config['data_size'] = input_shape
config['test_amount'] = 2
config['train_amount'] = 12
decay_step = 44 / (config['train_amount'] / 2)
################################################################

class Network:
    def __init__(self):
        self.train_models_dir = './airway_train_models/'
        self.train_sum_dir = './airway_sum/train/'
        self.test_results_dir = './airway_test_results/'
        self.test_sum_dir = './airway_sum/test/'

        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir)
            print 'test_results_dir: deleted and then created!\n'
        os.makedirs(self.test_results_dir)

        if os.path.exists(self.train_models_dir):
            print 'train_models_dir: existed! will be loaded! \n'
        else:
            os.makedirs(self.train_models_dir)

        if os.path.exists(self.train_sum_dir):
            print 'train_sum_dir: existed! \n'
        else:
            os.makedirs(self.train_sum_dir)

        if os.path.exists(self.test_sum_dir):
            shutil.rmtree(self.test_sum_dir)
            print 'test_sum_dir: deleted and then created!\n'
        os.makedirs(self.test_sum_dir)

    def Dense_Block(self,X,name,depth,growth,training):
        with tf.variable_scope(name):
            original = X.get_shape().as_list()[-1]
            c_e = []
            s_e = []
            layers = []
            layers.append(X)
            for i in range(depth):
                c_e.append(original + growth * (i + 1))
                s_e.append(1)
            for j in range(depth):
                with tf.variable_scope("input_"+str(j+1)):
                    input = tf.concat([sub_layer for sub_layer in layers], axis=4)
                with tf.variable_scope("dense_layer_"+str(j+1)):
                    layer = tools.Ops.batch_norm(input, 'bn_dense_1_1_' + str(j+1), training=training)
                    layer = tools.Ops.xxlu(layer, name='relu_1')
                    layer = tools.Ops.conv3d(layer, k=1, out_c=growth, str=s_e[j], name='dense_1_1_' + str(j+1))
                    layer = tools.Ops.batch_norm(layer, 'bn_dense_1_2_' + str(j), training=training)
                    layer = tools.Ops.xxlu(layer, name='relu_2')
                    layer = tools.Ops.conv3d(layer, k=3, out_c=growth, str=s_e[j], name='dense_1_2_' + str(j+1))
                layers.append(layer)
            with tf.variable_scope("out_put"):
                ret = tf.concat([sub_layer for sub_layer in layers], axis=4)
        return ret

    def Down_Sample(self,X,name,str,training,size):
        with tf.variable_scope(name):
            down_sample_input = tools.Ops.conv3d(X, k=3, out_c=size, str=1, name='down_sample_input')
            bn_input = tools.Ops.batch_norm(down_sample_input, "bn_input", training=training)
            relu_input = tools.Ops.xxlu(bn_input, name="relu_input")
            down_sample = tools.Ops.conv3d(relu_input, k=str, out_c=size, str=str, name='down_sample')
        return down_sample

    def Up_Sample(self,X,name,str,training,size):
        with tf.variable_scope(name):
            up_sample_input = tools.Ops.conv3d(X, k=3, out_c=size, str=1, name='up_sample_input')
            bn_1 = tools.Ops.batch_norm(up_sample_input, 'bn_after_dense_1', training=training)
            relu_1 = tools.Ops.xxlu(bn_1, name='relu_1')
            deconv_1 = tools.Ops.deconv3d(relu_1, k=2, out_c=size, str=str, name='deconv_up_sample_2')
        return deconv_1
            # concat_up_1 = tf.concat([deconv_1, layers_e[-1]], axis=4, name="concat_up_1")

    def Input(self,X,name,batch_size,size,training):
        with tf.variable_scope(name):
            X = tf.reshape(X, [batch_size, input_shape[0], input_shape[1], input_shape[2], 1])
            conv_input = tools.Ops.conv3d(X, k=3, out_c=size, str=1, name="conv_input")
            bn_input = tools.Ops.batch_norm(conv_input, "bn_input", training)
            relu_input = tools.Ops.xxlu(bn_input, "relu_input")
        return relu_input

    def Predict(self,X,name,training,threshold):
        with tf.variable_scope(name):
            predict_conv_1 = tools.Ops.conv3d(X, k=2, out_c=32, str=1, name="conv_predict_1")
            # bn_1 = tools.Ops.batch_norm(predict_conv_1,"bn_predict_1",training)
            # relu_1 = tools.Ops.xxlu(bn_1, name="relu_predict_1")
            predict_map = tools.Ops.conv3d(predict_conv_1, k=1, out_c=1, str=1, name="predict_map")
            # bn_2 = tools.Ops.batch_norm(predict_conv_2,"bn_predict_2",training)
            # relu_2 = tools.Ops.xxlu(predict_conv_2, name="relu_predict_2")
            vox_no_sig = predict_map
            vox_sig = tf.sigmoid(predict_map)
            vox_sig_modified = tf.maximum(vox_sig - threshold, 0.01)
        return vox_sig,vox_sig_modified,vox_no_sig

    def Concat(self,inputs,axis,size,name):
        with tf.variable_scope(name):
            concat_input = tf.concat([elem for elem in inputs],axis=axis,name="concat_input")
            concat_conv = tools.Ops.conv3d(concat_input,k=3,out_c=size,str=1,name="concat_conv")
        return concat_conv

    def ae_u(self,X,training,batch_size,threshold):
        original=original_g
        growth=growth_d
        dense_layer_num=layer_num_d
        X_input = self.Input(X,"input",batch_size,original,training)
        down_1 = self.Down_Sample(X_input,"down_sample_1",2,training,original)
        dense_1 = self.Dense_Block(down_1,"dense_block_1",dense_layer_num, growth,training)
        down_2 = self.Down_Sample(dense_1,"down_sample_2",2,training,original*2)

        dense_2 = self.Dense_Block(down_2,"dense_block_2",dense_layer_num,growth,training)

        up_input_1 = self.Concat([down_2,dense_2,
                                  self.Down_Sample(down_1,"cross_1",2,training,original),
                                  self.Down_Sample(X_input,"cross_2",4,training,original)],axis=4,size=original*3,name="concat_up_1")
        up_1 = self.Up_Sample(up_input_1,"up_sample_1",2,training,128)

        up_input_2 = self.Concat([up_1,dense_1],axis=4,size=original,name="concat_up_2")
        up_2 = self.Up_Sample(up_input_2,"up_sample_2",2,training,64)

        predict_input = self.Concat([up_2, X_input,
                                     self.Up_Sample(dense_2, "cross_3", 4, training, original),
                                     self.Up_Sample(up_1, "cross_5", 2, training, original)], axis=4,
                                    size=original * 4, name="predict_input")
        vox_sig, vox_sig_modified, vox_no_sig = self.Predict(predict_input, "predict", training, threshold)
        return vox_sig, vox_sig_modified, vox_no_sig

    def dis(self, X, Y,training):
        with tf.variable_scope("input"):
            X = tf.reshape(X, [batch_size, input_shape[0], input_shape[1], input_shape[2], 1])
            Y = tf.reshape(Y, [batch_size, output_shape[0], output_shape[1], output_shape[2], 1])
            # layer = tf.concat([X, Y], axis=4)
            layer = X*Y
            c_d = [1, 2, 64, 128, 256, 512]
            s_d = [0, 2, 2, 2, 2, 2]
            layers_d = []
            layers_d.append(layer)
        with tf.variable_scope("down_sample"):
            for i in range(1,6,1):
                layer = tools.Ops.conv3d(layers_d[-1],k=4,out_c=c_d[i],str=s_d[i],name='d_1'+str(i))
                if i!=5:
                    layer = tools.Ops.xxlu(layer, name='lrelu')
                    # batch normal layer
                    layer = tools.Ops.batch_norm(layer, 'bn_up' + str(i), training=training)
                layers_d.append(layer)
        with tf.variable_scope("flating"):
            y = tf.reshape(layers_d[-1], [batch_size, -1])
        return tf.nn.sigmoid(y)

    def train(self,configure):
        # data
        data = tools.Data(configure, epoch_walked/re_example_epoch)
        # network
        X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
        lr = tf.placeholder(tf.float32)
        training = tf.placeholder(tf.bool)
        threshold = tf.placeholder(tf.float32)
        with tf.variable_scope('generator'):
            Y_pred, Y_pred_modi, Y_pred_nosig = self.ae_u(X, training, batch_size, threshold)
        with tf.variable_scope('discriminator'):
            XY_real_pair = self.dis(X, Y, training)
        with tf.variable_scope('discriminator', reuse=True):
            XY_fake_pair = self.dis(X, Y_pred, training)

        # loss function
        # generator loss
        Y_ = tf.reshape(Y, shape=[batch_size, -1])
        Y_pred_modi_ = tf.reshape(Y_pred_modi, shape=[batch_size, -1])
        w = tf.placeholder(tf.float32)  # foreground weight
        g_loss = tf.reduce_mean(-tf.reduce_mean(w * Y_ * tf.log(Y_pred_modi_ + 1e-8), reduction_indices=[1]) -
                                tf.reduce_mean((1 - w) * (1 - Y_) * tf.log(1 - Y_pred_modi_ + 1e-8),
                                               reduction_indices=[1]))
        g_loss_sum = tf.summary.scalar("generator cross entropy",g_loss)
        # discriminator loss
        gan_d_loss = tf.reduce_mean(XY_fake_pair) - tf.reduce_mean(XY_real_pair)
        alpha = tf.random_uniform(shape=[batch_size, input_shape[0] * input_shape[1] * input_shape[2]], minval=0.0,
                                  maxval=1.0)
        Y_pred_ = tf.reshape(Y_pred, shape=[batch_size, -1])
        differences_ = Y_pred_ - Y_
        interpolates = Y_ + alpha * differences_
        with tf.variable_scope('discriminator', reuse=True):
            XY_fake_intep = self.dis(X, interpolates, training)
        gradients = tf.gradients(XY_fake_intep, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        gan_d_loss += 10 * gradient_penalty
        gan_d_loss_sum = tf.summary.scalar("total loss of discriminator",gan_d_loss)

        # generator loss with gan loss
        gan_g_loss = -tf.reduce_mean(XY_fake_pair)
        gan_g_w = 5
        ae_w = 100 - gan_g_w
        ae_gan_g_loss = ae_w * g_loss + gan_g_w * gan_g_loss
        ae_g_loss_sum = tf.summary.scalar("total loss of generator",ae_gan_g_loss)

        # trainers
        ae_var = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        dis_var = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        ae_g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
            ae_gan_g_loss, var_list=ae_var)
        dis_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
            gan_d_loss, var_list=dis_var)

        # accuracy
        block_acc = tf.placeholder(tf.float32)
        total_acc = tf.placeholder(tf.float32)
        train_sum = tf.summary.scalar("train_block_accuracy", block_acc)
        test_sum = tf.summary.scalar("total_test_accuracy", total_acc)
        train_merge_op = tf.summary.merge([train_sum,ae_g_loss_sum,gan_d_loss_sum,g_loss_sum])
        test_merge_op = tf.summary.merge([test_sum])

        saver = tf.train.Saver(max_to_keep=1)
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.visible_device_list = GPU0

        with tf.Session() as sess:
            # define tensorboard writer
            sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            sum_write_test = tf.summary.FileWriter(self.test_sum_dir, sess.graph)
            # load model data if pre-trained
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            if os.path.isfile(self.train_models_dir + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                saver.restore(sess, self.train_models_dir + 'model.cptk')
            learning_rate_g = ori_lr * pow(power, (epoch_walked / decay_step))
            # start training loop
            global_step = step_walked
            for epoch in range(epoch_walked, MAX_EPOCH):
                if epoch % re_example_epoch == 0 and epoch > 0:
                    del data
                    gc.collect()
                    data = tools.Data(configure, epoch/re_example_epoch)
                train_amount = len(data.train_numbers)
                test_amount = len(data.test_numbers)
                if train_amount >= test_amount and train_amount > 0 and test_amount > 0 and data.total_train_batch_num > 0 and data.total_test_seq_batch > 0:
                    # actual foreground weight
                    weight_for = 0.5 + (1-1.0*epoch/MAX_EPOCH)*0.35
                    if epoch % total_test_epoch == 0:
                        self.full_testing(sess,X,w,threshold,
                                          test_merge_op,
                                          sum_write_test,training,
                                          weight_for,total_acc,Y_pred,
                                          Y_pred_modi,Y_pred_nosig,epoch)
                    data.shuffle_X_Y_pairs()
                    total_train_batch_num = data.total_train_batch_num
                    print "total_train_batch_num:", total_train_batch_num
                    for i in range(total_train_batch_num):
                        X_train_batch, Y_train_batch = data.load_X_Y_voxel_train_next_batch()
                        # calculate loss value
                        # print "calculate begin"
                        gan_d_loss_c, = sess.run([gan_d_loss],
                                                 feed_dict={X: X_train_batch, Y: Y_train_batch, training: False,
                                                            w: weight_for, threshold: upper_threshold})
                        g_loss_c, gan_g_loss_c = sess.run([g_loss, ae_gan_g_loss],
                                                          feed_dict={X: X_train_batch, Y: Y_train_batch,
                                                                     training: False, w: weight_for,
                                                                     threshold: upper_threshold})
                        # print "calculate ended"
                        if epoch % decay_step == 0 and epoch > epoch_walked and i == 0:
                            learning_rate_g = learning_rate_g * power
                        sess.run([ae_g_optim],
                                 feed_dict={X: X_train_batch, threshold: upper_threshold, Y: Y_train_batch,
                                            lr: learning_rate_g, training: True, w: weight_for})
                        sess.run([dis_optim], feed_dict={X: X_train_batch, threshold: upper_threshold, Y: Y_train_batch,
                                                         lr: learning_rate_g, training: True, w: weight_for})
                        # print "training ended"
                        global_step += 1
                        # output some results
                        if i % show_step == 0:
                            print "epoch:", epoch, " i:", i, " train ae loss:", g_loss_c, " gan g loss:", gan_g_loss_c, " gan d loss:", gan_d_loss_c, " learning rate: ", learning_rate_g
                        if i % block_test_step == 0 and epoch % 1 == 0:
                            try:
                                X_test_batch, Y_test_batch = data.load_X_Y_voxel_test_next_batch(fix_sample=False)
                                g_loss_t, gan_g_loss_t, gan_d_loss_t, Y_test_pred, Y_test_modi, Y_test_pred_nosig = \
                                    sess.run([g_loss, ae_gan_g_loss, gan_d_loss, Y_pred, Y_pred_modi, Y_pred_nosig],
                                             feed_dict={X: X_test_batch,
                                                        threshold: upper_threshold + test_extra_threshold,
                                                        Y: Y_test_batch, training: False, w: weight_for})
                                predict_result = np.float32(Y_test_modi > 0.01)
                                predict_result = np.reshape(predict_result,
                                                            [batch_size, input_shape[0], input_shape[1],
                                                             input_shape[2]])
                                print np.max(Y_test_pred)
                                print np.min(Y_test_pred)
                                # IOU
                                predict_probablity = np.float32((Y_test_modi - 0.01) > 0)
                                predict_probablity = np.reshape(predict_probablity,
                                                                [batch_size, input_shape[0], input_shape[1],
                                                                 input_shape[2]])
                                accuracy = 2 * np.sum(np.abs(predict_probablity * Y_test_batch)) / np.sum(
                                    np.abs(predict_result) + np.abs(Y_test_batch))
                                print "epoch:", epoch, " global step: ", global_step, "\nIOU accuracy: ", accuracy, "\ntest ae loss:", g_loss_t, " gan g loss:", gan_g_loss_t, " gan d loss:", gan_d_loss_t
                                print "weight of foreground : ", weight_for
                                print "upper threshold of testing",(upper_threshold + test_extra_threshold)
                                train_summary = sess.run(train_merge_op, feed_dict={block_acc: accuracy,
                                                                                    X: X_test_batch,
                                                                                    threshold: upper_threshold + test_extra_threshold,
                                                                                    Y: Y_test_batch, training: False,
                                                                                    w: weight_for})
                                sum_writer_train.add_summary(train_summary, global_step=global_step)
                            except Exception, e:
                                print e
                        #### model saving
                        if i % model_save_step == 0 and epoch % 1 == 0:
                            saver.save(sess, save_path=self.train_models_dir + 'model.cptk')
                            print "epoch:", epoch, " i:", i, "regular model saved!"
                else:
                    print "bad data , next epoch", epoch

    def full_testing(self,sess,X,w,threshold,test_merge_op,sum_write_test,training,weight_for,total_acc,Y_pred, Y_pred_modi, Y_pred_nosig,epoch):
        print '********************** FULL TESTING ********************************'
        # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        # w = tf.placeholder(tf.float32)
        # threshold = tf.placeholder(tf.float32)
        time_begin = time.time()
        origin_data = read_dicoms(test_dir + "original1")
        mask_dir = test_dir + "airway"
        test_batch_size = batch_size
        # test_data = tools.Test_data(dicom_dir,input_shape)
        test_data = tools.Test_data(origin_data, input_shape, 'vtk_data')
        test_data.organize_blocks()
        test_mask = read_dicoms(mask_dir)
        array_mask = ST.GetArrayFromImage(test_mask)
        array_mask = np.transpose(array_mask, (2, 1, 0))
        print "mask shape: ", np.shape(array_mask)
        block_numbers = test_data.blocks.keys()
        for i in range(0, len(block_numbers), test_batch_size):
            batch_numbers = []
            if i + test_batch_size < len(block_numbers):
                temp_input = np.zeros(
                    [test_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                for j in range(test_batch_size):
                    temp_num = block_numbers[i + j]
                    temp_block = test_data.blocks[temp_num]
                    batch_numbers.append(temp_num)
                    block_array = temp_block.load_data()
                    block_shape = np.shape(block_array)
                    temp_input[j, 0:block_shape[0], 0:block_shape[1],
                    0:block_shape[2]] += block_array
                Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
                    [Y_pred, Y_pred_modi, Y_pred_nosig],
                    feed_dict={X: temp_input,
                               training: False,
                               w: weight_for,
                               threshold: upper_threshold + test_extra_threshold})
                for j in range(test_batch_size):
                    test_data.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])
            else:
                temp_batch_size = len(block_numbers) - i
                temp_input = np.zeros(
                    [temp_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                for j in range(temp_batch_size):
                    temp_num = block_numbers[i + j]
                    temp_block = test_data.blocks[temp_num]
                    batch_numbers.append(temp_num)
                    block_array = temp_block.load_data()
                    block_shape = np.shape(block_array)
                    temp_input[j, 0:block_shape[0], 0:block_shape[1],
                    0:block_shape[2]] += block_array
                X_temp = tf.placeholder(
                    shape=[temp_batch_size, input_shape[0], input_shape[1], input_shape[2]],
                    dtype=tf.float32)
                with tf.variable_scope('generator', reuse=True):
                    Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp = self.ae_u(X_temp, training,
                                                                                 temp_batch_size,
                                                                                 threshold)
                Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
                    [Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp],
                    feed_dict={X_temp: temp_input,
                               training: False,
                               w: weight_for,
                               threshold: upper_threshold + test_extra_threshold})
                for j in range(temp_batch_size):
                    test_data.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])
        test_result_array = test_data.get_result()
        print "result shape: ", np.shape(test_result_array)
        to_be_transformed = self.post_process(test_result_array)
        if epoch == 0:
            test_data.output_origin(self.test_results_dir)
        if epoch % output_epoch == 0:
            self.output_img(to_be_transformed, test_data.space, epoch)
        if epoch == total_test_epoch:
            mask_img = ST.GetImageFromArray(np.transpose(array_mask, [2, 1, 0]))
            mask_img.SetSpacing(test_data.space)
            ST.WriteImage(mask_img, self.test_results_dir + 'test_mask.vtk')
        test_IOU = 2 * np.sum(to_be_transformed * array_mask) / (
                np.sum(to_be_transformed) + np.sum(array_mask))
        test_summary = sess.run(test_merge_op, feed_dict={total_acc: test_IOU})
        sum_write_test.add_summary(test_summary, global_step=epoch)
        print "IOU accuracy: ", test_IOU
        time_end = time.time()
        print '******************** time of full testing: ' + str(time_end - time_begin) + 's ********************'

    def post_process(self,test_result_array):
        r_s = np.shape(test_result_array)  # result shape
        e_t = edge_thickness  # edge thickness
        to_be_transformed = np.zeros(r_s, np.float32)
        to_be_transformed[e_t:r_s[0] - e_t, e_t:r_s[1] - e_t,
        e_t:r_s[2] - e_t] += test_result_array[e_t:r_s[0] - e_t, e_t:r_s[1] - e_t, e_t:r_s[2] - e_t]
        print np.max(to_be_transformed)
        print np.min(to_be_transformed)
        return to_be_transformed

    def output_img(self,to_be_transformed,spacing,epoch):
        final_img = ST.GetImageFromArray(np.transpose(to_be_transformed, [2, 1, 0]))
        final_img.SetSpacing(spacing)
        print "writing full testing result"
        if not os.path.exists(self.test_results_dir):
            os.makedirs(self.test_results_dir)
        print self.test_results_dir + "test_result_" + str(epoch) + '.vtk'
        ST.WriteImage(final_img, self.test_results_dir + "test_result_" + str(epoch) + '.vtk')

    def test(self,dicom_dir):
        # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        test_input_shape = input_shape
        test_batch_size = batch_size * 2
        threshold = tf.placeholder(tf.float32)
        training = tf.placeholder(tf.bool)
        X = tf.placeholder(shape=[test_batch_size, test_input_shape[0], test_input_shape[1], test_input_shape[2]],
                           dtype=tf.float32)
        with tf.variable_scope('generator'):
            Y_pred, Y_pred_modi, Y_pred_nosig = self.ae_u(X, training, test_batch_size, threshold)

        print tools.Ops.variable_count()
        saver = tf.train.Saver(max_to_keep=1)
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.visible_device_list = GPU0
        with tf.Session() as sess:
            # if os.path.exists(self.train_models_dir):
            #     saver.restore(sess, self.train_models_dir + 'model.cptk')
            sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

            if os.path.exists(self.train_models_dir) and os.path.isfile(
                    self.train_models_dir + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                saver.restore(sess, self.train_models_dir + 'model.cptk')
            else:
                sess.run(tf.global_variables_initializer())
            test_data = tools.Test_data(dicom_dir, input_shape, 'dicom_data')
            test_data.organize_blocks()
            block_numbers = test_data.blocks.keys()
            for i in range(0, len(block_numbers), test_batch_size):
                batch_numbers = []
                if i + test_batch_size < len(block_numbers):
                    temp_input = np.zeros(
                        [test_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                    for j in range(test_batch_size):
                        temp_num = block_numbers[i + j]
                        temp_block = test_data.blocks[temp_num]
                        batch_numbers.append(temp_num)
                        block_array = temp_block.load_data()
                        block_shape = np.shape(block_array)
                        temp_input[j, 0:block_shape[0], 0:block_shape[1], 0:block_shape[2]] += block_array
                    Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run([Y_pred, Y_pred_modi, Y_pred_nosig],
                                                                           feed_dict={X: temp_input,
                                                                                      training: False,
                                                                                      threshold: upper_threshold + test_extra_threshold})
                    for j in range(test_batch_size):
                        test_data.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])
                else:
                    temp_batch_size = len(block_numbers) - i
                    temp_input = np.zeros(
                        [temp_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                    for j in range(temp_batch_size):
                        temp_num = block_numbers[i + j]
                        temp_block = test_data.blocks[temp_num]
                        batch_numbers.append(temp_num)
                        block_array = temp_block.load_data()
                        block_shape = np.shape(block_array)
                        temp_input[j, 0:block_shape[0], 0:block_shape[1], 0:block_shape[2]] += block_array
                    X_temp = tf.placeholder(
                        shape=[temp_batch_size, input_shape[0], input_shape[1], input_shape[2]],
                        dtype=tf.float32)
                    with tf.variable_scope('generator', reuse=True):
                        Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp = self.ae_u(X_temp, training,
                                                                                     temp_batch_size, threshold)
                    Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
                        [Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp],
                        feed_dict={X_temp: temp_input,
                                   training: False,
                                   threshold: upper_threshold + test_extra_threshold})
                    for j in range(temp_batch_size):
                        test_data.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])
            test_result_array = test_data.get_result()
            print "result shape: ", np.shape(test_result_array)
            r_s = np.shape(test_result_array)  # result shape
            e_t = 10  # edge thickness
            to_be_transformed = np.zeros(r_s, np.float32)
            to_be_transformed[e_t:r_s[0] - e_t, e_t:r_s[1] - e_t, 0:r_s[2] - e_t] += test_result_array[
                                                                                     e_t:r_s[0] - e_t,
                                                                                     e_t:r_s[1] - e_t,
                                                                                     0:r_s[2] - e_t]
            print np.max(to_be_transformed)
            print np.min(to_be_transformed)
            final_img = ST.GetImageFromArray(np.transpose(to_be_transformed, [2, 1, 0]))
            final_img.SetSpacing(test_data.space)
            print "writing final testing result"
            # print './test_result/test_result_final.vtk'
            ST.WriteImage(final_img, self.test_results_dir + "test_result" + '.vtk')
            return final_img

if __name__ == "__main__":
    dicom_dir = "./FU_LI_JUN/original1"
    net = Network()
    net.train(config)
    # final_img = net.test(dicom_dir)
    # ST.WriteImage(final_img,'./final_result.vtk')
