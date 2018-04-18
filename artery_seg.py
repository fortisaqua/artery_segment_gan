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
import utils as ut
# global variables
###############################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
batch_size = 2
# GPU0 = '1'
input_shape = [256,256,8]
output_shape = [256,256,8]
upper_threshold = 0.6
MAX_EPOCH = 2000
re_example_epoch = 2
total_test_epoch = 1
show_step = 10
block_test_step = 20
model_save_step = 50
output_epoch = total_test_epoch * 20
test_extra_threshold = 0.25
edge_thickness = 20
original_g = 24
growth_d = 24
layer_num_d = 4
config={}
config['batch_size'] = batch_size
config['meta_path'] = '/opt/artery_extraction/data_meta_artery.pkl'
################################################################

class Network:
    def __init__(self):
        self.train_models_dir = '/opt/artery_segment_gan/artery_train_models/'
        self.test_results_dir = '/opt/artery_segment_gan/artery_test_results/'

        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir)
            print 'test_results_dir: deleted and then created!\n'
        os.makedirs(self.test_results_dir)

        if os.path.exists(self.train_models_dir):
            print 'train_models_dir: existed! will be loaded! \n'
        else:
            os.makedirs(self.train_models_dir)

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
            predict_conv_1 = tools.Ops.conv3d(X, k=2, out_c=64, str=1, name="conv_predict_1")
            bn_1 = tools.Ops.batch_norm(predict_conv_1,"bn_predict_1",training)
            relu_1 = tools.Ops.xxlu(bn_1, name="relu_predict_1")
            predict_map = tools.Ops.conv3d(relu_1, k=1, out_c=1, str=1, name="predict_map")
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
        down_1 = self.Down_Sample(X_input,"down_sample_1",2,training,original*1)
        dense_1 = self.Dense_Block(down_1,"dense_block_1",dense_layer_num,growth,training)
        down_2 = self.Down_Sample(dense_1,"down_sample_2",2,training,original*2)
        dense_2 = self.Dense_Block(down_2,"dense_block_2",dense_layer_num,growth,training)
        down_3 = self.Down_Sample(dense_2,"down_sample_3",2,training,original*4)

        dense_3 = self.Dense_Block(down_3,"dense_block_3",dense_layer_num,growth,training)
        mid_input = self.Concat([dense_3,
                                  self.Down_Sample(dense_2, "cross_1", 2, training, original),
                                  self.Down_Sample(dense_1, "cross_2", 4, training, original),
                                  self.Down_Sample(X_input, "cross_3", 8, training, original),
                                  ],
                                 axis=4,size=original*6,name="concat_up_mid")
        dense_4 = self.Dense_Block(mid_input,"dense_block_4",dense_layer_num,growth,training)

        up_input_1 = self.Concat([down_3,dense_4],axis=4,size=original*8,name = "up_input_1")
        up_1 = self.Up_Sample(up_input_1,"up_sample_1",2,training,original*4)

        dense_input_5 = self.Concat([up_1,dense_2],axis=4,size=original*4,name = "dense_input_5")
        dense_5 = self.Dense_Block(dense_input_5,"dense_block_5",dense_layer_num,growth,training)

        up_input_2 = self.Concat([dense_5,down_2],axis=4,size=original*6,name = "up_input_2")
        up_2 = self.Up_Sample(up_input_2,"up_sample_2",2,training,original*2)

        dense_input_6 = self.Concat([up_2,dense_1],axis=4,size=original*2,name = "dense_input_6")
        dense_6 = self.Dense_Block(dense_input_6,"dense_block_6",dense_layer_num,growth,training)

        up_input_3 = self.Concat([dense_6,down_1],axis=4,size=original*6,name = "up_input_3")
        up_3 = self.Up_Sample(up_input_3,"up_sample_3",2,training,original*1)

        predict_input = self.Concat([up_3,
                                     self.Up_Sample(dense_6, "cross_4", 2, training, original),
                                     self.Up_Sample(up_2, "cross_5", 2, training, original),
                                     self.Up_Sample(dense_5, "cross_6", 4, training, original),
                                     self.Up_Sample(up_1, "cross_7", 4, training, original),
                                     self.Up_Sample(dense_4, "cross_8", 8, training, original),
                                     self.Up_Sample(mid_input, "cross_9", 8, training, original),
                                     self.Up_Sample(dense_3, "cross_10", 8, training, original)],
                                    axis=4,
                                    size=original * 4, name="predict_input")
        vox_sig, vox_sig_modified, vox_no_sig = self.Predict(predict_input, "predict", training, threshold)

        return vox_sig, vox_sig_modified, vox_no_sig

    def post_process(self,test_result_array):
        # r_s = np.shape(test_result_array)  # result shape
        # e_t = edge_thickness  # edge thickness
        # to_be_transformed = np.zeros(r_s, np.float32)
        # to_be_transformed[e_t:r_s[0] - e_t, e_t:r_s[1] - e_t,
        # e_t:r_s[2] - e_t] += test_result_array[e_t:r_s[0] - e_t, e_t:r_s[1] - e_t, e_t:r_s[2] - e_t]
        # print np.max(to_be_transformed)
        # print np.min(to_be_transformed)
        # return to_be_transformed
        time1 = time.time()
        to_be_transformed = ut.Select_biggest(test_result_array)
        time2 = time.time()
        print "time cost for post processing : ",str(time2-time1)," s"
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
        time1 = time.time()
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

            processed = self.post_process(to_be_transformed)
            time2 = time.time()
            print "time cost : ",str(time2-time1)," s"
            print np.max(processed)
            print np.min(processed)
            final_img = ST.GetImageFromArray(np.transpose(processed, [2, 1, 0]))
            final_img.SetSpacing(test_data.space)
            print "writing final testing result"
            print self.test_results_dir+'test_result_final.vtk'
            ST.WriteImage(final_img, self.test_results_dir+'test_result_final.vtk')
            return final_img

# if __name__ == "__main__":
#     dicom_dir = "./FU_LI_JUN/original1"
#     net = Network()
    # net.train(config)
    # net.test(dicom_dir)
    # ST.WriteImage(final_img,'./final_result.vtk')

def artery_seg(dicom_dir):
    net = Network()
    net.test(dicom_dir)