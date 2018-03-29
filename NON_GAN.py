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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
batch_size = 2
ori_lr = 0.0002
power = 0.9
# GPU0 = '1'
# [artery , airway , background]
weights = [0.8,0.1,0.1]
input_shape = [64,64,128]
output_shape = [64,64,128]
epoch_walked = 0
step_walked = 0
MAX_EPOCH = 2000
class_num = 3
re_example_epoch = 2
total_test_epoch = 4
show_step = 10
block_test_step = 20
model_save_step = 50
output_epoch = total_test_epoch * 10
edge_thickness = 15
test_dir = './FU_LI_JUN/'
config={}
config['batch_size'] = batch_size
config['meta_path'] = '/opt/artery_extraction/data_meta.pkl'
config['data_size'] = input_shape
config['test_amount'] = 2
config['train_amount'] = 8
decay_step = 2 * 16 / (config['train_amount'] - 1)
################################################################

class Network:
    def __init__(self):
        self.train_models_dir = './D_multi_train_models/'
        self.train_sum_dir = './D_multi_sum/train/'
        self.test_results_dir = './D_multi_test_results/'
        self.test_sum_dir = './D_multi_sum/test/'

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
                layer = tools.Ops.batch_norm(layers[-1], 'bn_dense_1_1_' + str(j), training=training)
                layer = tools.Ops.xxlu(layer, name='relu_1')
                layer = tools.Ops.conv3d(layer, k=1, out_c=growth, str=s_e[j], name='dense_1_1_' + str(j))
                layer = tools.Ops.batch_norm(layer, 'bn_dense_1_2_' + str(j), training=training)
                layer = tools.Ops.xxlu(layer, name='relu_2')
                layer = tools.Ops.conv3d(layer, k=3, out_c=growth, str=s_e[j], name='dense_1_2_' + str(j))
                next_input = tf.concat([layer, layers[-1]], axis=4)
                layers.append(next_input)
        return layers[-1]

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
            predict_conv_1 = tools.Ops.conv3d(X, k=3, out_c=32, str=1, name="conv_predict_1")
            bn_1 = tools.Ops.batch_norm(predict_conv_1,"bn_predict_1",training)
            relu_1 = tools.Ops.xxlu(bn_1, name="relu_predict_1")
            predict_conv_2 = tools.Ops.conv3d(relu_1, k=1, out_c=1, str=1, name="conv_predict_2")
            # bn_2 = tools.Ops.batch_norm(predict_conv_2,"bn_predict_2",training)
            relu_2 = tools.Ops.xxlu(predict_conv_2, name="relu_predict_2")
            vox_sig = tf.sigmoid(predict_conv_2)
            vox_sig_modified = tf.maximum(vox_sig - threshold, 0.01)
        return vox_sig,vox_sig_modified,relu_2

    def Pixel_Classifier(self,X,name,training,num_class):
        with tf.variable_scope(name):
            predict_conv_1 = tools.Ops.conv3d(X, k=3, out_c=32, str=1, name="conv_predict_1")
            bn_1 = tools.Ops.batch_norm(predict_conv_1,"bn_predict_1",training)
            relu_1 = tools.Ops.xxlu(bn_1, name="relu_predict_1")
            predict_conv_2 = tools.Ops.conv3d(relu_1, k=1, out_c=num_class, str=1, name="conv_predict_2")
            bn_2 = tools.Ops.batch_norm(predict_conv_2,"bn_predict_2",training)
            relu_2 = tools.Ops.xxlu(bn_2, name="relu_predict_2")
            pred = tf.nn.softmax(relu_2,dim=-1,name="classify_probability")
            pred_label = tf.arg_max(pred,dimension=-1)
        return relu_2,pred,pred_label

    def E_Loss(self,logits,label):
        return tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits,dim=-1)

    def Concat(self,inputs,axis,size,name):
        with tf.variable_scope(name):
            concat_input = tf.concat([elem for elem in inputs],axis=axis,name="concat_input")
            concat_conv = tools.Ops.conv3d(concat_input,k=3,out_c=size,str=1,name="concat_conv")
        return concat_conv

    def Segmentor(self,X,training,batch_size):
        original=64
        growth=12
        dense_layer_num=8
        X_input = self.Input(X,"input",batch_size,original,training)
        down_1 = self.Down_Sample(X_input,"down_sample_1",2,training,original)
        dense_1 = self.Dense_Block(down_1,"dense_block_1",dense_layer_num,growth,training)
        down_2 = self.Down_Sample(dense_1,"down_sample_2",2,training,original*2)

        dense_2 = self.Dense_Block(down_2,"dense_block_2",dense_layer_num,growth,training)

        up_input_1 = self.Concat([down_2,dense_2,
                                  self.Down_Sample(dense_1,"cross_1",2,training,original),
                                  self.Down_Sample(X_input,"cross_2",4,training,original)],axis=4,size=original*3,name="concat_up_1")
        up_1 = self.Up_Sample(up_input_1,"up_sample_1",2,training,original*2)

        dense_input_3 = self.Concat([up_1,dense_1],axis=4,size=original*2,name="concat_dense_3")
        dense_3 = self.Dense_Block(dense_input_3,"dense_block_3",dense_layer_num,growth,training)

        up_input_2 = self.Concat([dense_3,down_1],axis=4,size=original,name="concat_up_2")
        up_2 = self.Up_Sample(up_input_2,"up_sample_2",2,training,original)

        predict_input = tf.concat([up_2,X_input],axis=4,name="predict_input")

        return self.Pixel_Classifier(predict_input,"segment",training,class_num)

    def train(self,configure):
        #  data
        data = tools.Data_multi(configure, epoch_walked/re_example_epoch)
        # network
        X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2],class_num], dtype=tf.float32)
        lr = tf.placeholder(tf.float32)
        training = tf.placeholder(tf.bool)
        with tf.variable_scope('segment'):
            pixel_logits,prob,pred_label = self.Segmentor(X, training, batch_size)
        with tf.variable_scope('loss'):
            pixel_loss = self.E_Loss(pixel_logits,Y)

        #  weight map
        weight_map = tf.convert_to_tensor(np.zeros(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]]),dtype=tf.float32)
        pred_masks = []
        for i in range(class_num):
            pred_masks.append(tf.cast(tf.equal(pred_label,i),tf.float32))
        for i in range(class_num):
            weight_map = weight_map+weights[i]*pred_masks[i]

        #  weighted loss
        enhanced_loss = weight_map * pixel_loss

        #  accuracy
        accuracys = []
        for i in range(class_num):
            accuracys.append(tf.reduce_mean(tf.cast(tf.equal(pred_masks[i],Y[:,:,:,:,i]),tf.float32)))
        # [artery , airway , background]
        total_acc = tf.placeholder(tf.float32)
        artery_acc_sum = tf.summary.scalar("artery_accuracy",accuracys[0])
        airway_acc_sum = tf.summary.scalar("airway_accuracy",accuracys[1])
        background_acc_sum = tf.summary.scalar("background_acc",accuracys[2])
        total_acc_sum = tf.summary.scalar("total_accuracy",total_acc)
        train_merge_op = tf.summary.merge([artery_acc_sum,airway_acc_sum,background_acc_sum])
        test_merge_op = tf.summary.merge([total_acc_sum])


        with tf.Session() as sess:
            sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            sum_write_test = tf.summary.FileWriter(self.test_sum_dir, sess.grap)



        print "end training"



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
        if not os.path.exists("./test_result"):
            os.makedirs("./test_result")
        print './test_result/test_result' + str(epoch) + '.vtk'
        ST.WriteImage(final_img, './test_result/test_result' + str(epoch) + '.vtk')


if __name__ == "__main__":
    dicom_dir = "./FU_LI_JUN/original1"
    net = Network()
    net.train(config)
    # final_img = net.test(dicom_dir)
    # ST.WriteImage(final_img,'./final_result.vtk')
