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
ori_lr = 0.01
power = 0.9
# GPU0 = '1'
# [artery , airway , background]
mask_names = ["artery","airway","background"]
weights = [10,0.5,0.5]
input_shape = [64,64,128]
output_shape = [64,64,128]
epoch_walked = 0
step_walked = 0
MAX_EPOCH = 2000
class_num = 3
re_example_epoch = 2
total_test_epoch = 1
show_step = 10
block_test_step = 20
model_save_step = 50
output_epoch = total_test_epoch * 10
edge_thickness = 20
original_g = 24
growth_d = 16
layer_num_d = 4
mask_type = 0
test_dir = './FU_LI_JUN/'
config={}
config['batch_size'] = batch_size
config['meta_path'] = '/opt/artery_extraction/data_meta_multi_class.pkl'
config['data_size'] = input_shape
config['test_amount'] = 2
config['train_amount'] = 8
config['max_epoch'] = MAX_EPOCH
config['mask_names'] = mask_names[:-1]
config['full_zero_num'] = 1
config['class_num'] = class_num
decay_step = 2 * 16 / (config['train_amount'] / 2)
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
            softmax_pred = tf.nn.softmax(relu_2,dim=-1,name="classify_probability")
            argmax_label = tf.argmax(softmax_pred,axis=-1)
        return relu_2,softmax_pred,argmax_label

    def E_Loss(self,logits,label):
        return tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits,dim=-1)

    def Concat(self,inputs,axis,size,name):
        with tf.variable_scope(name):
            concat_input = tf.concat([elem for elem in inputs],axis=axis,name="concat_input")
            concat_conv = tools.Ops.conv3d(concat_input,k=3,out_c=size,str=1,name="concat_conv")
        return concat_conv

    def Segmentor(self,X,training,batch_size):
        original = original_g
        growth = growth_d
        dense_layer_num = layer_num_d
        X_input = self.Input(X, "input", batch_size, original, training)
        down_1 = self.Down_Sample(X_input, "down_sample_1", 2, training, original * 1)
        dense_1 = self.Dense_Block(down_1, "dense_block_1", dense_layer_num, growth, training)
        down_2 = self.Down_Sample(dense_1, "down_sample_2", 2, training, original * 2)
        dense_2 = self.Dense_Block(down_2, "dense_block_2", dense_layer_num, growth, training)
        down_3 = self.Down_Sample(dense_2, "down_sample_3", 2, training, original * 4)

        dense_3 = self.Dense_Block(down_3, "dense_block_3", dense_layer_num, growth, training)
        mid_input = self.Concat([dense_3,
                                 self.Down_Sample(dense_2, "cross_1", 2, training, original),
                                 self.Down_Sample(dense_1, "cross_2", 4, training, original),
                                 self.Down_Sample(X_input, "cross_3", 8, training, original),
                                 ],
                                axis=4, size=original * 6, name="concat_up_mid")
        dense_4 = self.Dense_Block(mid_input, "dense_block_4", dense_layer_num, growth, training)

        up_input_1 = self.Concat([down_3, dense_4], axis=4, size=original * 8, name="up_input_1")
        up_1 = self.Up_Sample(up_input_1, "up_sample_1", 2, training, original * 4)

        dense_input_5 = self.Concat([up_1, dense_2], axis=4, size=original * 4, name="dense_input_5")
        dense_5 = self.Dense_Block(dense_input_5, "dense_block_5", dense_layer_num, growth, training)

        up_input_2 = self.Concat([dense_5, down_2], axis=4, size=original * 6, name="up_input_2")
        up_2 = self.Up_Sample(up_input_2, "up_sample_2", 2, training, original * 2)

        dense_input_6 = self.Concat([up_2, dense_1], axis=4, size=original * 2, name="dense_input_6")
        dense_6 = self.Dense_Block(dense_input_6, "dense_block_6", dense_layer_num, growth, training)

        up_input_3 = self.Concat([dense_6, down_1], axis=4, size=original * 6, name="up_input_3")
        up_3 = self.Up_Sample(up_input_3, "up_sample_3", 2, training, original * 1)

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

        return self.Pixel_Classifier(predict_input,"segment",training,class_num)

    def train(self,configure):
        #  data
        data = tools.Data_multi(configure, epoch_walked/re_example_epoch)
        # network
        self.X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2],class_num], dtype=tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)
        X = self.X
        Y = self.Y
        lr = self.lr
        training = self.training
        with tf.variable_scope('segment'):
            # [relu , softmax, argmax]
            self.pred_unsoft, self.softmax_pred, self.argmax_label = self.Segmentor(self.X, self.training, batch_size)
            pred_unsoft = self.pred_unsoft
            softmax_pred = self.softmax_pred
            argmax_label = self.argmax_label
        with tf.variable_scope('loss'):
            self.pixel_loss = self.E_Loss(softmax_pred,Y)
            pixel_loss = self.pixel_loss

        #  weight map
        weight_map = tf.convert_to_tensor(np.zeros(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]]),dtype=tf.float32)
        pred_masks = []
        for i in range(class_num):
            pred_masks.append(tf.cast(tf.equal(argmax_label,i),tf.float32))
        for i in range(class_num):
            weight_map = weight_map+weights[i]*Y[:,:,:,:,i]

        #  weighted loss
        enhanced_loss = weight_map * pixel_loss
        mean_enhanced_loss = tf.reduce_mean(enhanced_loss)
        #  accuracy
        accuracys = []
        for i in range(class_num):
            temp_pred = tf.cast(pred_masks[i],tf.float32)
            temp_mask = tf.cast(Y[:,:,:,:,i],tf.float32)
            accuracys.append(2*tf.reduce_sum(tf.abs(temp_pred)*tf.abs(temp_mask))/tf.reduce_sum((tf.abs(temp_pred)+tf.abs(temp_mask))))
        # [artery , airway , background]
        self.total_acc = tf.placeholder(tf.float32)
        artery_acc_sum = tf.summary.scalar("artery_accuracy",accuracys[0])
        airway_acc_sum = tf.summary.scalar("airway_accuracy",accuracys[1])
        background_acc_sum = tf.summary.scalar("background_acc",accuracys[2])
        total_acc_sum = tf.summary.scalar("total_accuracy",self.total_acc)
        normal_loss_sum = tf.summary.scalar("normal_loss",tf.reduce_mean(pixel_loss))
        enhanced_loss_sum = tf.summary.scalar("enhanced_loss",mean_enhanced_loss)
        train_merge_op = tf.summary.merge([artery_acc_sum,airway_acc_sum,background_acc_sum,normal_loss_sum,enhanced_loss_sum])
        self.test_merge_op = tf.summary.merge([total_acc_sum])
        test_merge_op = self.test_merge_op

        # trainer
        train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(mean_enhanced_loss)

        saver = tf.train.Saver(max_to_keep=1)

        with tf.Session() as sess:

            # summary writers
            sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
            self.sum_write_test = tf.summary.FileWriter(self.test_sum_dir, sess.graph)

            # load model data if pre-trained
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            if os.path.isfile(self.train_models_dir + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                saver.restore(sess, self.train_models_dir + 'model.cptk')
            learning_rate = ori_lr * pow(power, (epoch_walked / decay_step))

            # start main loop
            global_step = step_walked
            for epoch in range(epoch_walked, MAX_EPOCH):
                # re-example
                if epoch % re_example_epoch == 0 and epoch > 0:
                    del data
                    gc.collect()
                    data = tools.Data_multi(configure, epoch_walked / re_example_epoch)
                train_amount = len(data.train_numbers)
                test_amount = len(data.test_numbers)
                if train_amount >= test_amount and train_amount > 0 and test_amount > 0 and data.total_train_batch_num > 0 and data.total_test_seq_batch > 0:
                    if epoch % total_test_epoch == 0 and epoch > 0:
                        self.full_testing(sess,epoch)
                    data.shuffle_X_Y_pairs()
                    total_train_batch_num = data.total_train_batch_num
                    for i in range(total_train_batch_num):
                        X_train_batch, Y_train_batch = data.load_X_Y_voxel_train_next_batch()
                        if epoch % decay_step == 0 and epoch > epoch_walked and i == 0:
                            learning_rate = learning_rate * power
                        _, loss_val = sess.run([train_op,mean_enhanced_loss],feed_dict={X:X_train_batch,Y:Y_train_batch,
                                                                                   lr:learning_rate,training:True})
                        global_step += 1

                        # temporary block result show
                        if i % show_step == 0 and i > 0:
                            print "epoch:", epoch, " i: ", i, " , training loss : ", loss_val, " , learning rate : ",learning_rate

                        # block test
                        if i % block_test_step == 0:
                            X_test_batch, Y_test_batch = data.load_X_Y_voxel_test_next_batch(fix_sample=False)
                            loss_val,artery_acc_val,airway_acc_val,background_acc_val,train_summay, predic_label\
                                = sess.run([mean_enhanced_loss,accuracys[0],accuracys[1],accuracys[2],train_merge_op,argmax_label],
                                           feed_dict={X:X_test_batch,Y:Y_test_batch,training:False})
                            # if i == block_test_step and i > 0 and epoch % (output_epoch / 2) ==0:
                            #     ST.WriteImage(ST.GetImageFromArray(np.int16(np.transpose(predic_label[0,:,:,:],[2,1,0]))),self.test_results_dir+"epoch_"+str(epoch)+".vtk")
                            print "epoch:", epoch, " global step: ", global_step
                            print "artery accuracy : ",artery_acc_val, "airway accuracy : ",airway_acc_val
                            sum_writer_train.add_summary(train_summay,global_step=global_step)
                        if i % model_save_step == 0 and i > 0:
                            saver.save(sess, save_path=self.train_models_dir + 'model.cptk')
                            print "epoch:", epoch, " i:", i, "regular model saved!"
                else:
                    print "bad data , next epoch", epoch
        print "end training"

    # need to be modified
    def full_testing(self,sess,epoch):
        print '********************** FULL TESTING ********************************'
        time_begin = time.time()
        origin_data = read_dicoms(test_dir + "original1")
        mask_dir = test_dir + "artery"
        test_batch_size = batch_size
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
                pred_unsoft, softmax_pred, argmax_label = \
                sess.run([self.pred_unsoft, self.softmax_pred, self.argmax_label],
                    feed_dict={self.X: temp_input,
                               self.training: False})
                for j in range(test_batch_size):
                    test_data.upload_result_multiclass(batch_numbers[j], argmax_label[j, :, :, :],mask_type)
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
                with tf.variable_scope("segment",reuse=True):
                    temp_unsoft, softmax_temp,argmax_temp = self.Segmentor(X_temp,self.training,batch_size)
                    pred_unsoft_temp, softmax_pred_temp, argmax_label_temp = \
                        sess.run([temp_unsoft, softmax_temp,argmax_temp],feed_dict={X_temp:temp_input,
                                                                                    self.training:False})
                    for j in range(temp_batch_size):
                        test_data.upload_result_multiclass(batch_numbers[j],argmax_label_temp[j,:,:,:],mask_type)
        test_result_array = test_data.get_result_()
        print "result shape: ", np.shape(test_result_array)
        to_be_transformed = self.post_process(test_result_array)
        if epoch == 0:
            mask_img = ST.GetImageFromArray(np.transpose(array_mask, [2, 1, 0]))
            mask_img.SetSpacing(test_data.space)
            ST.WriteImage(mask_img, './test_result/test_mask.vtk')
        test_IOU = 2 * np.sum(to_be_transformed * array_mask) / (
                np.sum(to_be_transformed) + np.sum(array_mask))
        test_summary = sess.run(self.test_merge_op, feed_dict={self.total_acc: test_IOU})
        self.sum_write_test.add_summary(test_summary, global_step=epoch)
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


if __name__ == "__main__":
    dicom_dir = "./FU_LI_JUN/original1"
    net = Network()
    net.train(config)
    # final_img = net.test(dicom_dir)
    # ST.WriteImage(final_img,'./final_result.vtk')
