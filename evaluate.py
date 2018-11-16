import os
from network import GAN
import tensorflow as tf
import numpy as np
import tools,gc
from data import Data

class Evaluator:
    def __init__(self, confPath = ""):
        self.network = GAN(confPath = confPath)

    def full_testing(self,sess,X,w,threshold,test_merge_op,sum_write_test,training,weight_for,total_acc,Y_pred, Y_pred_modi, Y_pred_nosig,epoch):
        print '********************** FULL TESTING ********************************'
        # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        # w = tf.placeholder(tf.float32)
        # threshold = tf.placeholder(tf.float32)
        time_begin = time.time()
        origin_data = read_dicoms(test_dir + "original1")
        mask_dir = test_dir + "artery"
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
