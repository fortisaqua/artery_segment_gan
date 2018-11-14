import os
from network import GAN
import tensorflow as tf
import numpy as np
import tools,gc
from data import Data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GANTrainier(GAN):
    def __init__(self, confPath = ""):
        GAN.__init__(self, confPath)

    def calculateLoss(self):
        # loss function
        # generator loss
        Y_ = tf.reshape(self.Y, shape=[self.batch_size, -1])
        Y_pred_modi_ = tf.reshape(self.Y_pred_modi, shape=[self.batch_size, -1])
        cross_entropy = tf.reduce_mean(-tf.reduce_mean(self.w * Y_ * tf.log(Y_pred_modi_ + 1e-8), reduction_indices=[1]) -
                                tf.reduce_mean((1 - self.w) * (1 - Y_) * tf.log(1 - Y_pred_modi_ + 1e-8),
                                               reduction_indices=[1]))
        g_loss_sum = tf.summary.scalar("generator cross entropy", cross_entropy)
        self.cross_entropy = cross_entropy
        self.cross_entropy_sum = g_loss_sum
        # discriminator loss
        gan_d_loss = tf.reduce_mean(self.XY_fake_pair) - tf.reduce_mean(self.XY_real_pair)
        alpha = tf.random_uniform(shape=[self.batch_size, self.data_shape[0] * self.data_shape[1] * self.data_shape[2]],
                                  minval=0.0,
                                  maxval=1.0)
        Y_pred_ = tf.reshape(self.Y_pred, shape=[self.batch_size, -1])
        differences_ = Y_pred_ - Y_
        interpolates = alpha * Y_ + (1 - alpha) * Y_pred_
        with tf.variable_scope('discriminator', reuse=True):
            XY_fake_intep = self.dis(self.X, interpolates, self.training)
        gradients = tf.gradients(XY_fake_intep, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        gan_d_loss += 10 * gradient_penalty
        gan_d_loss_sum = tf.summary.scalar("total loss of discriminator", gan_d_loss)
        self.d_loss = gan_d_loss
        self.d_sum = gan_d_loss_sum

        # generator loss with gan loss
        gan_g_loss = -tf.reduce_mean(self.XY_fake_pair)
        gan_g_w = 5
        ae_w = 100 - gan_g_w
        ae_gan_g_loss = ae_w * cross_entropy + gan_g_w * gan_g_loss
        ae_g_loss_sum = tf.summary.scalar("total loss of generator", ae_gan_g_loss)
        self.g_loss = ae_gan_g_loss
        self.g_sum = ae_g_loss_sum

    def getTrainiers(self):
        # trainers
        ae_var = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        dis_var = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        self.ae_g_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
            self.g_loss, var_list=ae_var)
        self.dis_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
            self.d_loss, var_list=dis_var)

    def trainLoop(self):
        with tf.Session() as sess:
            # define tensorboard writer
            sum_writer_train = tf.summary.FileWriter(self.conf["sumPathTrain"], sess.graph)
            sum_write_test = tf.summary.FileWriter(self.conf["sumPathTest"], sess.graph)
            # load model data if pre-trained
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            if os.path.isfile(self.conf["modelPath"] + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                self.saver.restore(sess, self.conf["modelPath"] + 'model.cptk')
            ori_lr = self.conf["learningRateOrigin"]
            power = self.conf["decayRate"]
            epoch_walked = self.conf["epochWalked"]
            decay_step = self.conf["decayEpoch"]
            re_example_epoch = self.conf["updateEpoch"]
            MAX_EPOCH = self.conf["maxEpoch"]
            upper_threshold = self.conf["predictThreshold"]
            learning_rate_g = ori_lr * pow(power, (epoch_walked / decay_step))
            # start training loop
            global_step = self.conf["stepWalked"]
            for epoch in range(epoch_walked, MAX_EPOCH):
                if epoch % re_example_epoch == 0 and epoch > 0:
                    del self.data
                    gc.collect()
                    self.data = Data(self.conf, epoch/re_example_epoch)
                train_amount = len(self.data.train_numbers)
                test_amount = len(self.data.test_numbers)
                if train_amount >= test_amount and train_amount > 0 and test_amount > 0 and \
                        self.data.total_train_batch_num > 0 and self.data.total_test_seq_batch > 0:
                    # actual foreground weight
                    weight_for = 0.5 + (1-1.0*epoch/MAX_EPOCH)*0.25
                    if epoch % self.conf["testEpoch"] == 0:
                        pass
                        # self.full_testing(sess,X,w,threshold,
                        #                   test_merge_op,
                        #                   sum_write_test,training,
                        #                   weight_for,total_acc,Y_pred,
                        #                   Y_pred_modi,Y_pred_nosig,epoch)
                    self.data.shuffle_X_Y_pairs()
                    total_train_batch_num = self.data.total_train_batch_num
                    print "total_train_batch_num:", total_train_batch_num
                    for i in range(total_train_batch_num):
                        X_train_batch, Y_train_batch = self.data.load_X_Y_voxel_train_next_batch()
                        # calculate loss value
                        # print "calculate begin"
                        gan_d_loss_c, = sess.run([self.d_loss],
                                                 feed_dict={self.X: X_train_batch, self.Y: Y_train_batch, self.training: False,
                                                            self.w: weight_for, self.threshold: upper_threshold})
                        g_loss_c, gan_g_loss_c = sess.run([self.cross_entropy, self.g_loss],
                                                          feed_dict={self.X: X_train_batch, self.Y: Y_train_batch,
                                                                     self.training: False, self.w: weight_for,
                                                                     self.threshold: upper_threshold})
                        # print "calculate ended"
                        if epoch % decay_step == 0 and epoch > epoch_walked and i == 0:
                            learning_rate_g = learning_rate_g * power
                        sess.run([self.ae_g_optim],
                                 feed_dict={self.X: X_train_batch, self.threshold: upper_threshold, self.Y: Y_train_batch,
                                            self.lr: learning_rate_g, self.training: True, self.w: weight_for})
                        sess.run([self.dis_optim], feed_dict={self.X: X_train_batch, self.threshold: upper_threshold,
                                                              self.Y: Y_train_batch,self.lr: learning_rate_g,
                                                              self.training: True, self.w: weight_for})
                        # print "training ended"
                        global_step += 1
                        # output some results
                        if i % self.conf["recordStep"] == 0:
                            print "epoch:", epoch, " i:", i, " train ae loss:", g_loss_c, " gan g loss:", gan_g_loss_c, " gan d loss:", gan_d_loss_c, " learning rate: ", learning_rate_g
                        if i % self.conf["testStep"] == 0 and epoch % 1 == 0:
                            try:
                                X_test_batch, Y_test_batch = self.data.load_X_Y_voxel_test_next_batch(fix_sample=False)
                                g_loss_t, gan_g_loss_t, gan_d_loss_t, Y_test_pred, Y_test_modi, Y_test_pred_nosig = \
                                    sess.run([self.cross_entropy, self.g_loss, self.d_loss, self.Y_pred, self.Y_pred_modi, self.Y_pred_nosig],
                                             feed_dict={self.X: X_test_batch,
                                                        self.threshold: upper_threshold,
                                                        self.Y: Y_test_batch, self.training: False, self.w: weight_for})
                                predict_result = np.float32(Y_test_modi > 0.01)
                                predict_result = np.reshape(predict_result,
                                                            [self.batch_size, self.conf["blockShape"][0], self.conf["blockShape"][1],
                                                             self.conf["blockShape"][2]])
                                print np.max(Y_test_pred)
                                print np.min(Y_test_pred)
                                # IOU
                                predict_probablity = np.float32((Y_test_modi - 0.01) > 0)
                                predict_probablity = np.reshape(predict_probablity,
                                                                [self.batch_size, self.conf["blockShape"][0], self.conf["blockShape"][1],
                                                                 self.conf["blockShape"][2]])
                                accuracy = 2 * np.sum(np.abs(predict_probablity * Y_test_batch)) / np.sum(
                                    np.abs(predict_result) + np.abs(Y_test_batch))
                                print "epoch:", epoch, " global step: ", global_step, "\nIOU accuracy: ", accuracy, "\ntest ae loss:", g_loss_t, " gan g loss:", gan_g_loss_t, " gan d loss:", gan_d_loss_t
                                print "weight of foreground : ", weight_for
                                print "upper threshold of testing",(self.conf["predictThreshold"])
                                train_summary = sess.run(self.train_merge_op, feed_dict={self.block_acc: accuracy,
                                                                                         self.X: X_test_batch,
                                                                                         self.threshold: self.conf["predictThreshold"],
                                                                                         self.Y: Y_test_batch, self.training: False,
                                                                                         self.w: weight_for})
                                sum_writer_train.add_summary(train_summary, global_step=global_step)
                            except Exception, e:
                                print e
                        #### model saving
                        if i % self.conf["saveStep"] == 0 and epoch % 1 == 0:
                            self.saver.save(sess, save_path=self.conf["modelPath"] + 'model.cptk')
                            print "epoch:", epoch, " i:", i, "regular model saved!"
                else:
                    print "bad data , next epoch", epoch

    def train(self):
        # network
        self.X = tf.placeholder(shape=[self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]], dtype=tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self.w = tf.placeholder(tf.float32)  # foreground weight
        self.training = tf.placeholder(tf.bool)
        self.threshold = tf.placeholder(tf.float32)
        with tf.variable_scope('generator'):
            self.Y_pred, self.Y_pred_modi, self.Y_pred_nosig = self.ae_u(self.X, self.training, self.batch_size, self.threshold)
        with tf.variable_scope('discriminator'):
            self.XY_real_pair = self.dis(self.X, self.Y, self.training)
        with tf.variable_scope('discriminator', reuse=True):
            self.XY_fake_pair = self.dis(self.X, self.Y_pred, self.training)
        self.calculateLoss()
        self.getTrainiers()
        # accuracy
        self.block_acc = tf.placeholder(tf.float32)
        self.total_acc = tf.placeholder(tf.float32)
        train_sum = tf.summary.scalar("train_block_accuracy", self.block_acc)
        test_sum = tf.summary.scalar("total_test_accuracy", self.total_acc)
        self.train_merge_op = tf.summary.merge([train_sum,self.g_sum,self.d_sum,self.cross_entropy_sum])
        self.test_merge_op = tf.summary.merge([test_sum])
        self.saver = tf.train.Saver(max_to_keep=1)
        self.trainLoop()

ganTrainier = GANTrainier("./tumor_conf.json")
ganTrainier.train()