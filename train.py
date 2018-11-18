import os
from network import GAN
import tensorflow as tf
import numpy as np
import tools,gc,time
import scipy.io as sio
from evaluate import Evaluator
from data import TrainData, TestData
import cPickle as pickle
import SimpleITK as ST

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GANTrainier(GAN):
    def __init__(self, confPath = ""):
        GAN.__init__(self, confPath)
        pickle_reader = open(self.conf["metaDataPath"])
        meta_data = pickle.load(pickle_reader)
        self.showTimesUsed = self.conf["showTimesUsed"]
        self.test_datas = []
        self.test_values = {}
        self.test_merges = {}
        self.evaluator = Evaluator()
        for testName in self.conf["testDataName"]:
            found = False
            for dataName in meta_data.values():
                if testName in dataName:
                    found = True
                    self.test_datas.append(TestData(sio.loadmat(dataName), self.blockShape,
                                                    self.evaluator, testName, self.conf["sampleStep"]))
                    self.test_values[testName] = {}
                    self.test_values[testName]["placeHolders"] = {}
                    self.test_values[testName]["sums"] = {}
                    for eName, description in self.conf["evaluateIndicate"].items():
                        self.test_values[testName]["placeHolders"][eName] = tf.placeholder(tf.float32)
                        evaluateSum = tf.summary.scalar(description, self.test_values[testName]["placeHolders"][eName])
                        self.test_values[testName]["sums"][eName] = evaluateSum
                    self.test_merges[testName] = tf.summary.merge([eSum for eSum in self.test_values[testName]["sums"].values()])
                    break

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
        alpha = tf.random_uniform(shape=[self.batch_size, self.blockShape[0] * self.blockShape[1] * self.blockShape[2]],
                                  minval=0.0,maxval=1.0)
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

    def recordPics(self, origin, mask, predict):
        outputDict = {}
        outputDict["origin"] = origin
        outputDict["mask"] = mask
        outputDict["predict"] = predict
        sio.savemat(self.conf["resultPath"] + "blockShots" + str(self.global_step) + ".mat", outputDict)
        #TODO: show windows that dynamically show matrixes
        pass

    def blockTestGD(self, sess, epoch):
        X_test_batch, Y_test_batch = self.data.load_X_Y_voxel_test_next_batch(fix_sample=False)
        g_loss_t, gan_g_loss_t, gan_d_loss_t, Y_test_pred, Y_test_modi, Y_test_pred_nosig = \
            sess.run([self.cross_entropy, self.g_loss, self.d_loss, self.Y_pred,
                      self.Y_pred_modi, self.Y_pred_nosig],
                     feed_dict={self.X: X_test_batch,
                                self.threshold: self.upper_threshold,
                                self.Y: Y_test_batch, self.training: False,
                                self.w: self.weight_for})
        predict_result = np.float32(Y_test_modi > 0.01)
        predict_result = np.reshape(predict_result,
                                    [self.batch_size, self.blockShape[0], self.blockShape[1],
                                     self.blockShape[2]])
        print np.max(Y_test_pred)
        print np.min(Y_test_pred)
        predict_probablity = np.float32((Y_test_modi - 0.01) > 0)
        predict_probablity = np.reshape(predict_probablity,
                                        [self.batch_size, self.blockShape[0], self.blockShape[1],
                                         self.blockShape[2]])
        # IOU
        accuracy = 2 * np.sum(np.abs(predict_probablity * Y_test_batch)) / np.sum(
            np.abs(predict_result) + np.abs(Y_test_batch))
        print "epoch:", epoch, " global step: ", self.global_step, "\nIOU accuracy: ", accuracy, "\ntest ae loss:", g_loss_t, " gan g loss:", gan_g_loss_t, " gan d loss:", gan_d_loss_t
        print "weight of foreground : ", self.weight_for
        print "upper threshold of testing", (self.conf["predictThreshold"])
        if accuracy > 0.95 and self.showTimesUsed <= self.conf["showTimes"]:
            self.recordPics(origin=X_test_batch[self.batch_size / 2, :, :, self.blockShape[2] / 2],
                          mask=Y_test_batch[self.batch_size / 2, :, :, self.blockShape[2] / 2],
                          predict=predict_probablity[self.batch_size / 2, :, :, self.blockShape[2] / 2])
            self.showTimesUsed += 1
        blockTest_summary = sess.run(self.blockTest_merge_op, feed_dict={self.block_acc: accuracy})
        self.sum_writer_train.add_summary(blockTest_summary, global_step=self.global_step)

    def blockTrainGD(self, sess, epoch):
        if epoch % self.re_example_epoch == 0 and epoch > 0:
            del self.data
            gc.collect()
            self.data = TrainData(self.conf, epoch / self.re_example_epoch)
            self.data.check_data()
        train_amount = len(self.data.train_numbers)
        test_amount = len(self.data.test_numbers)
        if train_amount >= test_amount and train_amount > 0 and test_amount > 0 and \
                self.data.total_train_batch_num > 0 and self.data.total_test_seq_batch > 0:
            # actual foreground weight
            self.weight_for = 0.5 + (1 - 1.0 * epoch / self.MAX_EPOCH) * 0.85
            self.data.shuffle_X_Y_pairs()
            total_train_batch_num = self.data.total_train_batch_num
            print "total_train_batch_num:", total_train_batch_num
            for i in range(total_train_batch_num):
                X_train_batch, Y_train_batch = self.data.load_X_Y_voxel_train_next_batch()
                # calculate loss value
                # print "calculate begin"
                # print "calculate ended"
                if epoch % self.decay_step == 0 and epoch > self.epoch_walked and i == 0:
                    self.learning_rate_g = self.learning_rate_g * self.power
                sess.run([self.dis_optim, self.ae_g_optim],
                         feed_dict={self.X: X_train_batch, self.threshold: self.upper_threshold,
                                    self.Y: Y_train_batch, self.lr: self.learning_rate_g,
                                    self.training: True, self.w: self.weight_for})
                # print "training ended"
                self.global_step += 1
                # output some results
                if self.global_step % self.conf["recordStep"] == 0:
                    cross_entropy_c, gan_g_loss_c, gan_d_loss_c, train_summary \
                        = sess.run([self.cross_entropy, self.g_loss, self.d_loss, self.train_merge_op],
                                   feed_dict={self.X: X_train_batch, self.Y: Y_train_batch,
                                              self.training: False, self.w: self.weight_for,
                                              self.threshold: self.upper_threshold})
                    self.sum_writer_train.add_summary(train_summary, global_step=self.global_step)
                    print "epoch:", epoch, " i:", i, " cross entropy loss:", cross_entropy_c, " gan g loss:", gan_g_loss_c, " gan d loss:", gan_d_loss_c, " learning rate: ", self.learning_rate_g
                if self.global_step % self.conf["testStep"] == 0 and epoch % 1 == 0:
                    try:
                        self.blockTestGD(sess = sess, epoch = epoch)
                    except Exception, e:
                        print e
                #### model saving
                if self.global_step % self.conf["saveStep"] == 0 and epoch % 1 == 0:
                    self.saver.save(sess, save_path=self.conf["modelPath"] + 'model.cptk')
                    print "epoch:", epoch, " i:", i, "regular model saved!"
        else:
            print "bad data , next epoch", epoch

    def blockTrainD(self, sess, epoch):
        if epoch % self.re_example_epoch == 0 and epoch > 0:
            del self.data
            gc.collect()
            self.data = TrainData(self.conf, epoch / self.re_example_epoch)
            self.data.check_data()
        train_amount = len(self.data.train_numbers)
        test_amount = len(self.data.test_numbers)
        if train_amount >= test_amount and train_amount > 0 and test_amount > 0 and \
                self.data.total_train_batch_num > 0 and self.data.total_test_seq_batch > 0:
            # actual foreground weight
            self.weight_for = 0.5 + (1 - 1.0 * epoch / self.MAX_EPOCH) * 0.35
            self.data.shuffle_X_Y_pairs()
            total_train_batch_num = self.data.total_train_batch_num
            print "total_train_batch_num:", total_train_batch_num
            for i in range(total_train_batch_num):
                X_train_batch, Y_train_batch = self.data.load_X_Y_voxel_train_next_batch()
                # calculate loss value
                # print "calculate begin"
                # print "calculate ended"
                if epoch % self.decay_step == 0 and epoch > self.epoch_walked and i == 0:
                    self.learning_rate_g = self.learning_rate_g * self.power

                sess.run([self.dis_optim], feed_dict={self.X: X_train_batch, self.threshold: self.upper_threshold,
                                                      self.Y: Y_train_batch, self.lr: self.learning_rate_g,
                                                      self.training: True, self.w: self.weight_for})
                # print "training ended"
                self.global_step += 1
                # output some results
                if i % self.conf["recordStep"] == 0:
                    gan_d_loss_c, d_summary = sess.run([self.d_loss, self.d_sum],
                                         feed_dict={self.X: X_train_batch, self.Y: Y_train_batch, self.training: False,
                                                    self.w: self.weight_for, self.threshold: self.upper_threshold})
                    print "epoch:", epoch, " i:", i,  " gan d loss:", gan_d_loss_c, " learning rate: ", self.learning_rate_g
                    self.sum_writer_train.add_summary(d_summary, global_step = self.global_step)
                #### model saving
                if i % self.conf["saveStep"] == 0 and epoch % 1 == 0:
                    self.saver.save(sess, save_path=self.conf["modelPath"] + 'model.cptk')
                    print "epoch:", epoch, " i:", i, "regular model saved!"
        else:
            print "bad data , next epoch", epoch

    def trainLoop(self):
        with tf.Session() as sess:
            # define tensorboard writer
            self.sum_writers_test = {}
            self.sum_writer_train = tf.summary.FileWriter(self.conf["sumPathTrain"], sess.graph)
            for testName in self.conf["testDataName"]:
                if not os.path.exists(self.conf["sumPathTest"] + testName + "/"):
                    os.makedirs(self.conf["sumPathTest"] + testName + "/")
                self.sum_writers_test[testName] = tf.summary.FileWriter(self.conf["sumPathTest"] + testName + "/", sess.graph)
            # load model data if pre-trained
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            if os.path.isfile(self.conf["modelPath"] + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                self.saver.restore(sess, self.conf["modelPath"] + 'model.cptk')
            self.ori_lr = self.conf["learningRateOrigin"]
            self.power = self.conf["decayRate"]
            self.epoch_walked = self.conf["epochWalked"]
            self.decay_step = self.conf["decayEpoch"]
            self.re_example_epoch = self.conf["updateEpoch"]
            self.MAX_EPOCH = self.conf["maxEpoch"]
            self.upper_threshold = self.conf["predictThreshold"]
            self.learning_rate_g = self.ori_lr * pow(self.power, (self.epoch_walked / self.decay_step))
            # start training loop
            self.global_step = self.conf["stepWalked"]
            for epoch in range(self.epoch_walked, self.MAX_EPOCH):
                if epoch % self.conf["testEpoch"] == 0:
                    for tData in self.test_datas:
                        self.full_testing(tData, sess, epoch)
                if epoch < self.conf["discriminatorTrainEpoch"]:
                    self.blockTrainD(sess = sess, epoch = epoch)
                else:
                    self.blockTrainGD(sess = sess, epoch = epoch)

    def train(self):
        # network
        self.X = tf.placeholder(shape=[self.batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2]], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[self.batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2]], dtype=tf.float32)
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
        blockTest_sum = tf.summary.scalar("train_block_accuracy", self.block_acc)

        self.train_merge_op = tf.summary.merge([self.g_sum,self.d_sum,self.cross_entropy_sum])
        self.blockTest_merge_op = tf.summary.merge([blockTest_sum])

        self.saver = tf.train.Saver(max_to_keep=1)
        self.trainLoop()

    def full_testing(self, testData, sess, epoch):
        print '********************** FULL TESTING ********************************'
        self.weight_for = 0.5 + (1 - 1.0 * epoch / self.MAX_EPOCH) * 0.35
        time_begin = time.time()
        test_batch_size = self.conf["batchSize"]
        print "mask shape: ", np.shape(testData.mask_array)
        block_numbers = testData.blocks.keys()
        for i in range(0, len(block_numbers), test_batch_size):
            batch_numbers = []
            if i + test_batch_size < len(block_numbers):
                temp_input = np.zeros(
                    [test_batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2]])
                for j in range(test_batch_size):
                    temp_num = block_numbers[i + j]
                    temp_block = testData.blocks[temp_num]
                    batch_numbers.append(temp_num)
                    block_array = temp_block.get_data()
                    block_shape = temp_block.get_shape()
                    temp_input[j, 0:block_shape[0], 0:block_shape[1],
                    0:block_shape[2]] += block_array
                Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
                    [self.Y_pred, self.Y_pred_modi, self.Y_pred_nosig],
                    feed_dict={self.X: temp_input,
                               self.training: False,
                               self.w: self.weight_for,
                               self.threshold: self.upper_threshold})
                for j in range(test_batch_size):
                    testData.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])
            else:
                temp_batch_size = len(block_numbers) - i
                temp_input = np.zeros(
                    [temp_batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2]])
                for j in range(temp_batch_size):
                    temp_num = block_numbers[i + j]
                    temp_block = testData.blocks[temp_num]
                    batch_numbers.append(temp_num)
                    block_array = temp_block.get_data()
                    block_shape = temp_block.get_shape()
                    temp_input[j, 0:block_shape[0], 0:block_shape[1],0:block_shape[2]] += block_array
                X_temp = tf.placeholder(
                    shape=[temp_batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2]],
                    dtype=tf.float32)
                with tf.variable_scope('generator', reuse=True):
                    Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp = self.ae_u(X_temp, self.training,
                                                                                 temp_batch_size,
                                                                                 self.threshold)
                    Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
                        [Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp],
                        feed_dict={X_temp: temp_input,
                                   self.training: False,
                                   self.w: self.weight_for,
                                   self.threshold: self.upper_threshold})
                    for j in range(temp_batch_size):
                        testData.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])

        testData.get_result()
        print "result shape: ", np.shape(testData.test_result_array)
        testData.post_process(self.conf["clampThickness"])

        if epoch % self.conf["outputEpoch"] == 0:
            testData.output_img(epoch = epoch, test_results_dir = self.conf["resultPath"])
        if epoch == self.conf["outputEpoch"]:
            mask_img = ST.GetImageFromArray(np.transpose(testData.mask_array, [2, 1, 0]))
            mask_img.SetSpacing(testData.space)
            ST.WriteImage(mask_img, self.conf["resultPath"] + 'test_mask.vtk')

        evaluateFeed = testData.get_evaluate(self.test_values[testData.Name]["placeHolders"])
        test_summary = sess.run(self.test_merges[testData.Name],
                                feed_dict=evaluateFeed)
        self.sum_writers_test[testData.Name].add_summary(test_summary, global_step=epoch)
        time_end = time.time()
        print '******************** time of full testing: ' + str(time_end - time_begin) + 's ********************'

ganTrainier = GANTrainier("./conf.json")
ganTrainier.checkPaths()
ganTrainier.train()