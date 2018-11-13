from configure import Condifure
from data import Data
import tensorflow as tf
import numpy as np
import tools,os

class Network:
    def __init__(self, configPath = ""):
        self.configure = Condifure(confPath = configPath)
        self.conf = self.configure.meta
        self.blockShape = self.conf["blockShape"]
        self.batch_size = self.conf["batchSize"]
        self.data_shape = self.conf["blockShape"]
        # epoch_walked/re_example_epoch

    def checkPaths(self):
        if not os.path.exists(self.conf.meta["resultPath"]):
            os.makedirs(self.conf.meta["resultPath"])
        if not os.path.exists(self.conf.meta["sumPathTest"]):
            os.makedirs(self.conf.meta["sumPathTest"])
        if not os.path.exists(self.conf.meta["modelPath"]):
            os.makedirs(self.conf.meta["modelPath"])
        if not os.path.exists(self.conf.meta["sumPathTrain"]):
            os.makedirs(self.conf.meta["sumPathTrain"])

    def DenseBlock(self,X,name,depth,growth,training):
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

    def DownSample(self,X,name,str,training,size):
        with tf.variable_scope(name):
            down_sample_input = tools.Ops.conv3d(X, k=3, out_c=size, str=1, name='down_sample_input')
            bn_input = tools.Ops.batch_norm(down_sample_input, "bn_input", training=training)
            relu_input = tools.Ops.xxlu(bn_input, name="relu_input")
            down_sample = tools.Ops.conv3d(relu_input, k=str, out_c=size, str=str, name='down_sample')
        return down_sample

    def UpSample(self,X,name,str,training,size):
        with tf.variable_scope(name):
            up_sample_input = tools.Ops.conv3d(X, k=3, out_c=size, str=1, name='up_sample_input')
            bn_1 = tools.Ops.batch_norm(up_sample_input, 'bn_after_dense_1', training=training)
            relu_1 = tools.Ops.xxlu(bn_1, name='relu_1')
            deconv_1 = tools.Ops.deconv3d(relu_1, k=2, out_c=size, str=str, name='deconv_up_sample_2')
        return deconv_1

    def Input(self,X,name,batch_size,size,training):
        with tf.variable_scope(name):
            X = tf.reshape(X, [batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2], 1])
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

class GAN(Network):
    def __init__(self, confPath = ""):
        Network.__init__(self, confPath)
        self.data = Data(self.conf,
                         self.conf["epochWalked"]/self.conf["updateEpoch"])

    def ae_u(self,X,training,batch_size,threshold):
        original=self.conf["network"]["generatorOriginSize"]
        growth=self.conf["network"]["denseBlockGrowth"]
        dense_layer_num=self.conf["network"]["denseBlockDepth"]
        X_input = self.Input(X,"input",batch_size,original,training)
        down_1 = self.DownSample(X_input,"down_sample_1",2,training,original*1)
        dense_1 = self.DenseBlock(down_1,"dense_block_1",dense_layer_num,growth,training)
        down_2 = self.DownSample(dense_1,"down_sample_2",2,training,original*2)
        dense_2 = self.DenseBlock(down_2,"dense_block_2",dense_layer_num,growth,training)
        down_3 = self.DownSample(dense_2,"down_sample_3",2,training,original*4)

        dense_3 = self.DenseBlock(down_3,"dense_block_3",dense_layer_num,growth,training)
        mid_input = self.Concat([dense_3,
                                  self.DownSample(dense_2, "cross_1", 2, training, original),
                                  self.DownSample(dense_1, "cross_2", 4, training, original),
                                  self.DownSample(X_input, "cross_3", 8, training, original),
                                  ],
                                 axis=4,size=original*6,name="concat_up_mid")
        dense_4 = self.DenseBlock(mid_input,"dense_block_4",dense_layer_num,growth,training)

        up_input_1 = self.Concat([down_3,dense_4],axis=4,size=original*8,name = "up_input_1")
        up_1 = self.UpSample(up_input_1,"up_sample_1",2,training,original*4)

        dense_input_5 = self.Concat([up_1,dense_2],axis=4,size=original*4,name = "dense_input_5")
        dense_5 = self.DenseBlock(dense_input_5,"dense_block_5",dense_layer_num,growth,training)

        up_input_2 = self.Concat([dense_5,down_2],axis=4,size=original*6,name = "up_input_2")
        up_2 = self.UpSample(up_input_2,"up_sample_2",2,training,original*2)

        dense_input_6 = self.Concat([up_2,dense_1],axis=4,size=original*2,name = "dense_input_6")
        dense_6 = self.DenseBlock(dense_input_6,"dense_block_6",dense_layer_num,growth,training)

        up_input_3 = self.Concat([dense_6,down_1],axis=4,size=original*6,name = "up_input_3")
        up_3 = self.UpSample(up_input_3,"up_sample_3",2,training,original*1)

        predict_input = self.Concat([up_3,
                                     self.UpSample(dense_6, "cross_4", 2, training, original),
                                     self.UpSample(up_2, "cross_5", 2, training, original),
                                     self.UpSample(dense_5, "cross_6", 4, training, original),
                                     self.UpSample(up_1, "cross_7", 4, training, original),
                                     self.UpSample(dense_4, "cross_8", 8, training, original),
                                     self.UpSample(mid_input, "cross_9", 8, training, original),
                                     self.UpSample(dense_3, "cross_10", 8, training, original)],
                                    axis=4,
                                    size=original * 4, name="predict_input")
        vox_sig, vox_sig_modified, vox_no_sig = self.Predict(predict_input, "predict", training, threshold)

        return vox_sig, vox_sig_modified, vox_no_sig

    def dis(self, X, Y,training):
        with tf.variable_scope("input"):
            X = tf.reshape(X, [self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2], 1])
            Y = tf.reshape(Y, [self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2], 1])
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
            y = tf.reshape(layers_d[-1], [self.batch_size, -1])
        return tf.nn.sigmoid(y)

    def calculateLoss(self, X, Y, Y_pred_modi, XY_real_pair, XY_fake_pair, Y_pred, training):
        # loss function
        # generator loss
        Y_ = tf.reshape(Y, shape=[self.batch_size, -1])
        Y_pred_modi_ = tf.reshape(Y_pred_modi, shape=[self.batch_size, -1])
        w = tf.placeholder(tf.float32)  # foreground weight
        g_loss = tf.reduce_mean(-tf.reduce_mean(w * Y_ * tf.log(Y_pred_modi_ + 1e-8), reduction_indices=[1]) -
                                tf.reduce_mean((1 - w) * (1 - Y_) * tf.log(1 - Y_pred_modi_ + 1e-8),
                                               reduction_indices=[1]))
        g_loss_sum = tf.summary.scalar("generator cross entropy", g_loss)
        self.cross_entropy_sum = g_loss_sum
        # discriminator loss
        gan_d_loss = tf.reduce_mean(XY_fake_pair) - tf.reduce_mean(XY_real_pair)
        alpha = tf.random_uniform(shape=[self.batch_size, self.data_shape[0] * self.data_shape[1] * self.data_shape[2]],
                                  minval=0.0,
                                  maxval=1.0)
        Y_pred_ = tf.reshape(Y_pred, shape=[self.batch_size, -1])
        differences_ = Y_pred_ - Y_
        interpolates = alpha * Y_ + (1 - alpha) * Y_pred_
        with tf.variable_scope('discriminator', reuse=True):
            XY_fake_intep = self.dis(X, interpolates, training)
        gradients = tf.gradients(XY_fake_intep, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        gan_d_loss += 10 * gradient_penalty
        gan_d_loss_sum = tf.summary.scalar("total loss of discriminator", gan_d_loss)
        self.d_loss = gan_d_loss
        self.d_sum = gan_d_loss_sum

        # generator loss with gan loss
        gan_g_loss = -tf.reduce_mean(XY_fake_pair)
        gan_g_w = 5
        ae_w = 100 - gan_g_w
        ae_gan_g_loss = ae_w * g_loss + gan_g_w * gan_g_loss
        ae_g_loss_sum = tf.summary.scalar("total loss of generator", ae_gan_g_loss)
        self.g_loss = ae_gan_g_loss
        self.g_sum = ae_g_loss_sum


    def train(self,configure):
        # network
        X = tf.placeholder(shape=[self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]], dtype=tf.float32)
        Y = tf.placeholder(shape=[self.batch_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]], dtype=tf.float32)
        lr = tf.placeholder(tf.float32)
        training = tf.placeholder(tf.bool)
        threshold = tf.placeholder(tf.float32)
        with tf.variable_scope('generator'):
            Y_pred, Y_pred_modi, Y_pred_nosig = self.ae_u(X, training, self.batch_size, threshold)
        with tf.variable_scope('discriminator'):
            XY_real_pair = self.dis(X, Y, training)
        with tf.variable_scope('discriminator', reuse=True):
            XY_fake_pair = self.dis(X, Y_pred, training)

        self.calculateLoss(X, Y, Y_pred_modi, XY_real_pair, XY_fake_pair, Y_pred, training)

        # trainers
        ae_var = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        dis_var = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        ae_g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
            self.g_loss, var_list=ae_var)
        dis_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
            self.d_loss, var_list=dis_var)

        # accuracy
        block_acc = tf.placeholder(tf.float32)
        total_acc = tf.placeholder(tf.float32)
        train_sum = tf.summary.scalar("train_block_accuracy", block_acc)
        test_sum = tf.summary.scalar("total_test_accuracy", total_acc)
        train_merge_op = tf.summary.merge([train_sum,self.g_sum,self.d_sum,self.cross_entropy_sum])
        test_merge_op = tf.summary.merge([test_sum])

        saver = tf.train.Saver(max_to_keep=1)
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.visible_device_list = GPU0

        with tf.Session() as sess:
            # define tensorboard writer
            sum_writer_train = tf.summary.FileWriter(self.conf["sumPathTrain"], sess.graph)
            sum_write_test = tf.summary.FileWriter(self.conf["sumPathTest"], sess.graph)
            # load model data if pre-trained
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))
            if os.path.isfile(self.conf["modelPath"] + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                saver.restore(sess, self.conf["modelPath"] + 'model.cptk')
            ori_lr = self.conf["learningRateOrigin"]
            power = self.conf["decayRate"]
            epoch_walked = self.conf["epochWalked"]
            decay_step = self.conf["decayEpoch"]
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
                                train_summary = sess.run(train_merge_op, feed_dict={block_acc: accuracy,
                                                                                    X: X_test_batch,
                                                                                    threshold: self.conf["predictThreshold"],
                                                                                    Y: Y_test_batch, training: False,
                                                                                    w: weight_for})
                                sum_writer_train.add_summary(train_summary, global_step=global_step)
                            except Exception, e:
                                print e
                        #### model saving
                        if i % self.conf["saveStep"] == 0 and epoch % 1 == 0:
                            saver.save(sess, save_path=self.conf["modelPath"] + 'model.cptk')
                            print "epoch:", epoch, " i:", i, "regular model saved!"
                else:
                    print "bad data , next epoch", epoch


# net = GAN("./conf.json")
# print(type(net))