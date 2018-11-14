from configure import Condifure
from data import Data
import tensorflow as tf
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




# net = GAN("./conf.json")
# print(type(net))