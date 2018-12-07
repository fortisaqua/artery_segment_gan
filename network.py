from configure import Condifure
from data import TrainData
import tensorflow as tf
import tools,os

class Network:
    def __init__(self, configPath = ""):
        self.configure = Condifure(confPath = configPath)
        self.conf = self.configure.meta
        self.blockShape = self.conf["blockShape"]
        self.batch_size = self.conf["batchSize"]
        # epoch_walked/re_example_epoch

    def checkPaths(self):
        if not os.path.exists(self.conf["resultPath"]):
            os.makedirs(self.conf["resultPath"])
        if not os.path.exists(self.conf["modelPath"]):
            os.makedirs(self.conf["modelPath"])
        if not os.path.exists(self.conf["sumPathTrain"]):
            os.makedirs(self.conf["sumPathTrain"])

    # # artery

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

    # # airway
    # def Dense_Block(self,X,name,depth,growth,training):
    #     with tf.variable_scope(name):
    #         original = X.get_shape().as_list()[-1]
    #         c_e = []
    #         s_e = []
    #         layers = []
    #         layers.append(X)
    #         for i in range(depth):
    #             c_e.append(original + growth * (i + 1))
    #             s_e.append(1)
    #         for j in range(depth):
    #             with tf.variable_scope("input_"+str(j+1)):
    #                 input = tf.concat([sub_layer for sub_layer in layers], axis=4)
    #             with tf.variable_scope("dense_layer_"+str(j+1)):
    #                 layer = tools.Ops.batch_norm(input, 'bn_dense_1_2_' + str(j), training=training)
    #                 layer = tools.Ops.xxlu(layer, name='relu_2')
    #                 layer = tools.Ops.conv3d(layer, k=3, out_c=growth, str=s_e[j], name='dense_1_2_' + str(j+1))
    #             layers.append(layer)
    #         with tf.variable_scope("out_put"):
    #             ret = tf.concat([sub_layer for sub_layer in layers], axis=4)
    #     return ret
    #
    # def Down_Sample(self,X,name,str,training,size):
    #     with tf.variable_scope(name):
    #         bn_input = tools.Ops.batch_norm(X, "bn_input", training=training)
    #         relu_input = tools.Ops.xxlu(bn_input, name="relu_input")
    #         down_sample = tools.Ops.conv3d(relu_input, k=str, out_c=size, str=str, name='down_sample')
    #     return down_sample
    #
    # def Up_Sample(self,X,name,str,training,size):
    #     with tf.variable_scope(name):
    #         bn_1 = tools.Ops.batch_norm(X, 'bn_after_dense_1', training=training)
    #         relu_1 = tools.Ops.xxlu(bn_1, name='relu_1')
    #         deconv_1 = tools.Ops.deconv3d(relu_1, k=2, out_c=size, str=str, name='deconv_up_sample_2')
    #     return deconv_1
    #         # concat_up_1 = tf.concat([deconv_1, layers_e[-1]], axis=4, name="concat_up_1")
    #
    # def Input(self,X,name,batch_size,size,training):
    #     with tf.variable_scope(name):
    #         X = tf.reshape(X, [batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2], 1])
    #         conv_input = tools.Ops.conv3d(X, k=3, out_c=size, str=1, name="conv_input")
    #         bn_input = tools.Ops.batch_norm(conv_input, "bn_input", training)
    #         relu_input = tools.Ops.xxlu(bn_input, "relu_input")
    #     return relu_input
    #
    # def Predict(self,X,name,training,threshold):
    #     with tf.variable_scope(name):
    #         bn_1 = tools.Ops.batch_norm(X,"bn_predict_1",training)
    #         relu_1 = tools.Ops.xxlu(bn_1, name="relu_predict_1")
    #         predict_conv_1 = tools.Ops.conv3d(relu_1, k=3, out_c=16, str=1, name="conv_predict_1")
    #         predict_map = tools.Ops.conv3d(predict_conv_1, k=1, out_c=1, str=1, name="predict_map")
    #         # bn_2 = tools.Ops.batch_norm(predict_conv_2,"bn_predict_2",training)
    #         # relu_2 = tools.Ops.xxlu(predict_conv_2, name="relu_predict_2")
    #         vox_no_sig = predict_map
    #         vox_sig = tf.sigmoid(predict_map)
    #         vox_sig_modified = tf.maximum(vox_sig - threshold, 0.01)
    #     return vox_sig,vox_sig_modified,vox_no_sig
    #
    # def Concat(self,inputs,axis,size,name):
    #     with tf.variable_scope(name):
    #         concat_input = tf.concat([elem for elem in inputs],axis=axis,name="concat_input")
    #         concat_conv = tools.Ops.conv3d(concat_input,k=3,out_c=size,str=1,name="concat_conv")
    #     return concat_conv

class GAN(Network):
    def __init__(self, confPath = ""):
        Network.__init__(self, confPath)
        self.data = TrainData(self.conf,
                         self.conf["epochWalked"]/self.conf["updateEpoch"])
        self.data.check_data()

    def ae_u(self,X,training,batch_size,threshold):
        original = self.conf["network"]["generatorOriginSize"]
        growth = self.conf["network"]["denseBlockGrowth"]
        dense_layer_num = self.conf["network"]["denseBlockDepth"]
        # input layer
        X=tf.reshape(X,[batch_size,self.blockShape[0],self.blockShape[1],self.blockShape[2],1])
        # image reduce layer
        conv_input_1=tools.Ops.conv3d(X,k=3,out_c=original,str=2,name='conv_input_down')
        conv_input_normed=tools.Ops.batch_norm(conv_input_1, 'bn_dense_0_0', training=training)
        conv_input_relu = tools.Ops.xxlu(conv_input_normed)
        conv_input_conv1 = tools.Ops.conv3d(conv_input_relu,k=2,out_c=original,str=1,name='conv_input_conv1')
        conv_input_conv2 = tools.Ops.conv3d(conv_input_conv1,k=2,out_c=original,str=1,name='conv_input_conv2')
        conv_input_conv3 = tools.Ops.conv3d(conv_input_conv2,k=2,out_c=original,str=1,name='conv_input_conv3')
        # network start
        conv_input=tools.Ops.conv3d(conv_input_conv3,k=2,out_c=original * 2,str=2,name='conv_input')
        conv_input=tf.reshape(conv_input, [self.batch_size, self.blockShape[0]/4, self.blockShape[1]/4, original * 2], name="reshape_1")
        with tf.variable_scope("dense_part_1"):
            ##### dense block 1
            c_e = []
            s_e = []
            layers_1=[]
            layers_1.append(conv_input)
            for i in range(dense_layer_num):
                c_e.append(original+growth*(i+1))
                s_e.append(1)
            for j in range(dense_layer_num):
                layer = tools.Ops.batch_norm(layers_1[-1], 'bn_dense_1_' + str(j), training=training)
                layer = tools.Ops.xxlu(layer, name="relu_"+str(j))
                layer = tools.Ops.conv2d(layer,k=3,out_c=growth,str=s_e[j],name='dense_1_'+str(j))
                next_input = tf.concat([layer,layers_1[-1]],axis=3)
                layers_1.append(next_input)

        with tf.variable_scope("middle_down_sample"):
            mid_layer = tools.Ops.batch_norm(layers_1[-1], 'bn_mid', training=training)
            mid_layer = tools.Ops.xxlu(mid_layer,name='lrelu')
            mid_layer = tools.Ops.conv2d(mid_layer,k=3,out_c=original+growth*dense_layer_num,str=1,name='mid_conv')
            mid_layer_down = tools.Ops.conv2d(mid_layer,k=2,out_c=original+growth*dense_layer_num,str=2,name='mid_down')
            # mid_layer_down = tools.Ops.maxpool3d(mid_layer,k=2,s=2,pad='SAME')

        ##### dense block
        with tf.variable_scope("dense_part_2"):
            # lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=d1 * d2 * d3 * cc, name='fc2'),name='relu')
            # lfc = tf.reshape(lfc, [bat, d1, d2, d3, cc])

            c_d = []
            s_d = []
            layers_2 = []
            layers_2.append(mid_layer_down)
            for i in range(dense_layer_num):
                c_d.append(original+growth*(dense_layer_num+i+1))
                s_d.append(1)
            for j in range(dense_layer_num):
                layer = tools.Ops.batch_norm(layers_2[-1],'bn_dense_2_'+str(j),training=training)
                layer = tools.Ops.xxlu(layer, name="relu_"+str(j))
                layer = tools.Ops.conv2d(layer,k=3,out_c=growth,str=s_d[j],name='dense_2_'+str(j))
                next_input = tf.concat([layer,layers_2[-1]],axis=3)
                layers_2.append(next_input)

        with tf.variable_scope("up_sampling1"):
            bn_1 = tools.Ops.batch_norm(layers_2[-1],'bn_after_dense',training=training)
            relu_1 = tools.Ops.xxlu(bn_1, name='relu_after_dense')
            deconv_1 = tools.Ops.deconv2d(relu_1,k=2,out_c=128,str=2,name='deconv_up_sample_1')

            bn_middle = tools.Ops.batch_norm(deconv_1,'bn_middle',training=training)
            relu_middle = tools.Ops.xxlu(bn_middle, name='relu_middle')
            conv_middle1 = tools.Ops.conv2d(relu_middle,k=3,out_c=original+growth*dense_layer_num*2,str=1,name='conv_middle1')
            conv_middle2 = tools.Ops.conv2d(conv_middle1,k=3,out_c=original+growth*dense_layer_num*2,str=1,name='conv_up_sample_2')
            conv_middle3 = tools.Ops.conv2d(conv_middle2,k=3,out_c=original+growth*dense_layer_num,str=1,name='conv_up_sample_3')

        with tf.variable_scope("up_sampling2"):
            # concat_up_1 = tf.concat([conv_middle3,mid_layer],axis=3)
            concat_up_1 = conv_middle3 + mid_layer
            concat_up_1 = tf.reshape(concat_up_1, [self.batch_size, self.blockShape[0]/4, self.blockShape[1]/4, 1, concat_up_1.get_shape()[3]], name="reshape_2")
            deconv_2 = tools.Ops.deconv3d(concat_up_1,k=3,out_c=64,str=2,name='deconv_up_sample_2')

            bn_middle2 = tools.Ops.batch_norm(deconv_2,'bn_middle2',training=training)
            relu_middle2 = tools.Ops.xxlu(bn_middle2, "relu_middle2")
            conv_middle4 = tools.Ops.conv3d(relu_middle2,k=3,out_c=64,str=1,name='conv_middle4')
            conv_middle5 = tools.Ops.conv3d(conv_middle4,k=3,out_c=64,str=1,name='conv_middle5')
            conv_middle6 = tools.Ops.conv3d(conv_middle5,k=3,out_c=64,str=1,name='conv_middle6')
            concat_up_2 = tf.concat([conv_middle6, conv_input_1], axis=4)

        with tf.variable_scope("predict"):
            predict_map_normed = tools.Ops.batch_norm(concat_up_2,'bn_after_dense_1',training=training)
            predict_map_relued = tools.Ops.xxlu(predict_map_normed, "relu_predict_map")
            predict_map_zoomed = tools.Ops.deconv3d(predict_map_relued,k=3,out_c=original,str=2,name='deconv_zoom_3')
            predict_map = tools.Ops.conv3d(predict_map_zoomed, k=1, out_c=1, str=1, name="predict_map")

            vox_no_sig = predict_map
            # vox_no_sig = tools.Ops.xxlu(vox_no_sig,name='relu')
            vox_sig = tf.sigmoid(predict_map)
            vox_sig_modified = tf.maximum(vox_sig - threshold,0.01)
        return vox_sig, vox_sig_modified,vox_no_sig

    def dis(self, X, Y,training):
        with tf.variable_scope("input"):
            X = tf.reshape(X, [self.batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2], 1])
            Y = tf.reshape(Y, [self.batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2], 1])
            # layer = tf.concat([X, Y], axis=4)
            layer = X*Y
            c_d = [1, 32, 64, 128, 256, 512]
            s_d = [0, 2, 2, 2, 2, 2]
            layers_d = []
            layers_d.append(layer)
        with tf.variable_scope("down_sample"):
            for i in range(1,6,1):
                if i <= 2:
                    layer = tools.Ops.conv3d(layers_d[-1],k=4,out_c=c_d[i],str=s_d[i],name='d_1'+str(i))
                else:
                    shape = layer.get_shape()
                    if len(shape)>4:
                        layer = tf.reshape(layers_d[-1], [shape[0], shape[1], shape[2], shape[4]], name="reshape1")
                    else:
                        layer = layers_d[-1]
                    layer = tools.Ops.conv2d(layer, k=4, out_c=c_d[i], str=s_d[i], name='d_1' + str(i))
                if i!=5:
                    # batch normal layer
                    layer = tools.Ops.batch_norm(layer, 'bn_down' + str(i), training=training)
                    layer = tools.Ops.xxlu(layer, name='lrelu')
                layers_d.append(layer)
        with tf.variable_scope("flating"):
            y = tf.reshape(layers_d[-1], [self.batch_size, -1])
        return tf.nn.sigmoid(y)

class GANAirway(Network):
    def __init__(self, confPath = ""):
        Network.__init__(self, confPath)
        self.data = TrainData(self.conf,
                         self.conf["epochWalked"]/self.conf["updateEpoch"])
        self.data.check_data()
        self.GPU0 = '0'

    def ae_u(self,X,training,batch_size,threshold):
        original = self.conf["network"]["generatorOriginSize"]
        growth = self.conf["network"]["denseBlockGrowth"]
        dense_layer_num = self.conf["network"]["denseBlockDepth"]
        X = tf.reshape(X, [self.batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2], 1])
        # image reduce layer
        X_input = self.Input(X, "input", batch_size, original, training)
        down_1 = self.Down_Sample(X_input, "down_sample_1", 2, training, original)
        dense_1 = self.Dense_Block(down_1, "dense_block_1", dense_layer_num, growth, training)
        down_2 = self.Down_Sample(dense_1, "down_sample_2", 2, training, original + dense_layer_num * growth * 1)

        dense_2 = self.Dense_Block(down_2, "dense_block_2", dense_layer_num, growth, training)

        up_input_1 = self.Concat([down_2, dense_2,
                                  self.Down_Sample(down_1, "cross_1", 2, training, original),
                                  self.Down_Sample(X_input, "cross_2", 4, training, original)], axis=4,
                                 size=original + dense_layer_num * growth * 2, name="concat_up_1")
        up_1 = self.Up_Sample(up_input_1, "up_sample_1", 2, training, 128)

        up_input_2 = self.Concat([up_1, dense_1], axis=4, size=original + dense_layer_num * growth * 1,
                                 name="concat_up_2")
        up_2 = self.Up_Sample(up_input_2, "up_sample_2", 2, training, 64)

        predict_input = self.Concat([up_2, X_input,
                                     self.Up_Sample(dense_2, "cross_3", 4, training, original),
                                     self.Up_Sample(up_1, "cross_5", 2, training, original)], axis=4,
                                    size=64, name="predict_input")
        vox_sig, vox_sig_modified, vox_no_sig = self.Predict(predict_input, "predict", training, threshold)
        return vox_sig, vox_sig_modified, vox_no_sig

    def dis(self, X, Y,training):
        with tf.variable_scope("input"):
            X = tf.reshape(X, [self.batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2], 1])
            Y = tf.reshape(Y, [self.batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2], 1])
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

class GANArtery(Network):
    def __init__(self, confPath = ""):
        Network.__init__(self, confPath)
        self.data = TrainData(self.conf,
                         self.conf["epochWalked"]/self.conf["updateEpoch"])
        self.data.check_data()
        self.GPU0 = '0'

    def ae_u(self,X,training,batch_size,threshold):
        original = self.conf["network"]["generatorOriginSize"]
        growth = self.conf["network"]["denseBlockGrowth"]
        dense_layer_num = self.conf["network"]["denseBlockDepth"]
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
        vox_sig, vox_sig_modified, vox_no_sig = self.Predict(predict_input, "predict", training, threshold)

        return vox_sig, vox_sig_modified, vox_no_sig

    def dis(self, X, Y,training):
        with tf.variable_scope("input"):
            X = tf.reshape(X, [self.batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2], 1])
            Y = tf.reshape(Y, [self.batch_size, self.blockShape[0], self.blockShape[1], self.blockShape[2], 1])
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