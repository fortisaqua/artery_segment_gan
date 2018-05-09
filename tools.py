import numpy as np
import os
import re
from random import shuffle
import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from mpl_toolkits import mplot3d
import random
import organize_data
from dicom_read import read_dicoms
import SimpleITK as ST

class Data_block:
    # single input data block
    def __init__(self,ranger,data_array):
        self.ranger=ranger
        self.data_array=data_array

    def get_range(self):
        return self.ranger

    def load_data(self):
        return self.data_array

class Test_data():
    # load data and translate to original array
    def __init__(self,data,block_shape,type):
        if type == 'dicom_data':
            self.img = read_dicoms(data)
        elif type == 'vtk_data':
            self.img = data
        self.space = self.img.GetSpacing()
        self.image_array = ST.GetArrayFromImage(self.img)
        self.image_array = np.transpose(self.image_array,[2,1,0])
        self.image_shape = np.shape(self.image_array)
        if "airway" in type:
            self.image_array = np.int16(self.image_shape < 0) * self.image_shape
        self.block_shape=block_shape
        self.blocks=dict()
        self.results=dict()

    def output_origin(self):
        if not os.path.exists("./temp_output"):
            os.mkdir("./temp_output")
        ST.WriteImage(ST.GetImageFromArray(np.transpose(self.image_array,[2,1,0])),"./temp_output/original.vtk")

    # do the simple threshold function
    def threshold(self,low,high):
        mask_array=np.float32(np.float32(self.image_array<=high)*np.float32(self.image_array>=low))
        return np.float32(np.float32(self.image_array<=high)*np.float32(self.image_array>=low))

    def organize_blocks(self):
        block_num=0
        original_shape=np.shape(self.image_array)
        print 'data shape: ', original_shape
        for i in range(0,original_shape[0],self.block_shape[0]/2):
            for j in range(0,original_shape[1],self.block_shape[1]/2):
                for k in range(0,original_shape[2],self.block_shape[2]/2):
                    if i<original_shape[0] and j<original_shape[1] and k<original_shape[2]:
                        block_array = self.image_array[i:i+self.block_shape[0],j:j+self.block_shape[1],k:k+self.block_shape[2]]
                        block_shape = np.shape(block_array)
                        ranger=[i,i+block_shape[0],j,j+block_shape[1],k,k+block_shape[2]]
                        this_block=Data_block(ranger,self.image_array[i:i+self.block_shape[0],j:j+self.block_shape[1],k:k+self.block_shape[2]])
                        self.blocks[block_num]=this_block
                        block_num+=1

    def upload_result(self,block_num,result_array):
        ranger = self.blocks[block_num].get_range()
        partial_result = np.float32((result_array-0.01)>0)
        this_result = Data_block(ranger,partial_result)
        self.results[block_num] = this_result

    def upload_result_multiclass(self,block_num,result_array,type_num):
        ranger = self.blocks[block_num].get_range()
        partial_result = np.float32(result_array[:,:,:] == type_num)
        this_result = Data_block(ranger,partial_result)
        self.results[block_num] = this_result

    def get_result(self):
        ret=np.zeros(self.image_shape,np.float32)
        for number in self.results.keys():
            try:
                ranger=self.results[number].get_range()
                xmin=ranger[0]
                xmax=ranger[1]
                ymin=ranger[2]
                ymax=ranger[3]
                zmin=ranger[4]
                zmax=ranger[5]
                temp_result = self.results[number].load_data()[:,:,:,0]
                # temp_shape = np.shape(temp_result)
                ret[xmin:xmax,ymin:ymax,zmin:zmax]+=temp_result[:xmax-xmin,:ymax-ymin,:zmax-zmin]
            except Exception,e:
                print np.shape(self.results[number].load_data()[:,:,:,0]),self.results[number].get_range()
        return np.float32(ret>6)

    def get_result_(self):
        ret=np.zeros(self.image_shape,np.float32)
        for number in self.results.keys():
            try:
                ranger=self.results[number].get_range()
                xmin=ranger[0]
                xmax=ranger[1]
                ymin=ranger[2]
                ymax=ranger[3]
                zmin=ranger[4]
                zmax=ranger[5]
                temp_result = self.results[number].load_data()[:,:,:]
                # temp_shape = np.shape(temp_result)
                temp1 = ret[xmin:xmax,ymin:ymax,zmin:zmax]
                temp2 = temp_result[:xmax-xmin,:ymax-ymin,:zmax-zmin]
                ret[xmin:xmax,ymin:ymax,zmin:zmax]+=temp_result[:xmax-xmin,:ymax-ymin,:zmax-zmin]
            except Exception,e:
                print np.shape(self.results[number].load_data()[:,:,:]),self.results[number].get_range()
        return np.float32(ret>=4)

class Data:
    def __init__(self,config,epoch):
        self.config = config
        self.train_batch_index = 0
        self.test_seq_index = 0
        self.epoch = epoch
        self.batch_size = config['batch_size']
        self.test_amount = config['test_amount']
        self.train_amount = config['train_amount']
        self.data_size = config['data_size']

        self.train_numbers,self.test_numbers = self.load_X_Y_numbers_special(config['meta_path'],self.epoch)

        print "train_numbers:",len(self.train_numbers),"---",self.train_numbers
        print "test_numbers:",len(self.test_numbers),"---",self.test_numbers
        self.total_train_batch_num,self.train_locs = self.load_X_Y_train_batch_num()
        self.total_test_seq_batch,self.test_locs = self.load_X_Y_test_batch_num()
        print "total_train_batch_num: ", self.total_train_batch_num
        print "total_test_seq_batch: ",self.total_test_seq_batch
        self.shuffle_X_Y_pairs()

    def load_X_Y_numbers_special(self,meta_path,epoch):
        self.dicom_origin,self.mask ,zero_numbers= organize_data.get_organized_data(meta_path,self.data_size,epoch,self.train_amount)
        numbers=[]
        train_numbers=[]
        test_numbers=[]
        for number in self.mask.keys():
            if len(self.mask[number])>0:
                numbers.append(number)
        for i in range(self.test_amount):
            test_number_temp = numbers[random.randint(0,len(numbers)-1)]
            while test_number_temp in zero_numbers:
                test_number_temp = numbers[random.randint(0, len(numbers) - 1)]
            test_numbers.append(test_number_temp)
        for number in numbers:
            if not number in test_numbers:
                train_numbers.append(number)
        return train_numbers,test_numbers

    def load_X_Y_train_batch_num(self):
        total_num=0
        locs=[]
        for number in self.train_numbers:
            for i in range(len(self.mask[number])):
                total_num=total_num+1
                locs.append([number,i])
        return int(total_num/self.batch_size),locs

    def load_X_Y_test_batch_num(self):
        total_num = 0
        locs=[]
        for number in self.test_numbers:
            for i in range(len(self.mask[number])):
                total_num = total_num + 1
                locs.append([number,i])
        return int(total_num / self.batch_size),locs

    def shuffle_X_Y_pairs(self):
        train_locs_new=[]
        test_locs_new=[]
        trains=self.train_locs
        tests=self.test_locs
        self.train_batch_index = 0
        train_index = range(len(trains))
        test_index = range(len(tests))
        shuffle(train_index)
        shuffle(test_index)
        for i in train_index:
            train_locs_new.append(trains[i])
        for j in test_index:
            test_locs_new.append(tests[j])
        self.train_locs=train_locs_new
        self.test_locs=test_locs_new

    def load_X_Y_voxel_train_next_batch(self):
        temp_locs=self.train_locs[self.batch_size*self.train_batch_index:self.batch_size*(self.train_batch_index+1)]
        X_data_voxels=[]
        Y_data_voxels=[]
        for pair in temp_locs:
            X_data_voxels.append(self.dicom_origin[pair[0]][pair[1]])
            Y_data_voxels.append(self.mask[pair[0]][pair[1]])
        self.train_batch_index += 1
        X_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2]],np.float32)
        Y_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2]],np.float32)
        for i in range(len(X_data_voxels)):
            temp_X = X_data_voxels[i][:,:,:]
            temp_y = Y_data_voxels[i][:,:,:]
            shape_X = np.shape(temp_X)
            shape_Y = np.shape(temp_y)
            X_data[i,:shape_X[0],:shape_X[1],:shape_X[2]] = X_data_voxels[i][:,:,:]
            Y_data[i,:shape_Y[0],:shape_Y[1],:shape_Y[2]] = Y_data_voxels[i][:,:,:]

        return X_data,Y_data

    def load_X_Y_voxel_test_next_batch(self,fix_sample=False):
        if fix_sample:
            random.seed(45)
        idx = random.sample(range(len(self.test_locs)), self.batch_size)
        X_test_voxels_batch=[]
        Y_test_voxels_batch=[]
        for i in idx:
            temp_pair=self.test_locs[i]
            X_test_voxels_batch.append(self.dicom_origin[temp_pair[0]][temp_pair[1]])
            Y_test_voxels_batch.append(self.mask[temp_pair[0]][temp_pair[1]])
        X_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2]],np.float32)
        Y_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2]],np.float32)
        '''
        X_test_voxels_batch=np.asarray(X_test_voxels_batch)
        Y_test_voxels_batch=np.asarray(Y_test_voxels_batch)
        '''
        for i in range(len(X_test_voxels_batch)):
            temp_X = X_test_voxels_batch[i][:,:,:]
            temp_y = Y_test_voxels_batch[i][:,:,:]
            shape_X = np.shape(temp_X)
            shape_Y = np.shape(temp_y)
            X_data[i,:shape_X[0],:shape_X[1],:shape_X[2]] = X_test_voxels_batch[i][:,:,:]
            Y_data[i,:shape_Y[0],:shape_Y[1],:shape_Y[2]] = Y_test_voxels_batch[i][:,:,:]
        return X_data,Y_data

    ###################  check datas
    def check_data(self):
        fail_list=[]
        tag=True
        for pair in self.train_locs:
            shape1 = np.shape(self.dicom_origin[pair[0]][pair[1]])
            shape2 = np.shape(self.mask[pair[0]][pair[1]])
            if shape1[0]==shape2[0]==self.data_size[0] and shape1[1]==shape2[1]==self.data_size[1] and shape1[2]==shape2[2]==self.data_size[2]:
                tag=True
            else:
                tag=False
                fail_list.append(pair)
        for pair in self.test_locs:
            shape1 = np.shape(self.dicom_origin[pair[0]][pair[1]])
            shape2 = np.shape(self.mask[pair[0]][pair[1]])
            if shape1[0]==shape2[0]==self.data_size[0] and shape1[1]==shape2[1]==self.data_size[1] and shape1[2]==shape2[2]==self.data_size[2]:
                tag=True
            else:
                tag=False
                fail_list.append(pair)
                print shape1
                print shape2
                print "=============================================="
        if tag:
            print "checked!"
        else:
            print "some are failed"
            for item in fail_list:
                print item

class Data_multi():
    def __init__(self,config,epoch):
        self.config = config
        self.train_batch_index = 0
        self.test_seq_index = 0
        self.epoch = epoch
        self.batch_size = config['batch_size']
        self.test_amount = config['test_amount']
        self.train_amount = config['train_amount']
        self.data_size = config['data_size']
        self.max_epoch = config['max_epoch']
        self.mask_names = config['mask_names']
        self.full_zero_num = config['full_zero_num']
        self.class_num = config['class_num']
        self.train_numbers,self.test_numbers = self.load_X_Y_numbers_special(config['meta_path'],self.epoch)

        print "train_numbers:",len(self.train_numbers),"---",self.train_numbers
        print "test_numbers:",len(self.test_numbers),"---",self.test_numbers
        self.total_train_batch_num,self.train_locs = self.load_X_Y_train_batch_num()
        self.total_test_seq_batch,self.test_locs = self.load_X_Y_test_batch_num()
        print "total_train_batch_num: ", self.total_train_batch_num
        print "total_test_seq_batch: ",self.total_test_seq_batch
        self.check_data()

    def load_X_Y_numbers_special(self,meta_path,epoch):
        self.dicom_origin,self.mask ,zero_numbers= organize_data.get_multi_data(meta_path,self.data_size,epoch,
                                                                                self.train_amount,self.max_epoch,
                                                                                self.mask_names,self.full_zero_num)
        numbers=[]
        train_numbers=[]
        test_numbers=[]
        for number in self.mask.keys():
            if len(self.mask[number])>0:
                numbers.append(number)
        for i in range(self.test_amount):
            test_number_temp = numbers[random.randint(0,len(numbers)-1)]
            while test_number_temp in zero_numbers:
                test_number_temp = numbers[random.randint(0, len(numbers) - 1)]
            test_numbers.append(test_number_temp)
        for number in numbers:
            if not number in test_numbers:
                train_numbers.append(number)
        return train_numbers,test_numbers

    def load_X_Y_train_batch_num(self):
        total_num=0
        locs=[]
        for number in self.train_numbers:
            for i in range(len(self.mask[number])):
                total_num=total_num+1
                locs.append([number,i])
        return int(total_num/self.batch_size),locs

    def load_X_Y_test_batch_num(self):
        total_num = 0
        locs=[]
        for number in self.test_numbers:
            for i in range(len(self.mask[number])):
                total_num = total_num + 1
                locs.append([number,i])
        return int(total_num / self.batch_size),locs

    def shuffle_X_Y_pairs(self):
        train_locs_new=[]
        test_locs_new=[]
        trains=self.train_locs
        tests=self.test_locs
        self.train_batch_index = 0
        train_index = range(len(trains))
        test_index = range(len(tests))
        shuffle(train_index)
        shuffle(test_index)
        for i in train_index:
            train_locs_new.append(trains[i])
        for j in test_index:
            test_locs_new.append(tests[j])
        self.train_locs=train_locs_new
        self.test_locs=test_locs_new

    def load_X_Y_voxel_train_next_batch(self):
        temp_locs=self.train_locs[self.batch_size*self.train_batch_index:self.batch_size*(self.train_batch_index+1)]
        X_data_voxels=[]
        Y_data_voxels=[]
        for pair in temp_locs:
            X_data_voxels.append(self.dicom_origin[pair[0]][pair[1]])
            Y_data_voxels.append(self.mask[pair[0]][pair[1]])
        self.train_batch_index += 1
        X_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2]],np.float32)
        Y_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2],self.class_num],np.float32)
        for i in range(len(X_data_voxels)):
            temp_X = X_data_voxels[i][:,:,:]
            shape_X = np.shape(temp_X)
            X_data[i,:shape_X[0],:shape_X[1],:shape_X[2]] = X_data_voxels[i][:,:,:]

            temp_y = Y_data_voxels[i]["artery"][:,:,:]
            shape_Y = np.shape(temp_y)
            Y_data[i, :shape_Y[0], :shape_Y[1], :shape_Y[2],0] = Y_data_voxels[i]["artery"][:, :, :]
            Y_data[i, :shape_Y[0], :shape_Y[1], :shape_Y[2],1] = Y_data_voxels[i]["airway"][:, :, :]
            Y_data[i, :shape_Y[0], :shape_Y[1], :shape_Y[2],2] = Y_data_voxels[i]["background"][:, :, :]
        return X_data,Y_data

    def load_X_Y_voxel_test_next_batch(self,fix_sample=False):
        if fix_sample:
            random.seed(45)
        idx = random.sample(range(len(self.test_locs)), self.batch_size)
        X_test_voxels_batch=[]
        Y_test_voxels_batch=[]
        for i in idx:
            temp_pair=self.test_locs[i]
            X_test_voxels_batch.append(self.dicom_origin[temp_pair[0]][temp_pair[1]])
            Y_test_voxels_batch.append(self.mask[temp_pair[0]][temp_pair[1]])
        X_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2]],np.float32)
        Y_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2],self.class_num],np.float32)
        for i in range(len(X_test_voxels_batch)):
            temp_X = X_test_voxels_batch[i][:, :, :]
            shape_X = np.shape(temp_X)
            X_data[i, :shape_X[0], :shape_X[1], :shape_X[2]] = X_test_voxels_batch[i][:, :, :]

            temp_y = Y_test_voxels_batch[i]["artery"][:, :, :]
            shape_Y = np.shape(temp_y)
            Y_data[i, :shape_Y[0], :shape_Y[1], :shape_Y[2], 0] = Y_test_voxels_batch[i]["artery"][:, :, :]
            Y_data[i, :shape_Y[0], :shape_Y[1], :shape_Y[2], 1] = Y_test_voxels_batch[i]["airway"][:, :, :]
            Y_data[i, :shape_Y[0], :shape_Y[1], :shape_Y[2], 2] = Y_test_voxels_batch[i]["background"][:, :, :]
        return X_data,Y_data

    ###################  check datas
    def check_data(self):
        fail_list = []
        tag = True
        for pair in self.train_locs:
            test_array = np.zeros(self.data_size,np.uint8)
            for name in self.mask_names:
                test_array = test_array + self.mask[pair[0]][pair[1]][name]
            test_array = test_array + self.mask[pair[0]][pair[1]]["background"]
            if np.max(test_array) == np.min(test_array) ==1:
                tag = True
            else:
                tag = False
                fail_list.append(pair)
        for pair in self.test_locs:
            test_array = np.zeros(self.data_size, np.uint8)
            for name in self.mask_names:
                test_array = test_array + self.mask[pair[0]][pair[1]][name]
            test_array = test_array + self.mask[pair[0]][pair[1]]["background"]
            if np.max(test_array) == np.min(test_array) == 1:
                tag = True
            else:
                tag = False
                fail_list.append(pair)
                print "=============================================="
        if tag:
            print "checked!"
        else:
            print "some are failed"
            for item in fail_list:
                print item


class Ops:

    @staticmethod
    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def xxlu(x,name='relu'):
        with tf.variable_scope(name):
            if name =='relu':
                return  Ops.relu(x)
            else:
                return  Ops.lrelu(x,leak=0.2)

    @staticmethod
    def variable_sum(var, name):
        with tf.name_scope(name):
            try:
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
            except Exception,e:
                print e

    @staticmethod
    def variable_count():
        total_para = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        return total_para

    @staticmethod
    def fc(x, out_d, name):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            in_d = x.get_shape()[1]
            w = tf.get_variable(name + '_w', [in_d, out_d], initializer=xavier_init)
            b = tf.get_variable(name + '_b', [out_d], initializer=zero_init)
            y = tf.nn.bias_add(tf.matmul(x, w), b)
            Ops.variable_sum(w, name)
            return y

    @staticmethod
    def maxpool3d(x,k,s,pad='SAME'):
        ker =[1,k,k,k,1]
        str =[1,s,s,s,1]
        y = tf.nn.max_pool3d(x,ksize=ker,strides=str,padding=pad)
        return y

    @staticmethod
    def conv3d(x, k, out_c, str, name,pad='SAME'):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            in_c = x.get_shape()[4]
            w = tf.get_variable(name + '_w', [k, k, k, in_c, out_c], initializer=xavier_init)
            b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)

            stride = [1, str, str, str, 1]
            y = tf.nn.bias_add(tf.nn.conv3d(x, w, stride, pad), b)
            Ops.variable_sum(w, name)
            return y

    @staticmethod
    def deconv3d(x, k, out_c, str, name,pad='SAME'):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            bat, in_d1, in_d2, in_d3, in_c = [int(d) for d in x.get_shape()]
            w = tf.get_variable(name + '_w', [k, k, k, out_c, in_c], initializer=xavier_init)
            b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
            out_shape = [bat, in_d1 * str, in_d2 * str, in_d3 * str, out_c]
            stride = [1, str, str, str, 1]
            y = tf.nn.conv3d_transpose(x, w, output_shape=out_shape, strides=stride, padding=pad)
            y = tf.nn.bias_add(y, b)
            Ops.variable_sum(w, name)
            return y

    @staticmethod
    def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999):
        '''Assume 2d [batch, values] tensor'''

        with tf.variable_scope(name_scope):
            size = x.get_shape().as_list()[-1]
            x_shape = x.get_shape()
            axis = list(range(len(x_shape) - 1))
            scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
            offset = tf.get_variable('offset', [size])

            pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
            pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
            batch_mean, batch_var = tf.nn.moments(x, axis)

            train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

            def batch_statistics():
                with tf.control_dependencies([train_mean_op, train_var_op]):
                    return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

            def population_statistics():
                return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

            return tf.cond(training, batch_statistics, population_statistics)
