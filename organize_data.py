import SimpleITK as ST
import numpy as np
import cPickle as pickle
import scipy.io as sio
import sys
# import zoom
import time
import random

def get_range(mask,type=0):
    begin_switch = 0
    begin = 0
    end_switch = 0
    end = np.shape(mask)[2] - 1
    for i in range(np.shape(mask)[2]):
        if np.max(mask[:, :, i]) == 1 and begin_switch == 0:
            begin_switch = 1
            begin = i
        if np.max(mask[:, :, i]) == 0 and end_switch == 0 and begin_switch == 1:
            end = i
            end_switch = 1
        if end_switch:
            break
    return begin,end

def get_organized_data_fixed_2D(meta_path, type, half_size):
    dicom_datas = dict()
    clipped_datas = dict()
    pickle_readier = open(meta_path)
    meta_data = pickle.load(pickle_readier)
    for number, dataset in meta_data['matrixes'].items():
        try:
            patient_data = sio.loadmat(dataset['PATIENT_DICOM'])
            mask_data = sio.loadmat(dataset[type])
            original_array = patient_data['original_resized']
            mask = mask_data[type + '_mask_resized']
            dicom_datas[number] = list()
            clipped_datas[number] = list()
            # get the binary mask
            mask = np.int8(mask > 0)
            if np.max(mask) <= 0:
                continue
            # get the valid mask area
            begin, end = get_range(mask,0)
            origin = original_array[:, :, begin:end]
            # clip = original_array[:,:,begin:end]*mask[:,:,begin:end]
            clip = mask[:, :, begin:end]
            # if number=='5':
            #     dicom_img = ST.GetImageFromArray(np.transpose(dicom_datas[number],(2,1,0)) )
            #     clipped_img = ST.GetImageFromArray(np.transpose(clipped_data[number],(2,1,0)) )
            #     ST.WriteImage(dicom_img,'./dicom_img.vtk')
            #     ST.WriteImage(clipped_img,'./clipped_img.vtk')
            #     exit(0)
            print "valid area: ", begin, ":", end
            for i in range(begin, end, half_size):
                origin_slice = original_array[:, :, i - half_size:i + half_size]
                clip_slice = mask[:, :, i - half_size:i + half_size]
                if not 0 in np.shape(origin_slice) and not 0 in np.shape(clip_slice):
                    if np.shape(origin_slice)[-1] == half_size * 2 and np.shape(clip_slice)[-1] == half_size * 2:
                        dicom_datas[number].append(origin_slice)
                        clipped_datas[number].append(clip_slice)
        except Exception, e:
            print e
    return dicom_datas, clipped_datas

# def resize_img(img_array,input_size):
#     shape = np.shape(img_array)
#     ret = img_array
#     if shape[0]<input_size[0] or shape[1]<input_size[1]:
#         ret = zoom.Array_Zoom_in(img_array,float(input_size[0])/float(shape[0]),float(input_size[1])/float(shape[1]))
#     if shape[0]>input_size[0] or shape[1]>input_size[1]:
#         ret = zoom.Array_Reduce(img_array,float(input_size[0])/float(shape[0]),float(input_size[1])/float(shape[1]))
#     shape_resized=np.shape(ret)
#     if shape_resized[0]<input_size[0] or shape_resized[1]<input_size[1]:
#         return_array = np.zeros(input_size,dtype=np.float32)
#         return_array[0:shape_resized[0],0:shape_resized[1],:]=ret[:,:,:]
#         ret = return_array
#     if shape_resized[0]>input_size[0] or shape_resized[1]>input_size[1]:
#         ret = ret[0:input_size[0],0:input_size[1],:]
#     return ret

def get_organized_data(meta_path, single_size,epoch,train_amount):
    rand = random.Random()
    dicom_datas = dict()
    mask_datas = dict()
    pickle_reader = open(meta_path)
    meta_data = pickle.load(pickle_reader)
    # accept_zeros = rand.sample(meta_data.keys(),8)
    total_keys = meta_data.keys()
    begin = (epoch*(train_amount/2))%len(total_keys)
    end = (epoch*(train_amount/2)+train_amount)%len(total_keys)
    if begin<end:
        to_be_trained = total_keys[begin:end]
    else:
        to_be_trained = total_keys[begin:]+total_keys[:end]
    accept_zeros = rand.sample(to_be_trained, 1)
    # for i in range(8):
    #     accept_zeros = to_be_trained[accept_zeros[i]]
    for number,data_dir in meta_data.items():
        if number in to_be_trained:
            print number
            zero_counting = 0
            dataset = sio.loadmat(data_dir)
            dicom_datas[number]=list()
            mask_datas[number]=list()
            original_array = dataset['original']
            mask_array = dataset['mask']
            data_shape = np.shape(mask_array)
            for i in range(0,data_shape[0],single_size[0]/2):
                for j in range(0,data_shape[1],single_size[1]/2):
                    for k in range(0,data_shape[2],single_size[2]/2):
                        if i+single_size[0]/2<data_shape[0] and j+single_size[1]/2<data_shape[1] and k+single_size[2]/2<data_shape[2]:
                            clipped_mask = mask_array[i:i+single_size[0],j:j+single_size[1],k:k+single_size[2]]
                            if np.sum(np.float32(clipped_mask))/(single_size[0]*single_size[1]*single_size[2])<=(0.05*(1-epoch*1.0/2000)) and number in accept_zeros:
                                clipped_dicom = original_array[i:i+single_size[0],j:j+single_size[1],k:k+single_size[2]]
                                dicom_datas[number].append(clipped_dicom)
                                mask_datas[number].append(clipped_mask)
                            elif np.sum(np.float32(clipped_mask))/(single_size[0]*single_size[1]*single_size[2])>(0.05*(1-epoch*1.0/2000)):
                                clipped_dicom = original_array[i:i + single_size[0], j:j + single_size[1], k:k + single_size[2]]
                                dicom_datas[number].append(clipped_dicom)
                                mask_datas[number].append(clipped_mask)
                        # clipped_dicom = original_array[i:i + single_size[0], j:j + single_size[1], k:k + single_size[2]]
                        # dicom_datas[number].append(clipped_dicom)
                        # mask_datas[number].append(clipped_mask)
    return dicom_datas,mask_datas,accept_zeros

# method of getting many masks for multi-class classifying job on pixel level
def get_multi_data(meta_path, single_size,epoch,train_amount,max_epoch,mask_names,full_zero_num):
    rand = random.Random()
    dicom_datas = dict()
    mask_datas = dict()
    pickle_reader = open(meta_path)
    meta_data = pickle.load(pickle_reader)
    # accept_zeros = rand.sample(meta_data.keys(),8)
    total_keys = meta_data.keys()
    begin = (epoch*(train_amount/2))%len(total_keys)
    end = (epoch*(train_amount/2)+train_amount)%len(total_keys)
    if begin<end:
        to_be_trained = total_keys[begin:end]
    else:
        to_be_trained = total_keys[begin:]+total_keys[:end]
    accept_zeros = rand.sample(to_be_trained, full_zero_num)
    # for i in range(8):
    #     accept_zeros = to_be_trained[accept_zeros[i]]
    for number,data_dir in meta_data.items():
        if number in to_be_trained:
            print number
            dataset = sio.loadmat(data_dir)
            dicom_datas[number] = list()
            mask_datas[number] = list()
            original_array = dataset['original']
            mask_arrays = dict()
            for name in dataset.keys():
                if name in mask_names:
                    mask_arrays[name] = dataset[name]
            data_shape = np.shape(original_array)

            for i in range(0,data_shape[0],single_size[0]/2):
                for j in range(0,data_shape[1],single_size[1]/2):
                    for k in range(0,data_shape[2],single_size[2]/2):
                        if i+single_size[0]/2<data_shape[0] and j+single_size[1]/2<data_shape[1] and k+single_size[2]/2<data_shape[2]:
                            clipped_mask = dict()
                            flag = False
                            for name in mask_arrays.keys():
                                temp_mask = mask_arrays[name][i:i+single_size[0],j:j+single_size[1],k:k+single_size[2]]
                                temp_shape = np.shape(temp_mask)
                                clipped_mask[name] = np.zeros(single_size,np.uint8)
                                clipped_mask[name][:temp_shape[0],:temp_shape[1],:temp_shape[2]] += temp_mask
                                if np.sum(np.float32(clipped_mask[name])) / (single_size[0] * single_size[1] * single_size[2]) >= (0.05 * (1 - epoch * 1.0 / max_epoch))\
                                        or number in accept_zeros:
                                    flag = True
                            if flag:
                                temp_dicom = original_array[i:i+single_size[0],j:j+single_size[1],k:k+single_size[2]]
                                temp_shape = np.shape(temp_dicom)
                                clipped_dicom = np.zeros(single_size,np.int32)
                                clipped_dicom[:temp_shape[0], :temp_shape[1], :temp_shape[2]] += temp_dicom
                                dicom_datas[number].append(clipped_dicom)
                                clipped_mask["background"] = np.uint8((clipped_mask["airway"]+clipped_mask["artery"])==0)
                                mask_datas[number].append(clipped_mask)

    return dicom_datas,mask_datas,accept_zeros

# def test():
#     meta_path = '/opt/analyse_airway/data_meta.pkl'
#     single_size = [64,64,64]
#     dicom_datas,mask_datas=get_organized_data(meta_path,single_size)
#     print dicom_datas.keys()
#     print mask_datas.keys()

# test()
