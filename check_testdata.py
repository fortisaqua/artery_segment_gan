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

input_shape = [64,64,128]
test_dir = './FU_LI_JUN/'

origin_dir = read_dicoms(test_dir + "original1")
test_data = tools.Test_data(origin_dir, input_shape, 'vtk_data')
test_data.output_origin()
print "end"