import SimpleITK as ST
import dicom_read
import numpy as np

img_dir = './WANG_REN/airway'
img = dicom_read.read_dicoms(img_dir)
airway_array = ST.GetArrayFromImage(img)
airway_array_1 = np.transpose(airway_array,[2,1,0])
airway_array = np.transpose(airway_array_1,[2,1,0])
airway_img = ST.GetImageFromArray(airway_array)
ST.WriteImage(airway_img,'./WANG_REN/airway.vtk')
