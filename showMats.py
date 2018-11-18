import matplotlib.pyplot as plt
import PIL.Image as Image
import scipy.io as sio
import numpy as np

path = "./material.mat"
datas = sio.loadmat(path)
tag = 1
plt.figure("block result")
plt.axis('equal')
for key, val in datas.items():
    try:
        up = np.max(val)
        down = np.min(val)
        toImageArray = (val - down + 0.00001) / (up - down + 0.00001) * 255
        toShowImage = Image.fromarray(toImageArray.astype('uint8')).convert('L')
        plt.subplot(1,3,tag)
        plt.imshow(toShowImage)
        tag += 1
    except Exception,e:
        pass
plt.show()