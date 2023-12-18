import h5py
import numpy as np
import time

f = h5py.File(path, 'r')
data_train = f['data'][:].astype('float32')
label_train = f['label'][:].astype('int64')
voxel_num_train = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
voxel_sequence_train = f['voxel_sequence'][:].astype('int64')
f.close()

f = h5py.File(path, 'r')
data_test = f['data'][:].astype('float32')
label_test = f['label'][:].astype('int64')
voxel_num_test = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
voxel_sequence_test = f['voxel_sequence'][:].astype('int64')
f.close()

data=np.concatenate((data_train,data_test),axis=0)
label=np.concatenate((label_train,label_test),axis=0)
voxel_num=np.concatenate((voxel_num_train,voxel_num_test),axis=0)
voxel_sequence=np.concatenate((voxel_sequence_train,voxel_sequence_test),axis=0)

f = h5py.File(name, "w")
f.create_dataset("data", data=data)
f.create_dataset("label", data=label)
f.create_dataset("voxel_num", data=voxel_num)
f.create_dataset("voxel_sequence", data=voxel_sequence)
f.close()