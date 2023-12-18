import h5py
import numpy as np
res=np.array([0.2,0.4,0.8])
data_result_shape=np.ones([1,6])


mode=str(res[0])
f = h5py.File('G:\Desktop\data_remake\\3DIKEA_train_%s'%(mode), 'r')
data = f['data'][:].astype('float32')
label = f['label'][:].astype('int64')
voxel_num = f['voxel_num'][:].astype('int64')
voxel_sequence = f['voxel_sequence'][:].astype('int64')
voxel_point_number = f['voxel_point_number'][:].astype('int64')


f.close()
data_1=data.reshape([data.shape[0],data.shape[3],data.shape[1]*data.shape[2]])
voxel_num_1=voxel_num
voxel_sequence_1=voxel_sequence
voxel_point_1=voxel_point_number

mode=str(res[1])
f = h5py.File('G:\Desktop\data_remake\\3DIKEA_train_%s'%(mode), 'r')
data = f['data'][:].astype('float32')
label = f['label'][:].astype('int64')
voxel_num = f['voxel_num'][:].astype('int64')
voxel_sequence = f['voxel_sequence'][:].astype('int64')
voxel_point_number = f['voxel_point_number'][:].astype('int64')
f.close()
data_2=data.reshape([data.shape[0],data.shape[3],data.shape[1]*data.shape[2]])
voxel_num_2=voxel_num
voxel_sequence_2=voxel_sequence
voxel_point_2=voxel_point_number


mode=str(res[2])
f = h5py.File('G:\Desktop\data_remake\\3DIKEA_train_%s'%(mode), 'r')
data = f['data'][:].astype('float32')
label = f['label'][:].astype('int64')
voxel_num = f['voxel_num'][:].astype('int64')
voxel_sequence = f['voxel_sequence'][:].astype('int64')
voxel_point_number = f['voxel_point_number'][:].astype('int64')
f.close()
data_3=data.reshape([data.shape[0],data.shape[3],data.shape[1]*data.shape[2]])
voxel_num_3=voxel_num
voxel_sequence_3=voxel_sequence
voxel_point_3=voxel_point_number



data_result=np.concatenate((data_1,data_2,data_3),axis=-1)
voxel_num_result=np.concatenate((voxel_num_1,voxel_num_2,voxel_num_3),axis=-1)
voxel_sequence_result=np.concatenate((voxel_sequence_1,voxel_sequence_2,voxel_sequence_3),axis=-1)
voxel_point_result=np.concatenate((voxel_point_1,voxel_point_2,voxel_point_3),axis=-1)