import h5py
f = h5py.File('G:\Desktop\data_remake\\3DIKEA_train_0.8', "r")
for key in f.keys():
    print(f[key].name)  #获得名称，相当于字典中的key
    print(f[key].shape)
f.close()