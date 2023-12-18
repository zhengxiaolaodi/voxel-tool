import h5py
import numpy as np
import time

def voxel_point_make(path,name_result):
    f = h5py.File(path, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    voxel_num = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
    voxel_sequence = f['voxel_sequence'][:].astype('int64')
    f.close()

    voxel_point_number = np.ones([data.shape[0], data.shape[1]])
    vectorc_0 = np.array([0, 0, 0]).reshape([1, 3])

    for i in range(data.shape[0]):
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))









        for j in range(data.shape[1]):
            m = 0
            for k in range(data.shape[2]):
                zzz = data[i, j, k, :].reshape([1, 3])
                if zzz[0, 0] != 0 and zzz[0, 1] != 0 and zzz[0, 2] != 0:
                    m = m + 1
            voxel_point_number[i, j] = m
    h5f = h5py.File(name_result, 'w')
    h5f.create_dataset('data', data=data)
    h5f.create_dataset('label', data=label)
    h5f.create_dataset('voxel_num', data=voxel_num)
    h5f.create_dataset('voxel_sequence', data=voxel_sequence)
    h5f.create_dataset('voxel_point_number', data=voxel_point_number)
    h5f.close()

path="H:\dataset_h5\\3D_IKEA\\3DIKEA_6cls_voxel_356_seq_train"
name_result='H:\dataset_h5_voxel_pointnumber\\3D_IKEA\\3DIKEA_train_0.2_voxel_number'
voxel_point_make(path,name_result)
path="H:\dataset_h5\\3D_IKEA\\3DIKEA_6cls_voxel_356_seq_test"
name_result='H:\dataset_h5_voxel_pointnumber\\3D_IKEA\\3DIKEA_test_0.2_voxel_number'
voxel_point_make(path,name_result)

path="H:\dataset_h5\sun_rgbd\\sun_9cls_1987_voxel_748_seq_train"
name_result="H:\dataset_h5_voxel_pointnumber\sun_rgbd\\sun_9cls_1987_voxel_748_seq_train_0.2_point_number"
voxel_point_make(path,name_result)
path="H:\dataset_h5\sun_rgbd\\sun_9cls_1878_voxel_748_seq_test"
name_result="H:\dataset_h5_voxel_pointnumber\sun_rgbd\\sun_9cls_1878_voxel_748_seq_test_0.2_point_number"
voxel_point_make(path,name_result)




for i in range(0, 5, 1):
    path=path+'nyu2_sub_22cls_voxel759_dsp220_train_%s.h5'%(i)
    name_result='nyu2_sub_22cls_voxel759_dsp220_train_%s_voxel_point_number_res0_2.h5'%(i)
    voxel_point_make(path, name_result)

path='I:\h5_dataset\\nyu2_sub_22cls_sequence\\nyu2_sub_22cls_voxel759_dsp220_test_0_21_res0_2.h5'
name_result = 'I:\h5_dataset\\nyu2_sub_22cls_sequence\\nyu2_sub_22cls_voxel759_dsp220_test_0_21_voxel_point_number_res0_2.h5'
voxel_point_make(path, name_result)





class Solution {
private:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking(vector<int>& candidates, int target, int sum, int startIndex, vector<bool>& used) {
        if (sum == target) {
            result.push_back(path);
            return;
        }
        for (int i = startIndex; i < candidates.size() && sum + candidates[i] <= target; i++) {
            // used[i - 1] == true，说明同一树枝candidates[i - 1]使用过
            // used[i - 1] == false，说明同一树层candidates[i - 1]使用过
            // 要对同一树层使用过的元素进行跳过
            if (i > 0 && candidates[i] == candidates[i - 1] && used[i - 1] == false) {
                continue;
            }
            sum += candidates[i];
            path.push_back(candidates[i]);
            used[i] = true;
            backtracking(candidates, target, sum, i + 1, used); // 和39.组合总和的区别1，这里是i+1，每个数字在每个组合中只能使用一次
            used[i] = false;
            sum -= candidates[i];
            path.pop_back();
        }
    }

public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<bool> used(candidates.size(), false);
        path.clear();
        result.clear();
        // 首先把给candidates排序，让其相同的元素都挨在一起。
        sort(candidates.begin(), candidates.end());
        backtracking(candidates, target, 0, 0, used);
        return result;
    }
};


