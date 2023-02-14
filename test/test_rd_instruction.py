# 测试下MDS降维应用在门上面

from upstream.randomwalk_model import RandomwalkModel

from upstream.dimensionality_reduction import MDS
import numpy as np

import pickle

path = 'rwm_5qubit.pkl'
model = RandomwalkModel.load(path)

vecs = []
for circuit_info in model.dataset:
    for sparse_vec in circuit_info['instruction2sparse_vecs']:
        # print(sparse_vec)
        vec = np.zeros((len(model.hash_table), 1))
        for index1 in sparse_vec:
            # print(index1)
            vec[index1][0] = 1
        vecs.append(vec)
vecs = np.array(vecs)

print(vecs.shape)

params, reduced_vecs = MDS(vecs, 100, 10, print_interval=1)
print(params)

path = 'pattern_extractor/model/mds_q5.pkl'
file = open(path, 'wb')
pickle.dump(params, file)
file.close()
