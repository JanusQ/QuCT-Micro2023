from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.parser import get_circuit_duration
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path
import random
import numpy as np

from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator
from utils.backend import devide_chip, gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle


n_qubits = 18

with open(f"upstream_model_18_0_2500.pkl","rb")as f:
    upstream_model = pickle.load(f)

# with open(f"execute_18bits_train_0_2500.pkl","rb")as f:
#     dataset = pickle.load(f)
    
# filter_dataset = []
# for cir in dataset:
#     cir['ground_truth_fidelity']  = cir['grount_truth_fidelity']
#     if cir['ground_truth_fidelity'] > 0.4:
#         filter_dataset.append(cir)
    
# index = np.arange(len(filter_dataset))
# random.shuffle(index)
# train_index, test_index = index[:-1500], index[-1500:]
# train_dataset, test_dataset = np.array(filter_dataset)[train_index], np.array(filter_dataset)[test_index]
# with open(f"split_dataset_execute_18bits_train_0_2500.pkl","wb")as f:
#     pickle.dump((train_dataset, test_dataset),f)
    
    
with open(f"split_dataset_execute_18bits_train_0_2500.pkl","rb")as f:
    train_dataset, test_dataset = pickle.load(f)

downstream_model = FidelityModel(upstream_model)
downstream_model.train(train_dataset, epoch_num = 200)

predicts, reals, durations = [], [], []
for idx, cir in enumerate(test_dataset):
    # cir = upstream_model.vectorize(cir)
    if idx % 100 == 0:
        print(idx, "predict finished!")
    predict = downstream_model.predict_fidelity(cir)

    predicts.append(cir['circuit_predict'])
    reals.append(cir['ground_truth_fidelity'])
    durations.append(cir['duration'])
    # print(predict, cir['ground_truth_fidelity'])
    
    
    # print(predict)
with open(f"error_params_predicts_execute_18bits_train_0_2500_bias.pkl","wb")as f:
    pickle.dump((downstream_model, predicts, reals, durations), f)
    
# find_error_path(upstream_model, downstream_model.error_params)


fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = plot_duration_fidelity(fig, axes, test_dataset)
fig.savefig(f"duration_fidelity_execute_18bits_train_0_2500_bias.svg")  # step
