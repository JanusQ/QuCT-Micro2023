import cloudpickle as pickle
import sys


# sys.path.append("/home/luliqiang/python-cnn-tools-test/QuCT-Micro2023")

from openpyxl import Workbook
from downstream.fidelity_predict.evaluate_tools import plot_top_ratio
from plot.plot import get_duration2circuit_infos
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.parser import get_circuit_duration, get_couple_prop
from circuit.utils import make_circuitlet
# from generate_dataset import gen_train_dataset
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path, plot_correlation, plot_real_predicted_fidelity
import random
import numpy as np
import ray
# from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator, get_random_erroneous_pattern
from utils.backend import devide_chip, gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle
import os
from sklearn.model_selection import train_test_split
from scipy import io as sio

# with open(f"execute_18bit/upstream_model_18_0_2500.pkl","rb")as f:
#     _upstream_model = pickle.load(f)
    
# res_500_2500 = sio.loadmat("execute_18bit/measure_results_20230322_500_2500.mat")
# res_0_500 = sio.loadmat("execute_18bit/measure_results_20230321.mat")
# all_results_load = {**res_0_500, **res_500_2500}

# dataset = _upstream_model.dataset
# assert len(dataset) == 2500



paths = [
    'execute_18bits_algos_more_info_mirror.pkl',  # 没加一层单比特, 每个算法每个10个
    'execute_18bits_algos_more_info.pkl', # 没加一层单比特门
    'execute_18bits_train_20_170_more_info_cycle.pkl', # cycle的barrier好像没加上
    'execute_18bits_train_20_170_more_info.pkl', # 
    'execute_18bits_validate_1500_more_info.pkl', #4220
    'execute_18bits_validate_more_info_3000.pkl', ##5840
    'execute_18bits_validate_various_input_200_more_info.pkl'#8780
]
# dataset  = []
# for path in paths:
#     with open(path,'rb') as f:
#         _dataset = pickle.load(f)
#         print(len(_dataset))
#         dataset += _dataset
# print(len(dataset))
# dataset_size = 4220
# dataset = dataset[:dataset_size]
# all_results_load_1 = sio.loadmat("measure_results_20230412_0_2000.mat")
# all_results_load_2 = sio.loadmat("measure_results_20230412_2000_4000.mat")
# all_results_load = {**all_results_load_1, **all_results_load_2}



    
all_results_load = sio.loadmat("measure_results_20230417_0000_8980_abandon.mat")

size = 6
n_qubits = 18
# topology = gen_grid_topology(size)  # 3x3 9 qubits
# new_topology = defaultdict(list)
# for qubit in topology.keys():
#     if qubit < n_qubits:
#         for ele in topology[qubit]:
#             if ele < n_qubits:
#                 new_topology[qubit].append(ele)
                
# topology =  new_topology      
# # neighbor_info = Backend(n_qubits=n_qubits, topology=None).topology
# neighbor_info = topology
# coupling_map = topology_to_coupling_map(topology)  
# backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
#                   basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)
# with open("execute_18bit_new/step1/upstream_model.pkl","rb") as f:
#     upstream_model = pickle.load(f)

# upstream_model = RandomwalkModel(1, 20, backend=backend, travel_directions=('parallel, former'))
# upstream_model.train(dataset, multi_process=True, process_num = 20, remove_redundancy=False)
# with open("execute_18bit_new/step1/upstream_model.pkl","wb") as f:
#     pickle.dump(upstream_model , f)

from circuit.utils import cut_circuit
@ray.remote
def fun_cut(sub_dataset, start, all_results_load, upstream_model):
    cut_dataset = []
    for cir_idx in range(0, len(sub_dataset)):
        cir = sub_dataset[cir_idx]
        if 'gate2layer' not in cir and 'gates2layer' in cir:
            cir['gate2layer']  = cir['gates2layer']
        cir = upstream_model.vectorize(cir)
        res = cut_circuit(cir)
        for devide_qubit, cut_cir in zip(cir['devide_qubits'], res):
            sub_state = {}
            state_count = dict([(s, cnt) for s, cnt in zip(
                            np.squeeze(all_results_load[f'circuit No.{cir_idx + start} state_index']), 
                            np.squeeze(all_results_load[f'circuit No.{cir_idx + start} measure_counts']))])
            for k, v in state_count.items():
                sub_k = int("".join(np.array(list('{0:018b}'.format(k)))[devide_qubit]),2)
                if sub_k in sub_state:
                    sub_state[sub_k] += v
                else:
                    sub_state[sub_k] = v
            
            cut_cir['state_count'] = sub_state
            if 0 not in sub_state:
                continue
            cut_cir['ground_truth_fidelity'] = sub_state[0]/1000
            cut_dataset.append(cut_cir)
    return cut_dataset

from plot.plot import plot_correlation


def sumarize_datasize(dataset, name):
    data = []
    labels = []
    for circuit_info in dataset:
        data.append([len(circuit_info['gates']),
                     circuit_info['ground_truth_fidelity'], circuit_info['duration'], len(circuit_info['layer2gates']),
        ])
        # labels.append(circuit_info['label'])

    random.shuffle(data)
    data = data[:3000]  # 太大了画出来的图太密了
    data = np.array(data)
    plot_correlation(data, [
                            'n_gates', 'ground_truth_fidelity', 'duration', 'depth',], color_features=None, name=name)

# with open("execute_18bit_new/step1/cut_dataset.pkl","rb") as f:
#     cut_dataset = pickle.load(f)

# sumarize_datasize(cut_dataset, 'train_5bit.svg')
# cut_dataset = []
# # for future in futures:
# #     cut_dataset += ray.get(future)

# with open("execute_18bit_new/step1/cut_dataset.pkl","wb") as f:
#           pickle.dump(cut_dataset , f)
    
# filter_dataset = []
# for cir in cut_dataset:
#     if cir['ground_truth_fidelity'] > 0.4:
#         filter_dataset.append(cir)
    
# index = np.arange(len(cut_dataset))
# random.shuffle(index)
# train_index, test_index = index[:-1500], index[-1500:]


# load_index = False
# if load_index:
#     with open("execute_18bit_new/devide_index.pkl", 'rb') as f:
#         train_index, test_index = pickle.load(f)
# else:
#     with open("execute_18bit_new/devide_index.pkl", 'wb') as f:
#         pickle.dump((train_index, test_index) , f)

# train_dataset, test_dataset = np.array(cut_dataset)[train_index], np.array(cut_dataset)[test_index]

# downstream_model = FidelityModel(upstream_model)
# with open('params_144.pkl', 'rb') as f:
#     params = pickle.load(f)
# downstream_model.error_params = params

# with open(f"execute_18bit_new/step1/downstream_model.pkl","wb")as f:
#     pickle.dump(downstream_model,f)

with open(f"execute_18bit_new/step1/downstream_model.pkl","rb")as f:
    downstream_model = pickle.load(f)
upstream_model = downstream_model.upstream_model


mode = 'validate_5bit'
if mode == 'validate_5bit':
    with open(paths[4],'rb') as f:
        test_dataset_5 = pickle.load(f)
        print(len(test_dataset_5))
        for i, cir in enumerate(test_dataset_5):
            assert 'devide_qubits' in cir and  cir['devide_qubits'] is not None
    # test_dataset = []
    # for cir in test_dataset_5:
    #     test_dataset.append(upstream_model.vectorize(cir))
    futures = []
    for start in range(0, 1620, 100):
        futures.append(fun_cut(test_dataset_5[start:start+100], start + 4220, all_results_load, upstream_model))
        
    test_dataset = []
    for future in futures:
        test_dataset += ray.get(future)
    
elif mode == 'validate_18bit':
    with open(paths[5],'rb') as f:
        test_dataset_18 = pickle.load(f)
        print(len(test_dataset_18))
        for i, cir in enumerate(test_dataset_18):
            assert 'devide_qubits' not in cir or cir['devide_qubits'] is None
    test_dataset = []
    for idx, cir in enumerate(test_dataset_18):
        new_cir = upstream_model.vectorize(cir)
        state_count = dict([(s, cnt) for s, cnt in zip(
                            all_results_load[f'circuit No.{idx + 5840} state_index'].reshape(-1), 
                            all_results_load[f'circuit No.{idx + 5840} measure_counts'].reshape(-1))])
        new_cir['state_count'] = state_count
        if 0 not in state_count:
            continue
        new_cir['ground_truth_fidelity'] = state_count[0] / 1000
        test_dataset.append(new_cir)
        
elif mode == 'validate_18bit_various_input':
    with open(paths[6],'rb') as f:
        test_dataset_18_various_input = pickle.load(f)
        print(len(test_dataset_18_various_input))
        for i, cir in enumerate(test_dataset_18_various_input):
            assert 'devide_qubits' not in cir or cir['devide_qubits'] is None
    test_dataset = []
    for idx, cir in enumerate(test_dataset_18_various_input):
        new_cir = upstream_model.vectorize(cir)
        state_count = dict([(s, cnt) for s, cnt in zip(
                            all_results_load[f'circuit No.{idx + 8780} state_index'].reshape(-1), 
                            all_results_load[f'circuit No.{idx + 8780} measure_counts'].reshape(-1))])
        new_cir['state_count'] = state_count
        if 0 not in state_count:
            continue
        new_cir['ground_truth_fidelity'] = state_count[0] / 1000
        test_dataset.append(new_cir)  

predicts, reals, props = [], [], []
for idx, cir in enumerate(test_dataset):
    if idx % 100 == 0:
        print(idx, "predict finished!")
    predict = downstream_model.predict_fidelity(cir)

    predicts.append(cir['circuit_predict'])
    reals.append(cir['ground_truth_fidelity'])
    props.append(cir['duration'])
    # print(predict, cir['ground_truth_fidelity'])
    
predicts = np.array(predicts)
reals = np.array(reals)
props = np.array(props)
with open(f"execute_18bit_new/step1/error_params_predicts_{mode}.pkl","wb")as f:
    pickle.dump((downstream_model, predicts, reals, props, test_dataset), f)
    
# find_error_path(upstream_model, downstream_model.error_params)

# with open(f"execute_18bit/error_params_predicts_execute_18bits_train_0_2500.pkl", "rb")as f:
#     downstream_model, predicts, reals, durations, test_dataset = pickle.load(f)
# upstream_model = downstream_model.upstream_model


print('average inaccuracy = ', np.abs(predicts - reals).mean())
print('average inaccuracy = ', np.abs(predicts - reals).std())
# fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
# duration_X, duration2circuit_index = plot_duration_fidelity(fig, axes, test_dataset)
# fig.savefig(f"duration_fidelity_execute_18bits_train_0_2500_step2.svg")  # step


# fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
# duration_X, duration2circuit_index = plot_duration_fidelity(
#     fig, axes, test_dataset)
# fig.savefig(f"duration_fidelity_{n_qubits}__step2.svg")  # step
# plt.close(fig)

# 画x: real fidelity, y: predicted fidelity
fig, axes = plt.subplots(figsize=(10, 10))  # 创建一个图形对象和一个子图对象
axes.axis([0, 1, 0, 1])
axes.scatter(reals, predicts)
# axes.set_xlim(.4, 1)
# axes.set_ylim(.4, 1)
axes.set_xlabel('real ')
axes.set_ylabel('predict')
axes.plot([[0,0],[1,1]])
fig.savefig(f"execute_18bit_new/step1/real_predictedy_{mode}.svg")  # step

# # 画x: real fidelity, y: inaccuracy
# fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
# duration_X, duration2circuit_index = get_duration2circuit_infos(
#     durations, 100, 0)

# delta = []
# for circuit_index in duration2circuit_index:
#     delta.append(np.abs(reals[circuit_index] - predicts[circuit_index]).mean())

# axes.plot(duration_X, delta, markersize=12, linewidth=2, label='delta', marker='^')
# axes.set_xlabel('duration')
# axes.set_ylabel('fidelity')
# axes.legend()  # 添加图例
# fig.savefig(f"inaccuracy_fidelity_{n_qubits}__step2.svg")
# print(np.array(delta).mean())


# 存path error 到 excel
# error_params = downstream_model.error_params['gate_params']
# device_index2device = {}
# for device  in upstream_model.device2path_table.keys():
#     device_index = list(upstream_model.device2path_table.keys()).index(device)
#     device_index2device[device_index] = device

# error_params_path_weight = {}
# error_params_path = {}
# for idx, device_error_param in enumerate(error_params):
#     device = device_index2device[idx]
#     sort = np.argsort(device_error_param)
#     sort = sort[::-1]
#     device_error_params_path_weight = []
#     device_error_params_path = []
#     for i in sort:
#         if int(i) in upstream_model.device2reverse_path_table[device].keys():
#             path = upstream_model.device2reverse_path_table[device][int(i)]
#             if isinstance(path, str):
#                 device_error_params_path_weight.append(
#                     (path, float(device_error_param[i]), downstream_model.path_count[path]))
#                 device_error_params_path.append(path)
#     error_params_path_weight[device] = device_error_params_path_weight
#     error_params_path[device] = device_error_params_path

# wb = Workbook()

# for device, device_error_params_path_weight in error_params_path_weight.items():
#     ws = wb.create_sheet(str(device))
#     for row in device_error_params_path_weight:
#         row = list(row)
#         # if row[0] in upstream_model.erroneous_pattern[device]:
#         #     row.append("true")
#         # else:
#         #     row.append("false")
#         ws.append(row)

# # Save the file
# wb.save("sample_18qubits_step2.xlsx")

# with open(f"execute_18bit/split_dataset_execute_18bits_train_0_2500.pkl","rb")as f:
#     train_dataset, test_dataset = pickle.load(f)
    
# def sumarize_datasize(dataset, name):
#     data = []
#     labels = []
#     for circuit_info in dataset:
#         data.append([len(circuit_info['gates']), circuit_info['two_qubit_prob'],
#                      circuit_info['ground_truth_fidelity'],
#         ])
#         # labels.append(circuit_info['label'])

#     random.shuffle(data)
#     data = data[:3000]  # 太大了画出来的图太密了
#     data = np.array(data)
#     plot_correlation(data, [
#                             'n_gates',  'two_qubit_prob', 'ground_truth_fidelity', ], color_features=None, name=name)

# for cir in train_dataset:
#     cir['two_qubit_prob'] = get_couple_prop(cir)
# for cir in test_dataset:
#     cir['two_qubit_prob'] = get_couple_prop(cir)
# sumarize_datasize(train_dataset, f'train_dataset_{n_qubits}')
# sumarize_datasize(test_dataset, f'test_dataset_{n_qubits}')