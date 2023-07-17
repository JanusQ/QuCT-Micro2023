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


def add_idle(circuit_info):
    idle_id = len(circuit_info['gates'])
    for layer_id, layer_gates in enumerate(circuit_info['layer2gates']):
        is_idle = np.zeros(circuit_info['num_qubits'])
        for gate in layer_gates:
            is_idle[np.array(gate['qubits'])] = 1
        is_idle = np.where(is_idle == 0)[0]
        for idle_qubit in is_idle:
            idle_gate = {}
            idle_gate['id'] = idle_id
            idle_gate['name'] = 'idle'
            idle_gate['qubits'] = [idle_qubit]
            idle_gate['params'] = []
            
            circuit_info['gates'].append(idle_gate)
            circuit_info['gate2layer'].append(layer_id)
            circuit_info['layer2gates'][layer_id].append(idle_gate)
            idle_id += 1
    


paths = [
    'execute_18bits_algos_more_info_mirror.pkl',  # 没加一层单比特, 每个算法每个10个
    'execute_18bits_algos_more_info.pkl', # 没加一层单比特门
    'execute_18bits_train_20_170_more_info_cycle.pkl', # cycle的barrier好像没加上
    'execute_18bits_train_20_170_more_info.pkl', # 
    # 'execute_18bits_validate_1500_more_info.pkl',
    # 'execute_18bits_validate_more_info_3000.pkl',
    # 'execute_18bits_validate_various_input_200_more_info.pkl'
]
dataset  = []
for path in paths:
    with open(path,'rb') as f:
        _dataset = pickle.load(f)
        print(len(_dataset))
        dataset += _dataset
print(len(dataset))
dataset_size = 4000
dataset = dataset[:dataset_size]
all_results_load_1 = sio.loadmat("measure_results_20230412_0_2000.mat")
all_results_load_2 = sio.loadmat("measure_results_20230412_2000_4000.mat")
all_results_load = {**all_results_load_1, **all_results_load_2}

size = 6
n_qubits = 18
topology = gen_grid_topology(size)  # 3x3 9 qubits
new_topology = defaultdict(list)
for qubit in topology.keys():
    if qubit < n_qubits:
        for ele in topology[qubit]:
            if ele < n_qubits:
                new_topology[qubit].append(ele)
                
topology =  new_topology      
# neighbor_info = Backend(n_qubits=n_qubits, topology=None).topology
neighbor_info = copy.deepcopy(topology)
coupling_map = topology_to_coupling_map(topology)  
backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                  basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

upstream_model = RandomwalkModel(2, 20, backend=backend, travel_directions=('parallel, former'))

# with open('execute_18bit/step2_finetune/filter_path.pkl',  'rb') as f:
#     filter_path = pickle.load(f)
# for circuit_info in dataset:
#     add_idle(circuit_info)

upstream_model.train(dataset, multi_process=True, process_num = 20, remove_redundancy=False, is_filter_path= False)
with open("execute_18bit/step2_finetune/upstream_model.pkl","wb") as f:
    pickle.dump(upstream_model , f)
from circuit.utils import cut_circuit

@ray.remote
def fun_cut(sub_dataset, start, all_results_load):
    cut_dataset = []
    for cir_idx in range(0, len(sub_dataset)):
        cir = sub_dataset[cir_idx]
        if 'gate2layer' not in cir and 'gates2layer' in cir:
            cir['gate2layer']  = cir['gates2layer']
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
            cut_cir['ground_truth_fidelity'] = sub_state[0]/1000
            cut_dataset.append(cut_cir)
    return cut_dataset

futures = []
for start in range(20,4000,100):
    futures.append(fun_cut.remote(dataset[start:start+100], start, all_results_load))
cut_dataset = []
for future in futures:
    cut_dataset += ray.get(future)

with open("execute_18bit/step2_finetune/cut_dataset.pkl","wb") as f:
          pickle.dump(cut_dataset , f)


with open("execute_18bit/devide_index.pkl", 'rb') as f:
    train_index, test_index = pickle.load(f)
    
train_dataset, test_dataset = np.array(cut_dataset)[train_index], np.array(cut_dataset)[test_index]
train_dataset = list(train_dataset)
test_dataset = list(test_dataset)
downstream_model = FidelityModel(upstream_model)
downstream_model.train(train_dataset, epoch_num=200)



with open(f"execute_18bit/step2_finetune/downstream_model.pkl","wb")as f:
    pickle.dump(downstream_model,f)


    

predicts, reals, props = [], [], []
for idx, cir in enumerate(test_dataset):
    if idx % 100 == 0:
        print(idx, "predict finished!")
    # cir  = upstream_model.vectorize(cir)
    predict = downstream_model.predict_fidelity(cir)

    predicts.append(cir['circuit_predict'])
    reals.append(cir['ground_truth_fidelity'])
    props.append(cir['duration'])
    # print(predict, cir['ground_truth_fidelity'])
    
predicts = np.array(predicts)
reals = np.array(reals)
props = np.array(props)
with open(f"execute_18bit/step2_finetune/error_params_predicts_execute_18bits_train_0_2500_step2_finetune.pkl","wb")as f:
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
axes.set_xlim(.4, 1)
axes.set_ylim(.4, 1)
axes.set_xlabel('real ')
axes.set_ylabel('predict')
axes.plot([[0,0],[1,1]])
fig.savefig(f"execute_18bit/step2_finetune/real_predictedy__step2_finetune_0.4.svg")  # step

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