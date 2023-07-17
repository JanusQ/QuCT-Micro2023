import copy
import numpy as np
from simulator import NoiseSimulator
import os
import pickle
from utils.backend import default_basis_single_gates, default_basis_two_gates
from utils.backend import devide_chip, gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map

def get_opt_error_path(upstream_model, error_params, name = None):
    error_params = np.array(error_params)
    erroneous_pattern = upstream_model.erroneous_pattern
    
    device_index2device = {} #两比特门与但单比特门映射为一维下标
    for device  in upstream_model.device2path_table.keys():
        device_index = list(upstream_model.device2path_table.keys()).index(device)
        device_index2device[device_index] = device
        
    error_params_path_weight = {} #训练好的参数对应的path及其权重
    error_params_path = {}
    for idx, device_error_param in enumerate(error_params):
        device = device_index2device[idx]
        sort = np.argsort(device_error_param)
        sort = sort[::-1]
        device_error_params_path_weight = []
        device_error_params_path = []
        for i in sort:
            if int(i) in upstream_model.device2reverse_path_table[device].keys():
                path = upstream_model.device2reverse_path_table[device][int(i)]
                if isinstance(path,str):
                    device_error_params_path_weight.append((path,device_error_param[i]))
                    device_error_params_path.append(path)
        error_params_path_weight[device] = device_error_params_path_weight
        error_params_path[device] = device_error_params_path
        
    erroneous_pattern_weight = {} #手动添加的error_path在训练完参数中的排位
    for device, patterns in erroneous_pattern.items():
        device_error_params_path = error_params_path[device]
        device_erroneous_pattern_weight = []
        for pattern in patterns:
            if pattern in device_error_params_path:
                k = device_error_params_path.index(pattern)
                device_erroneous_pattern_weight.append((pattern,k))
        erroneous_pattern_weight[device] = device_erroneous_pattern_weight
    
    new_erroneous_pattern = copy.deepcopy(erroneous_pattern)   
    top = 0.1
    for device, pattern_weights in erroneous_pattern_weight.items():
        path_table_size = len(upstream_model.device2path_table[device].keys())
        for pattern_weight in pattern_weights:
            if  pattern_weight[1] < top * path_table_size:
                if pattern_weight[0] in new_erroneous_pattern[device]:
                    new_erroneous_pattern[device].remove(pattern_weight[0])
                print(device, pattern_weight[0])
        
    return new_erroneous_pattern
size = 8
n_qubits = 50
n_steps = 1

# topology = gen_grid_topology(size)  # 3x3 9 qubits
# new_topology = defaultdict(list)
# for qubit in topology.keys():
#     if qubit < n_qubits:
#         for ele in topology[qubit]:
#             if ele < n_qubits:
#                 new_topology[qubit].append(ele)
# topology = new_topology
# # topology = gen_linear_topology(n_qubits)
# coupling_map = topology_to_coupling_map(topology)
# neighbor_info = copy.deepcopy(topology)




# # all_to_all_backend = Backend(n_qubits=n_qubits, topology=None, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
# #                 basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

# backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
#                   basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

dir_size = f'simulate_50_350/{n_qubits}'
dataset_path = os.path.join(dir_size, "dataset.pkl")
error_params_predicts_path = os.path.join(dir_size, f"error_params_predicts_{n_qubits}.pkl")

with open(dataset_path, "rb")as f:
    train_dataset, validation_dataset, test_dataset = pickle.load(f)

with open(error_params_predicts_path, "rb")as f:
    downstream_model, predicts, reals, durations, test_dataset = pickle.load(f)
upstream_model = downstream_model.upstream_model
total_dataset = np.concatenate([train_dataset, validation_dataset, test_dataset])
backend = upstream_model.backend

reals_1, durations = [],[]
for cir in total_dataset:
    reals_1.append(cir['ground_truth_fidelity'])
    durations.append(cir['duration'])
reals_1 = np.array(reals_1)

erroneous_pattern = get_opt_error_path(upstream_model, downstream_model.error_params['gate_params'])


all_to_all_backend = Backend(n_qubits=5, topology=None, neighbor_info=None, basis_single_gates=default_basis_single_gates,
                basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)    
simulator = NoiseSimulator(all_to_all_backend)

erroneous_pattern = simulator.get_error_results(total_dataset,
                          upstream_model, multi_process=True, erroneous_pattern=erroneous_pattern)
upstream_model.erroneous_pattern = erroneous_pattern

reals_2,durations = [],[]
for cir in total_dataset:
    reals_2.append(cir['ground_truth_fidelity'])
    durations.append(cir['duration'])
reals_2 = np.array(reals_2)

from plot.plot import get_duration2circuit_infos
import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(16, 10)) 

durations = np.array(durations)    

duration_X, duration2circuit_index = get_duration2circuit_infos(durations,200,4000)

real_y1,real_y2 = [],[]
for circuit_index in duration2circuit_index:
    real_y1.append(reals_1[circuit_index].mean())
    real_y2.append(reals_2[circuit_index].mean())


axes.plot(duration_X, real_y1 ,markersize = 12,linewidth = 2, label='real',marker = '^' )
axes.plot(duration_X, real_y2 ,markersize = 12,linewidth = 2, label='real_opt',marker = '^' )
axes.set_xlabel('duration ')
axes.set_ylabel('fidelity')
axes.legend() # 添加图例
fig.show()

fig_path = os.path.join(dir_size, "remore_error_opt.svg")
fig.savefig(fig_path)

target_dataset_path = os.path.join(dir_size, "dataset_remove_error_pattern.pkl")
target_upstream_model__path = os.path.join(dir_size, f"upstream_model_remove_error_pattern.pkl")
with open(target_upstream_model__path, "wb")as f:
    pickle.dump(upstream_model, f)

with open(target_dataset_path, "wb")as f:
    pickle.dump((train_dataset, validation_dataset, test_dataset), f)

# error_data = []
# for circuit_info in total_dataset:
#     error_data.append([circuit_info['n_erroneous_patterns'], len(circuit_info['layer2gates']),len(circuit_info['gates']),
#                       circuit_info['n_erroneous_patterns'] / len(circuit_info['gates']), circuit_info['two_qubit_prob'],
#                       circuit_info['ground_truth_fidelity'], circuit_info['independent_fidelity'], circuit_info['ground_truth_fidelity'] - circuit_info['independent_fidelity']
#                     ])

# error_data = np.array(error_data)
# plot_correlation(error_data, ['n_erroneous_patterns', 'depth ',
#                  'n_gates', 'error_prop', 'two_qubit_prob', 'ground_truth_fidelity', 'independent_fidelity', 'ground_truth_fidelity - independent_fidelity'], name  = 'opt_5_test.png')

# print('erroneous patterns = ', upstream_model.erroneous_pattern)

# retrain = True
# # # TODO: 要用fidelity 大于 0.5的阶段
# if retrain:
#     downstream_model = FidelityModel(upstream_model)
#     downstream_model.train(
#         train_dataset, validation_dataset, epoch_num=200)
    
#     predicts, reals, props = [], [], []
#     for idx, cir in enumerate(test_dataset):
#         cir = upstream_model.vectorize(cir)
#         if idx % 100 == 0:
#             print(idx, "predict finished!")
#         predict = downstream_model.predict_fidelity(cir)

#         predicts.append(predict)
#         reals.append(cir['ground_truth_fidelity'])
#         props.append(cir['duration'])

#     reals = np.array(reals)
#     predicts = np.array(predicts)
#     props = np.array(props)

#     #     # print(predict)
#     with open(f"{dir_size}/error_params_predicts_{n_qubits}.pkl", "wb")as f:
#         pickle.dump((downstream_model, predicts, reals, props, test_dataset), f)
# else:
#     with open(f"{dir_size}/error_params_predicts_{n_qubits}.pkl", "rb")as f:
#         downstream_model, predicts, reals, props, test_dataset = pickle.load(f)
#     upstream_model = downstream_model.upstream_model
    
# print('average inaccuracy = ', np.abs(predicts - reals).mean())

# # 画找到path的数量
# find_error_path(upstream_model, downstream_model.error_params['gate_params'])

# fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
# duration_X, duration2circuit_index = plot_duration_fidelity(
#     fig, axes, test_dataset)
# fig.savefig(f"{dir_size}/duration_fidelity_{n_qubits}_step1.svg")  # step
# plt.close(fig)

# # 画x: real fidelity, y: predicted fidelity
# fig, axes = plt.subplots(figsize=(10, 10))  # 创建一个图形对象和一个子图对象
# plot_real_predicted_fidelity(fig, axes, test_dataset)
# fig.savefig(f"{dir_size}/real_predictedy_{n_qubits}_step1.svg")  # step
 
# # 画x: real fidelity, y: inaccuracy
# fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
# duration_X, duration2circuit_index = get_duration2circuit_infos(
#     props, 100, 0)

# delta = []
# for circuit_index in duration2circuit_index:
#     delta.append(np.abs(reals[circuit_index] - predicts[circuit_index]).mean())

# axes.plot(duration_X, delta, markersize=12, linewidth=2, label='delta', marker='^')
# axes.set_xlabel('duration')
# axes.set_ylabel('fidelity')
# axes.legend()  # 添加图例
# fig.savefig(f"{dir_size}/inaccuracy_fidelity_{n_qubits}_step1.svg")
# print(np.array(delta).mean())


# # 存path error 到 excel
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
#         if row[0] in upstream_model.erroneous_pattern[device]:
#             row.append("true")
#         else:
#             row.append("false")
#         ws.append(row)

# # Save the file
# wb.save(f"{dir_size}/sample_5qubits.xlsx")
