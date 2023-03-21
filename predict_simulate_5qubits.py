from openpyxl import Workbook
from downstream.fidelity_predict.evaluate_tools import plot_top_ratio
from plot.plot import get_duration2circuit_infos
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.parser import get_circuit_duration
from circuit.utils import make_circuitlet
from generate_dataset import gen_train_dataset
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path, plot_correlation, plot_real_predicted_fidelity
import random
import numpy as np

from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator, get_random_erroneous_pattern
from utils.backend import devide_chip, gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle
import os
from sklearn.model_selection import train_test_split

size = 3
n_qubits = 5
n_steps = 1

topology = gen_grid_topology(size)  # 3x3 9 qubits
new_topology = defaultdict(list)
for qubit in topology.keys():
    if qubit < n_qubits:
        for ele in topology[qubit]:
            if ele < n_qubits:
                new_topology[qubit].append(ele)
topology = new_topology
neighbor_info = copy.deepcopy(topology)
coupling_map = topology_to_coupling_map(topology)

backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                  basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

dir_size = 'temp_data'
dataset_path = os.path.join(dir_size, f"dataset_{n_qubits}.pkl")
upstream_model_path = os.path.join(dir_size, f"upstream_model_{n_qubits}.pkl")
regen = True
if regen:
    # dataset = gen_train_dataset(
    #     n_qubits, topology, neighbor_info, coupling_map, 2000)
    dataset = gen_random_circuits(min_gate=20, max_gate=150, n_circuits=25, two_qubit_gate_probs=[
        3, 7], gate_num_step=20, backend=backend, multi_process=True)

    test_dataset = gen_random_circuits(min_gate=20, max_gate=3000, n_circuits=1, two_qubit_gate_probs=[
        3, 7], gate_num_step=60, backend=backend, multi_process=True)

    print('train dataset size = ', len(dataset))
    print('test dataset size = ', len(test_dataset))

    upstream_model = RandomwalkModel(
        n_steps, 20, backend=backend, travel_directions=('parallel', 'former'))
    
    upstream_model.train(dataset + test_dataset, multi_process=True,
                         remove_redundancy=n_steps > 1)

    print("original", len(dataset))
    # dataset = make_circuitlet(dataset)
    print("cutted", len(dataset))
    

    simulator = NoiseSimulator(backend)
    erroneous_pattern = get_random_erroneous_pattern(
        upstream_model, error_pattern_num_per_device=2)
    
    # 每个subcircuit是单独的NoiseSimulator，backend对应不对
    erroneous_pattern = simulator.get_error_results(
        dataset + test_dataset, upstream_model, multi_process=True, erroneous_pattern=erroneous_pattern)
    upstream_model.erroneous_pattern = erroneous_pattern

    with open(upstream_model_path, "wb")as f:
        pickle.dump(upstream_model, f)

    train_dataset, validation_dataset = train_test_split(dataset, test_size = 500)
    
    test_dataset = np.array(test_dataset)

    with open(dataset_path, "wb")as f:
        pickle.dump((train_dataset, validation_dataset, test_dataset), f)

else:
    with open(dataset_path, "rb")as f:
        train_dataset, validation_dataset, test_dataset = pickle.load(f)

    with open(upstream_model_path, "rb")as f:
        upstream_model = pickle.load(f)

total_dataset = np.concatenate([train_dataset, validation_dataset, test_dataset])
error_data = []
for circuit_info in total_dataset:
    error_data.append([circuit_info['n_erroneous_patterns'], len(circuit_info['gates']),
                      circuit_info['n_erroneous_patterns'] / len(circuit_info['gates']), circuit_info['two_qubit_prob'],
                      circuit_info['ground_truth_fidelity'], circuit_info['independent_fidelity'], circuit_info['ground_truth_fidelity'] - circuit_info['independent_fidelity']
                    ])

error_data = np.array(error_data)
plot_correlation(error_data, ['n_erroneous_patterns',
                 'n_gates', 'error_prop', 'two_qubit_prob', 'ground_truth_fidelity', 'independent_fidelity', 'ground_truth_fidelity - independent_fidelity'], None)

print('erroneous patterns = ', upstream_model.erroneous_pattern)

retrain = True
# # TODO: 要用fidelity 大于 0.5的阶段
if retrain:
    downstream_model = FidelityModel(upstream_model)
    downstream_model.train(
        train_dataset, validation_dataset, epoch_num=200)
    
    predicts, reals, durations = [], [], []
    for idx, cir in enumerate(test_dataset):
        cir = upstream_model.vectorize(cir)
        if idx % 100 == 0:
            print(idx, "predict finished!")
        predict = downstream_model.predict_fidelity(cir)

        predicts.append(cir['circuit_predict'])
        reals.append(cir['ground_truth_fidelity'])
        durations.append(cir['duration'])
        # print(predict, cir['ground_truth_fidelity'])

    reals = np.array(reals)
    predicts = np.array(predicts)
    durations = np.array(durations)

    #     # print(predict)
    with open(f"error_params_predicts_{n_qubits}.pkl", "wb")as f:
        pickle.dump((downstream_model, predicts, reals, durations, test_dataset), f)
else:
    with open(f"error_params_predicts_{n_qubits}.pkl", "rb")as f:
        downstream_model, predicts, reals, durations, test_dataset = pickle.load(f)
    upstream_model = downstream_model.upstream_model
    
print('average inaccuracy = ', np.abs(predicts - reals).mean())

# 画找到path的数量
find_error_path(upstream_model, downstream_model.error_params)

fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = plot_duration_fidelity(
    fig, axes, test_dataset)
fig.savefig(f"duration_fidelity_{n_qubits}_step1.svg")  # step
plt.close(fig)

# 画x: real fidelity, y: predicted fidelity
fig, axes = plt.subplots(figsize=(10, 10))  # 创建一个图形对象和一个子图对象
plot_real_predicted_fidelity(fig, axes, test_dataset)
fig.savefig(f"real_predictedy_{n_qubits}_step1.svg")  # step

# 画x: real fidelity, y: inaccuracy
fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = get_duration2circuit_infos(
    durations, 100, 0)

delta = []
for circuit_index in duration2circuit_index:
    delta.append(np.abs(reals[circuit_index] - predicts[circuit_index]).mean())

axes.plot(duration_X, delta, markersize=12, linewidth=2, label='delta', marker='^')
axes.set_xlabel('duration')
axes.set_ylabel('fidelity')
axes.legend()  # 添加图例
fig.savefig(f"inaccuracy_fidelity_{n_qubits}_step1.svg")
print(np.array(delta).mean())


# 存path error 到 excel
error_params = downstream_model.error_params
device_index2device = {}
for device  in upstream_model.device2path_table.keys():
    device_index = list(upstream_model.device2path_table.keys()).index(device)
    device_index2device[device_index] = device

error_params_path_weight = {}
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
            if isinstance(path, str):
                device_error_params_path_weight.append(
                    (path, float(device_error_param[i]), downstream_model.path_count[path]))
                device_error_params_path.append(path)
    error_params_path_weight[device] = device_error_params_path_weight
    error_params_path[device] = device_error_params_path

wb = Workbook()

for device, device_error_params_path_weight in error_params_path_weight.items():
    ws = wb.create_sheet(str(device))
    for row in device_error_params_path_weight:
        row = list(row)
        if row[0] in upstream_model.erroneous_pattern[device]:
            row.append("true")
        else:
            row.append("false")
        ws.append(row)

# Save the file
wb.save("sample_5qubits.xlsx")
