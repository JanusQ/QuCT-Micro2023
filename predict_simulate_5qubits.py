from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.parser import get_circuit_duration
from circuit.utils import make_circuitlet
from generate_dataset import gen_train_dataset
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path
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
regen = False
if regen:
    dataset = gen_train_dataset(
        n_qubits, topology, neighbor_info, coupling_map, 4000)

    upstream_model = RandomwalkModel(
        n_steps, 20, backend=backend, travel_directions=('parallel', 'former'))
    print(len(dataset), "circuit generated")
    upstream_model.train(dataset, multi_process = True, remove_redundancy = n_steps > 1)

    with open(upstream_model_path, "wb")as f:
        pickle.dump(upstream_model, f)

    print("original", len(dataset))
    dataset = make_circuitlet(dataset)
    print("cutted", len(dataset))

    simulator = NoiseSimulator(backend)
    erroneous_pattern = get_random_erroneous_pattern(upstream_model, error_pattern_num_per_device = 3)
    erroneous_pattern = simulator.get_error_results(dataset, upstream_model, multi_process=True)
    upstream_model.erroneous_pattern = erroneous_pattern

    with open(upstream_model_path, "wb")as f:
        pickle.dump(upstream_model, f)

    index = np.arange(len(dataset))
    random.shuffle(index)
    n = -200 * n_qubits
    train_index, test_index = index[:n], index[n:]
    train_dataset, test_dataset = np.array(
        dataset)[train_index], np.array(dataset)[test_index]

    with open(dataset_path, "wb")as f:
        pickle.dump((train_dataset, test_dataset), f)

else:
    with open(dataset_path, "rb")as f:
        train_dataset, test_dataset = pickle.load(f)

    with open(upstream_model_path, "rb")as f:
        upstream_model = pickle.load(f)

for circuit_info in train_dataset:
    print(circuit_info['n_erroneous_patterns'], len(circuit_info['gates']), circuit_info['n_erroneous_patterns'] / len(circuit_info['gates']), )
print(upstream_model.erroneous_pattern)

# TODO: 要用fidelity 大于 0.5的阶段
 
downstream_model = FidelityModel(upstream_model)
downstream_model.train(train_dataset, epoch_num = 1000)

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

    # print(predict)
with open(f"error_params_predicts_{n_qubits}.pkl", "wb")as f:
    pickle.dump((downstream_model.error_params, predicts, reals, durations), f)

find_error_path(upstream_model, downstream_model.error_params)


fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = plot_duration_fidelity(
    fig, axes, test_dataset)
fig.savefig(f"duration_fidelity_{n_qubits}_step1.svg")  # step
plt.close(fig)

# 画inaccuracy
from plot.plot import get_duration2circuit_infos

fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = get_duration2circuit_infos(durations, 100,0)
    
delta= []
for circuit_index in duration2circuit_index:
    delta.append(np.abs(reals[circuit_index] - predicts[circuit_index]).mean())
    
axes.plot(duration_X, delta ,markersize = 12,linewidth = 2, label='delta',marker = '^' )
axes.set_xlabel('duration ')
axes.set_ylabel('fidelity')
axes.legend() # 添加图例
fig.savefig(f"inaccuracy_fidelity_{n_qubits}_step1.svg")
print(np.array(delta).mean())