from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.parser import get_circuit_duration
from circuit.utils import make_circuitlet, stitching
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path
import random
import numpy as np

from circuit import gen_random_circuits, label_ground_truth_fidelity
from simulator.noise_simulator import get_random_erroneous_pattern
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator
from utils.backend import devide_chip, gen_grid_topology, get_devide_qubit, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle

def gen_train_dataset(n_qubits, topology, neighbor_info, coupling_map, dataset_size, max_cut_type, min_cut_qubit = 5, devide_size=5,circuit_type = 'random'):
    covered_couplng_map = set()
    dataset = []

    devide_cut_backends = []
    devide_maps = []
    deivde_reverse_maps = []
    all_devide_qubits = []
    cut_type = 0
    while cut_type< max_cut_type:
        before = len(covered_couplng_map)
        devide_qubits = get_devide_qubit(topology, devide_size)

        _devide_qubits = []
        for devide_qubit in devide_qubits:
            if len(devide_qubit) >= min_cut_qubit:
                _devide_qubits.append(devide_qubit)
        
        devide_qubits = _devide_qubits
        
        cut_backends = []
        maps = []
        reverse_maps = []
        for i in range(len(devide_qubits)):
            _map = {}
            _reverse_map = {}
            for idx, qubit in enumerate(devide_qubits[i]):
                _map[idx] = qubit
                _reverse_map[qubit] = idx
            maps.append(_map)
            reverse_maps.append(_reverse_map)

            cut_coupling_map = []
            for ele in coupling_map:
                if ele[0] in devide_qubits[i] and ele[1] in devide_qubits[i]:
                    cut_coupling_map.append(
                        (_reverse_map[ele[0]], _reverse_map[ele[1]]))
                    covered_couplng_map.add(tuple(ele))

            cut_backends.append(Backend(n_qubits=len(devide_qubits[i]), topology=topology, neighbor_info=neighbor_info, coupling_map=cut_coupling_map,
                                basis_single_gates=default_basis_single_gates, basis_two_gates=default_basis_two_gates, divide=False, decoupling=False))

        if before == len(covered_couplng_map) and len(covered_couplng_map) != len(coupling_map):
            continue

        print(devide_qubits)
        all_devide_qubits.append(devide_qubits)
        devide_cut_backends.append(cut_backends)
        devide_maps.append(maps)
        deivde_reverse_maps.append(reverse_maps)
        
        cut_type += 1


    if len(covered_couplng_map) != len(coupling_map):
        return  []

    # dataset_5qubit = []
    n_circuits = dataset_size // 60 // len(devide_cut_backends)
    for cut_backends, devide_qubits, maps,  reverse_maps in zip(devide_cut_backends, all_devide_qubits, devide_maps,  deivde_reverse_maps):
        cut_datasets = []
        for cut_backend in cut_backends:
            _dataset = gen_random_circuits(min_gate=20, max_gate=170, n_circuits=n_circuits, two_qubit_gate_probs=[
                                            1, 5], gate_num_step=10, backend=cut_backend, multi_process=True,circuit_type=circuit_type)
            cut_datasets.append(_dataset)

        # def get_n_instruction2circuit_infos(dataset):
        #     n_instruction2circuit_infos = defaultdict(list)
        #     for circuit_info in dataset:
        #         # qiskit_circuit = circuit_info['qiskit_circuit']
        #         gate_num = len(circuit_info['gates'])
        #         n_instruction2circuit_infos[gate_num].append(circuit_info)

        #     # print(n_instruction2circuit_infos[gate_num])
        #     gate_nums = list(n_instruction2circuit_infos.keys())
        #     gate_nums.sort()

        #     return n_instruction2circuit_infos, gate_nums

        # n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(cut_datasets[0])
        # for gate in gate_nums:
        #     print(gate, len(n_instruction2circuit_infos[gate]))
        
        dataset += stitching(n_qubits, cut_datasets,
                             devide_qubits, maps, reverse_maps, align = False)

    print(len(dataset), "circuit generated")
    # print(len(dataset_5qubit), "5bit circuit generated")
        
    return dataset

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
neighbor_info = copy.deepcopy(topology)
coupling_map = topology_to_coupling_map(topology)      



backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                  basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)
max_cut_type = 4
for t in range(100):
    dataset = gen_train_dataset(n_qubits, topology, neighbor_info, coupling_map, 4000, max_cut_type=max_cut_type)
    if len(dataset) != 0:
        break

if len(dataset) == 0:
    raise Exception('can not cover coupling map')
upstream_model = RandomwalkModel(1, 20, backend=backend, travel_directions=('parallel', 'former'))
print(len(dataset), "circuit generated")
upstream_model.train(dataset, multi_process=True)

with open(f"upstream_model_{max_cut_type}.pkl","wb")as f:
    pickle.dump(upstream_model,f)

print("original",len(dataset))
dataset = make_circuitlet(dataset)
print("cutted",len(dataset))

with open(f"cutted_dataset_{max_cut_type}.pkl","wb")as f:
    pickle.dump(dataset,f)

simulator = NoiseSimulator(backend)
erroneous_pattern = get_random_erroneous_pattern(upstream_model)
erroneous_pattern = simulator.get_error_results(dataset, upstream_model, erroneous_pattern, multi_process=True)
upstream_model.erroneous_pattern = erroneous_pattern

with open(f"cutted_dataset_{max_cut_type}.pkl","wb")as f:
    pickle.dump(dataset,f)

with open(f"upstream_model_{max_cut_type}.pkl","wb")as f:
    pickle.dump(upstream_model,f)


random.shuffle(dataset)
train_dataset, test_dataset  = dataset[:-1000], dataset[-1000:]

    
downstream_model = FidelityModel(upstream_model)
downstream_model.train(train_dataset)

predicts, reals, durations = [], [], []
for idx, cir in enumerate(test_dataset):
    if idx % 100 == 0:
        print(idx, "predict finished!")
    predict = downstream_model.predict_fidelity(cir)

    predicts.append(cir['circuit_predict'])
    reals.append(cir['ground_truth_fidelity'])
    durations.append(cir['duration'])
print(np.abs(np.array(predicts) - np.array(reals)).mean())

with open(f"res_{max_cut_type}.pkl","wb")as f:
    pickle.dump((downstream_model, test_dataset, predicts, reals, durations), f)
    
