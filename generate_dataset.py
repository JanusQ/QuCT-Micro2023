from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.dataset_loader import gen_algorithms
from circuit.formatter import layered_circuits_to_executable_code
from circuit.formatter import layered_circuits_to_qiskit

from circuit.parser import get_circuit_duration
from circuit.utils import get_extra_info, stitching
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path
import random
import numpy as np

from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator
from utils.backend import devide_chip, gen_grid_topology, get_devide_qubit, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle


def gen_validate_dataset(n_qubits, topology, neighbor_info, coupling_map):
    backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                      basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

    dataset = gen_random_circuits(min_gate=24, max_gate=2004, n_circuits=15, two_qubit_gate_probs=[
                                  1, 5], gate_num_step=10, backend=backend, multi_process=True)

    dataset_machine = []
    for cir in dataset:
        dataset_machine.append(cir['layer2gates'])

    with open('execute_18bits_validate.pkl', 'wb')as f:
        pickle.dump(dataset_machine, f)

    with open('execute_18bits_validate_more_info.pkl', 'wb')as f:
        pickle.dump(dataset, f)


def gen_train_dataset(n_qubits, topology, neighbor_info, coupling_map, dataset_size, devide_size=5):
    covered_couplng_map = set()
    dataset = []

    devide_cut_backends = []
    devide_maps = []
    deivde_reverse_maps = []
    all_devide_qubits = []
    while True:
        before = len(covered_couplng_map)
        devide_qubits = get_devide_qubit(topology, devide_size)
        
        '''不能有只有一个比特的'''
        '''这明明是全等于5'''
        # has_1_qubit = False
        # for devide_qubit in devide_qubits:
        #     if len(devide_qubit) < devide_size:
        #         has_1_qubit = True
        # if has_1_qubit:
        #     continue
        
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

        if before == len(covered_couplng_map):
            continue

        print(devide_qubits)
        all_devide_qubits.append(devide_qubits)
        devide_cut_backends.append(cut_backends)
        devide_maps.append(maps)
        deivde_reverse_maps.append(reverse_maps)

        if len(covered_couplng_map) == len(coupling_map):
            break

    # dataset_5qubit = []
    n_circuits = dataset_size // 50 // len(devide_cut_backends)
    for cut_backends, devide_qubits, maps,  reverse_maps in zip(devide_cut_backends, all_devide_qubits, devide_maps,  deivde_reverse_maps):
        cut_datasets = []
        for cut_backend in cut_backends:
            if cut_backend.n_qubits == 3:
                _dataset = gen_random_circuits(min_gate=14, max_gate=144, n_circuits=n_circuits, two_qubit_gate_probs=[
                                               1, 5], gate_num_step=10, backend=cut_backend, multi_process=True)
                cut_datasets.append(_dataset)
                # dataset_5qubit += _dataset
            elif cut_backend.n_qubits == 4:
                _dataset = gen_random_circuits(min_gate=12, max_gate=142, n_circuits=n_circuits, two_qubit_gate_probs=[
                                               1, 5], gate_num_step=10, backend=cut_backend, multi_process=True)
                cut_datasets.append(_dataset)
                # dataset_5qubit += _dataset
            else:
                _dataset = gen_random_circuits(min_gate=20, max_gate=150, n_circuits=n_circuits, two_qubit_gate_probs=[
                                               1, 5], gate_num_step=10, backend=cut_backend, multi_process=True)
                cut_datasets.append(_dataset)
                # dataset_5qubit += _dataset

        dataset += stitching(n_qubits, cut_datasets,
                             devide_qubits, maps, reverse_maps)

    print(len(dataset), "circuit generated")
    # print(len(dataset_5qubit), "5bit circuit generated")

    return dataset
    # dataset_machine = []
    # for cir in dataset:
    #     dataset_machine.append(cir['layer2gates'])

    # with open('execute_18bits_train.pkl','wb')as f:
    #     pickle.dump(dataset_machine, f)

    # with open('execute_18bits_train_more_info.pkl','wb')as f:
    #     pickle.dump(dataset, f)


def gen_algorithm_dataset(n_qubits, topology, neighbor_info, coupling_map, mirror):
    algos = gen_algorithms(n_qubits, coupling_map, mirror)

    get_extra_info(algos)
    for algo in algos:
        # print(layered_circuits_to_qiskit(18, algo['layer2gates']))
        print(algo['id'], len(algo['layer2gates']), len(
            algo['gates']), algo['duration'], algo['prop'])
    print(len(algos))
    algos_machine = []
    for cir in algos:
        algos_machine.append(cir['layer2gates'])

    title = "_mirror" if mirror else ""
    with open(f'execute_18bits_algos{title}.pkl', 'wb')as f:
        pickle.dump(algos_machine, f)

    with open(f'execute_18bits_algos_more_info{title}.pkl', 'wb')as f:
        pickle.dump(algos, f)


# size = 6
# n_qubits = 18
# topology = gen_grid_topology(size)  # 3x3 9 qubits
# new_topology = defaultdict(list)
# for qubit in topology.keys():
#     if qubit < n_qubits:
#         for ele in topology[qubit]:
#             if ele < n_qubits:
#                 new_topology[qubit].append(ele)
# topology = new_topology
# neighbor_info = copy.deepcopy(topology)
# coupling_map = topology_to_coupling_map(topology)

# gen_train_dataset(n_qubits, topology, neighbor_info, coupling_map)
# gen_validate_dataset(n_qubits, topology, neighbor_info, coupling_map)
# gen_algorithm_dataset(n_qubits, topology, neighbor_info, coupling_map, True)
# gen_algorithm_dataset(n_qubits, topology, neighbor_info, coupling_map, False)
