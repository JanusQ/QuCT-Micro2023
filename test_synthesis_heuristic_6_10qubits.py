from circuit import gen_random_circuits
from circuit.algorithm.get_data_sys import get_dataset_alg_component
from circuit.formatter import qiskit_to_my_format_circuit, layered_circuits_to_qiskit, get_layered_instructions
from upstream import RandomwalkModel
from collections import defaultdict

import math
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, gen_linear_topology, get_linear_neighbor_info, gen_fulllyconnected_topology

from qiskit import QuantumCircuit
import pennylane as qml
import numpy as np
from downstream.synthesis.pennylane_op_jax import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op_jax import layer_circuit_to_matrix

from scipy.stats import unitary_group

from downstream.synthesis.synthesis_model_pca_unitary_jax import SynthesisModelNN, SynthesisModelRandom, create_unitary_gate, find_parmas, pkl_dump, pkl_load, matrix_distance_squared, SynthesisModel, synthesize
from itertools import combinations
import time
from qiskit import transpile
import random
import cloudpickle as pickle
from circuit.formatter import layered_circuits_to_qiskit, to_unitary
from qiskit.quantum_info import Operator
import ray
from utils.ray_func import wait
from utils.unitaries import qft_U, grover_U, optimal_grover
import os
import json


def cnot_count(qc: QuantumCircuit):
    count_ops = qc.count_ops()
    if 'cx' in count_ops:
        return count_ops['cx']
    return 0


def cz_count(qc: QuantumCircuit):
    count_ops = qc.count_ops()
    if 'cz' in count_ops:
        return count_ops['cz']
    return 0


def eval(index, U, n_qubits, synthesis_model: SynthesisModel, n_unitary_candidates, n_neighbors):
    # topology = gen_linear_topology(n_qubits)
    # neigh_info = gen_fulllyconnected_topology(n_qubits)
    # target_backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
    #                         basis_two_gates=['cz'], divide=False, decoupling=False)

    # synthesis_model = SynthesisModelRandom(target_backend)
    if isinstance(synthesis_model, str):
        synthesis_model: SynthesisModel = SynthesisModel.load(synthesis_model)

    target_backend = synthesis_model.backend
    # assert  synthesis_model.backend == target_backend  #TODO

    print('synthesize', target_backend.n_qubits)

    for use_heuristic in [False]:
        # TODO:给他直接喂算法的电路
        synthesis_log = {}

        # print(matrix_distance_squared(U @ U.T.conj(), np.eye(2**n_qubits)))
        assert matrix_distance_squared(
            U @ U.T.conj(), np.eye(2**n_qubits)) < 1e-4
        synthesized_circuit, cpu_time = synthesize(U, backend=target_backend, allowed_dist=5e-2,
                                                   multi_process=True, heuristic_model=synthesis_model if use_heuristic else None,
                                                   verbose=True, lagre_block_penalty=4, synthesis_log=synthesis_log, n_unitary_candidates=n_unitary_candidates, timeout=7*24*3600)

        qiskit_circuit = layered_circuits_to_qiskit(
            n_qubits, synthesized_circuit, barrier=False)

        result = {
            'n_qubits': n_qubits,
            'U': U,
            'qiskit circuit': qiskit_circuit,
            '#gate': len(qiskit_circuit),
            '#two-qubit gate': cnot_count(qiskit_circuit) + cz_count(qiskit_circuit),
            'depth': qiskit_circuit.depth(),
            'cpu time': cpu_time,
            'use heuristic': use_heuristic,
            'n_unitary_candidates': n_unitary_candidates,
            # 'baseline_name': filename,
            # 'baseline_dir': dirpath,
            'n_neighbors': n_neighbors,
        }
        result.update(synthesis_log)

        print('RESULT')
        print({key: item for key, item in result.items()
              if key not in ('print', 'qiskit circuit', 'U')})
        print('\n')

        with open(f'{n_qubits}/result/{use_heuristic}_{n_unitary_candidates}_{index}_{n_neighbors}.pkl', 'wb') as f:
            pickle.dump(result, f)


@ray.remote
def eval_remote(*args):
    return eval(*args)


for n_qubits in [8]:
    print('#Qubit', n_qubits)
    n_step = 4

    n_neighbors = n_qubits * 2

    topology = gen_fulllyconnected_topology(n_qubits)
    neigh_info = gen_fulllyconnected_topology(n_qubits)
    target_backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                            basis_two_gates=['cz'], divide=False, decoupling=False)
                            
    min_gate, max_gate = max([4**n_qubits - 100, 10]), 4**n_qubits
    # max_gate//20
    dataset = gen_random_circuits(min_gate=max_gate-20, max_gate=max_gate, gate_num_step=10, n_circuits=3,
                                two_qubit_gate_probs=[4, 7], backend=target_backend, reverse=False, optimize=False, multi_process=True, circuit_type='random')

    # dataset += get_dataset_alg_component(n_qubits, backend)

    start_time = time.time()
    
    upstream_model = RandomwalkModel(
        n_step, 50, target_backend, travel_directions=('parallel', 'next'))
    upstream_model.train(dataset, multi_process=True,
                        remove_redundancy=False, full_vec=False)

    print('Path table size is', upstream_model.max_table_size)
    print('Device', len(upstream_model.device2path_table))

    synthesis_model = SynthesisModelNN(upstream_model)
    data = synthesis_model.construct_data(dataset, multi_process=True, n_random=1)
    synthesis_model.construct_model(data, n_neighbors)

    print(f'Data size is {len(data[0])}')
    print('model construction', time.time() - start_time)
    print('\n\n\n\n')


exit()

# list(range(6, 10, 2))
for n_qubits in [6, 8]:
    futures = []

    n_unitary_candidates = 3*n_qubits

    # if n_qubits == 5:
    #     n_neighbors = 7
    # elif n_qubits == 4:
    #     n_neighbors = 10
    # else:
    n_neighbors = n_qubits * 2

    topology = gen_fulllyconnected_topology(n_qubits)
    neigh_info = gen_fulllyconnected_topology(n_qubits)
    # topology = gen_linear_topology(n_qubits)
    # neigh_info = gen_fulllyconnected_topology(n_qubits)
    target_backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                            basis_two_gates=['cz'], divide=False, decoupling=False)


    n_step = 5
    synthesis_model_name = f'synthesis_{n_qubits}_{n_step}_{n_neighbors}NN'
    if False:
        min_gate, max_gate = max([4**n_qubits - 100, 10]), 4**n_qubits
        # max_gate//20
        dataset = gen_random_circuits(min_gate=min_gate, max_gate=max_gate, gate_num_step=20, n_circuits=5,
                                    two_qubit_gate_probs=[4, 7], backend=target_backend, reverse=False, optimize=True, multi_process=True, circuit_type='random')

        # dataset += get_dataset_alg_component(n_qubits, backend)

        upstream_model = RandomwalkModel(
            n_step, 50, target_backend, travel_directions=('parallel', 'next'))
        upstream_model.train(dataset, multi_process=True,
                            remove_redundancy=False, full_vec=False)

        synthesis_model = SynthesisModelNN(upstream_model, synthesis_model_name)
        data = synthesis_model.construct_data(dataset, multi_process=True, n_random=2)
        print(f'data size of {synthesis_model_name} is {len(data[0])}')
        synthesis_model.construct_model(data, n_neighbors)
        synthesis_model.save()
    else:
        synthesis_model = SynthesisModelRandom(target_backend)
        # synthesis_model: SynthesisModel = SynthesisModel.load(synthesis_model_name)
        # target_backend: Backend = synthesis_model.backend
        pass

    for epoch_index in range(1, 2): # 先只跑两个
        n_batches = 2
        futures = []
        for batch_index in range(n_batches):
            # futures.append(eval_remote.remote(n_batches*epoch_index + batch_index, unitary_group.rvs(2**n_qubits), n_qubits, synthesis_model, n_unitary_candidates, n_neighbors))
            futures.append(eval(2 + n_batches*epoch_index + batch_index, unitary_group.rvs(2**n_qubits), n_qubits, synthesis_model, n_unitary_candidates, n_neighbors))
        wait(futures)
        
