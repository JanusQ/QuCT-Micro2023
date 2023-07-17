from circuit import gen_random_circuits
from circuit.algorithm.get_data_sys import get_dataset_alg_component
from circuit.formatter import qiskit_to_my_format_circuit, layered_circuits_to_qiskit, get_layered_instructions
from upstream import RandomwalkModel, extract_device
from collections import defaultdict

import math
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, gen_linear_topology, get_linear_neighbor_info, gen_fulllyconnected_topology

from qiskit import QuantumCircuit
import pennylane as qml
import numpy as np
from downstream.synthesis.pennylane_op_jax import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op_jax import layer_circuit_to_matrix

from scipy.stats import unitary_group

from downstream.synthesis.synthesis_model_pca_unitary_jax_可以跑但是最后会一直插一个 import SynthesisModelNN, SynthesisModelRandom, create_unitary_gate, find_parmas, pkl_dump, pkl_load, matrix_distance_squared, SynthesisModel, synthesize
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

def eval(n_qubits):
    n_step = n_qubits
    
    synthesis_model_name = f'synthesis_{n_qubits}_{n_step}NN'
    print('synthesize grover', n_qubits)
    
    init_unitary_mat = unitary_group.rvs(2**n_qubits)
    # init_unitary_mat = grover_U(n_qubits)
    topology = gen_fulllyconnected_topology(n_qubits)
    neigh_info = gen_fulllyconnected_topology(n_qubits)
    backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                    basis_two_gates=['cz'], divide=False, decoupling=False)
        
    regen = True
    if regen:

        min_gate, max_gate = max([2**n_qubits - 200, 10]), 2**n_qubits
        dataset = gen_random_circuits(min_gate=min_gate, max_gate=max_gate, gate_num_step=max_gate//20, n_circuits=10,
                                    two_qubit_gate_probs=[4, 7], backend=backend, reverse=False, optimize=True, multi_process=True, circuit_type = 'random')[:10]
        
        # dataset += get_dataset_alg_component(n_qubits, backend)

        upstream_model = RandomwalkModel(n_step, 4**n_step, backend, travel_directions=('parallel', 'next'))
        upstream_model.train(dataset, multi_process=False, remove_redundancy=False, full_vec=False, min_count=0)

        for circuit_info in dataset:
            for vec, gate in zip(circuit_info['sparse_vecs'], circuit_info['gates']):
                print(vec)
                print(layered_circuits_to_qiskit(n_qubits, upstream_model.reconstruct(extract_device(gate), vec)))
                print()
                
        synthesis_model = SynthesisModelNN(upstream_model, synthesis_model_name)
        data = synthesis_model.construct_data(dataset, multi_process=True)
        print(f'data size of {synthesis_model_name} is {len(data[0])}')
        synthesis_model.construct_model(data)
        synthesis_model.save()
    else:
        synthesis_model = SynthesisModelRandom(backend)
        
        # synthesis_model: SynthesisModel = SynthesisModel.load(synthesis_model_name)
        # backend: Backend = synthesis_model.backend
    
    print('synthesize', backend.n_qubits)
    
    for use_heuristic in [False]:
        start_time = time.time()

        # TODO:给他直接喂算法的电路
        synthesis_log = {}
        synthesized_circuit, cpu_time = synthesize(init_unitary_mat, backend=backend, allowed_dist=1e-2,
                                                multi_process=True, heuristic_model=synthesis_model if use_heuristic else None,
                                                verbose=True, lagre_block_penalty=3, synthesis_log = synthesis_log)
        synthesis_time = time.time() - start_time

        qiskit_circuit = layered_circuits_to_qiskit(
            n_qubits, synthesized_circuit, barrier=False)

        result = {
            'n_qubits': n_qubits,
            'U': init_unitary_mat,
            'qiskit circuit': qiskit_circuit,
            '#gate': len(qiskit_circuit),
            '#two-qubit gate': cnot_count(qiskit_circuit) + cz_count(qiskit_circuit),
            'depth': qiskit_circuit.depth(),
            # 'synthesis time': synthesis_time,  # 直接函数里面会存的
            'cpu time': cpu_time,
            'use heuristic': use_heuristic,
        }        
        result.update(synthesis_log)
        
        print({ key: item for key, item in result.items() if key not in ('print', 'qiskit circuit', 'U')})
        print('\n')

        with open(f'temp_data/synthesis_result/3_29/{n_qubits}_{use_heuristic}_grover_result.pkl', 'wb') as f:
            pickle.dump(result, f)

@ray.remote
def eval_remote(*args):
    return eval(*args)

# n_qubits = 3
for n_qubits in range(5, 7):

    futures = []
    # futures.append(eval_remote.remote(n_qubits))
    futures.append(eval(n_qubits))
        
wait(futures)