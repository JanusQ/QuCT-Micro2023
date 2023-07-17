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


def eval(name, U, n_qubits, synthesis_model: SynthesisModel, n_unitary_candidates):
    target_backend = synthesis_model.backend

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
            # 'n_neighbors': n_neighbors,
        }
        result.update(synthesis_log)

        print('RESULT')
        print({key: item for key, item in result.items()
              if key not in ('print', 'qiskit circuit', 'U')})
        print('\n')

        with open(f'{n_qubits}/result/{use_heuristic}_{n_unitary_candidates}_{name}.pkl', 'wb') as f:
            pickle.dump(result, f)


@ray.remote
def eval_remote(*args):
    return eval(*args)

    # '8/result/ghz_8qiskit_only.json',
     # '8/result/grover_8qiskit_only.json',
    # '8/result/ising_8qiskit_only.json',
    # '8/result/qsvm_8qiskit_only.json',
n_qubits = 8
futures = []
paths = [   
    # '8/result/qft_8qiskit_only.json',
    # '8/result/qknn_8qiskit_only.json',
    '8/result/hamiltonian_simulation_8qiskit_only.json',
    '8/result/vqc_8qiskit_only.json',
]


# topology = gen_fulllyconnected_topology(n_qubits)
# neigh_info = gen_fulllyconnected_topology(n_qubits)

topology = gen_linear_topology(n_qubits)
neigh_info = get_linear_neighbor_info(n_qubits, 1)

target_backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                        basis_two_gates=['cz'], divide=False, decoupling=False)

for path in paths:
    with open(path, mode='rb') as f:
        result_dict = json.load(f)
        # Deserialize fields
        picked_unitary = json.loads(result_dict['Unitary']).encode('latin-1')
    unitary = pickle.loads(picked_unitary)
    name = result_dict['Experiment Name']
        
    n_unitary_candidates = 4*n_qubits

    synthesis_model = SynthesisModelRandom(target_backend)

    # futures.append(eval_remote.remote(n_batches*epoch_index + batch_index, unitary_group.rvs(2**n_qubits), n_qubits, synthesis_model, n_unitary_candidates, n_neighbors))
    futures.append(eval(name, unitary, n_qubits, synthesis_model, n_unitary_candidates)) 

wait(futures)
    
