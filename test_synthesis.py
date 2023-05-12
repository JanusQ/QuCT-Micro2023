from circuit import gen_random_circuits
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

from downstream.synthesis.synthesis_model_pca_unitary_jax import find_parmas, pkl_dump, pkl_load, matrix_distance_squared, SynthesisModel, synthesize
from itertools import combinations
import time
from qiskit import transpile
import random
import cloudpickle as pickle
from circuit.formatter import layered_circuits_to_qiskit, to_unitary
from qiskit.quantum_info import Operator

from utils.unitaries import qft_U, grover_U

# n_qubits = 2
# layer2gates = [
#     [
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#         {'name': 'u', 'qubits': [1], 'params': [np.pi]*3},
#     ],
#     [{'name': 'cz', 'qubits': [0,1], 'params': []}],
#     [
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#         {'name': 'u', 'qubits': [1], 'params': [np.pi]*3},
#     ],
#     [{'name': 'cz', 'qubits': [0,1], 'params': []}],
#     [
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#         {'name': 'u', 'qubits': [1], 'params': [np.pi]*3},
#     ],
#     [{'name': 'cz', 'qubits': [0,1], 'params': []}],
#     [
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#         {'name': 'u', 'qubits': [1], 'params': [np.pi]*3},
#     ],
#     [{'name': 'cz', 'qubits': [0,1], 'params': []}],
#     [
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#         {'name': 'u', 'qubits': [1], 'params': [np.pi]*3},
#     ],
# ]
# U = unitary_group.rvs(2**n_qubits)
# params = find_parmas(n_qubits, layer2gates, U, lr=1e-1, max_epoch=1000, allowed_dist=1e-2, n_iter_no_change=100, no_change_tolerance=1e-2, random_params= True, verbose = True)

# model_name = f'q{n_qubits}_02204_dnn_unitary_eval_data' #f'q{n_qubits}_0220_pca_unitary'
        
for n_qubits in range(3, 4):
    for index in range(5):
        # n_qubits = 5
        
        topology = gen_fulllyconnected_topology(n_qubits)
        neigh_info = gen_fulllyconnected_topology(n_qubits)
        
        # topology = gen_linear_topology(n_qubits)
        # neigh_info = get_linear_neighbor_info(n_qubits, 1)

        backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                        basis_two_gates=['cz'], divide=False, decoupling=False)

        init_unitary_mat = unitary_group.rvs(2**n_qubits)
        # init_unitary_mat = qft_U(n_qubits)
        start_time = time.time()
        synthesized_circuit = synthesize(init_unitary_mat, backend = backend, allowed_dist=1e-2, multi_process = True, heuristic_model=None, verbose=True, lagre_block_penalty = 4)
        # print(synthesized_circuit)
        
        synthesis_time = time.time() - start_time
        print('Synthesis costs', time.time() - start_time, 's')

        qiskit_circuit = layered_circuits_to_qiskit(
            n_qubits, synthesized_circuit, barrier=False)

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

        # layer_U: jnp.array = layer_circuit_to_matrix(layer2gates, n_qubits)
        print(qiskit_circuit)
        print('gate = ', len(qiskit_circuit))
        print('#two-qubit gates = ', cnot_count(qiskit_circuit) + cz_count(qiskit_circuit))
        print('depth = ', qiskit_circuit.depth())
        print('finish')
        print()
        
        result  = {
            'index': index,
            'n_qubits': n_qubits,
            'init_unitary_mat': init_unitary_mat,
            'qiskit_circuit': qiskit_circuit,
            '#gate': len(qiskit_circuit),
            '#two-qubit gate': cnot_count(qiskit_circuit) + cz_count(qiskit_circuit),
            'depth': qiskit_circuit.depth(),
            'synthesis_time': synthesis_time,
        }
        
        with open(f'temp_data/synthesis_result/3_20/{n_qubits}_{index}_result.pkl', 'wb') as f:
            pickle.dump(result, f)
        
# global phase: 2.9292
#          ┌───────────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐   ┌──────────────────────────┐
# q_0 -> 0 ┤ U(0.20947,-1.9891,3.0689) ├─■─┤ U(0.036089,-π/2,π/2) ├─■─┤ U(0.0080176,0,π/2) ├─■─┤ U(2.6347,-2.705,0.46604) ├
#          └┬──────────────────────────┤ │ └─┬──────────────────┬─┘ │ └─┬────────────────┬─┘ │ ├──────────────────────────┤
# q_1 -> 1 ─┤ U(1.7976,0.08717,1.6688) ├─■───┤ U(π/2,-2.1623,0) ├───■───┤ U(π/2,-π/2,-π) ├───■─┤ U(1.4675,-2.6786,1.4432) ├
#           └──────────────────────────┘     └──────────────────┘       └────────────────┘     └──────────────────────────┘
