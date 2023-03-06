from circuit import gen_random_circuits
from circuit.formatter import qiskit_to_my_format_circuit, layered_circuits_to_qiskit, get_layered_instructions
from upstream import RandomwalkModel
from collections import defaultdict

import math
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, gen_linear_topology, get_linear_neighbor_info

from qiskit import QuantumCircuit
import pennylane as qml
import numpy as np
from downstream.synthesis.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op import layer_circuit_to_matrix

from scipy.stats import unitary_group

from downstream.synthesis.synthesis_model_pca_unitary import find_parmas, pkl_dump, pkl_load, matrix_distance_squared 
from downstream.synthesis.synthesis_model_pca_unitary import SynthesisModel, synthesize
from itertools import combinations
import time
from qiskit import transpile
import random

from circuit.formatter import layered_circuits_to_qiskit, to_unitary
from qiskit.quantum_info import Operator

# n_qubits = 2
# qiskit_circuit = QuantumCircuit(n_qubits)
# # qiskit_circuit.u(1, 1, 2, 0)
# qiskit_circuit.u(1, 2, 3, 0)
# # qiskit_circuit.u(1, 2, 3, 1)
# qiskit_circuit.u(0, 0, 0, 1)
# # qiskit_circuit.cz(1, 0)

# qiskit_unitary = Operator(qiskit_circuit).data
# # .reverse_bits()

# print(qiskit_circuit)
# qiskit_circuit = qiskit_circuit.reverse_bits()  # 真他妈离谱，画出来和转换出来的是不一样的
# # qiskit_circuit = QuantumCircuit(n_qubits)
# # qiskit_circuit.u(1, 2, 3, 1)
# # qiskit_circuit.u(0, 0, 0, 0)

# print(qiskit_circuit)
# layer2instructions, _, _, _, _ = get_layered_instructions(qiskit_circuit)  # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的
# synthesized_circuit, _, _ = qiskit_to_my_format_circuit(layer2instructions)  # 转成一个更不占内存的格式
# print(synthesized_circuit)
# synthesized_matrix = layer_circuit_to_matrix(synthesized_circuit, n_qubits)

# qiskit_circuit_2 = layered_circuits_to_qiskit(n_qubits, synthesized_circuit, False)
# qiskit_unitary_2 = Operator(qiskit_circuit_2).data

# # my_circuit = qml.from_qiskit(qiskit_circuit_2)

# print(matrix_distance_squared(synthesized_matrix, qiskit_unitary))
# print(matrix_distance_squared(synthesized_matrix, qiskit_unitary_2))
# print(matrix_distance_squared(qiskit_unitary, qiskit_unitary_2))


# gird topological information
# grid_num = 2
# topology = gen_grid_topology(grid_num) # 3x3 9 qubits
# neigh_info = get_grid_neighbor_info(grid_num, 1)
# n_qubits = grid_num ** 2

# init_unitary_mat = unitary_group.rvs(2**2)

n_qubits = 2
layer2gates = [
    [
        {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
        {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
    ],
    [{'name': 'cz', 'qubits': [1,0], 'params': []}],
    [
        {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
        {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
    ],
    [{'name': 'cz', 'qubits': [0,1], 'params': []}],
    [
        {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
        {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
    ],
    [{'name': 'cz', 'qubits': [0,1], 'params': []}],
    [
        {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
        {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
    ],
    [{'name': 'cz', 'qubits': [1,0], 'params': []}],
    [
        {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
        {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
    ],
]
U = unitary_group.rvs(2**n_qubits)
params = find_parmas(n_qubits, layer2gates, U, max_epoch=100, random_params= True, verbose = True)

# from qiskit.quantum_info import Operator

# circuit = QuantumCircuit(2)
# unitary = unitary_group.rvs(2**2)
# gate = Operator(unitary)
# circuit.append(gate, list(range(2)))
# unitary_circuit = transpile(circuit, optimization_level=3, basis_gates=['u', 'cz'])

# print(np.allclose(unitary, Operator(circuit).data))

# 2-qubit topological information

n_qubits = 5
topology = gen_linear_topology(n_qubits)
neigh_info = get_linear_neighbor_info(n_qubits, 1)

# print(to_unitary(np.ones((2**n_qubits, 2**n_qubits))))

backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                  basis_two_gates=['cz'], divide=False, decoupling=False)

model_name = f'q{n_qubits}_02204_dnn_unitary_eval_data' #f'q{n_qubits}_0220_pca_unitary'

# '''生成用于测试的模板电路'''
# max_gate = 2**n_qubits
# min_gate = 5
# circuits = gen_random_circuits(min_gate=min_gate, max_gate=max_gate, gate_num_step=max(
#     [1, (max_gate+max_gate)//50]), n_circuits=50, two_qubit_gate_probs=[3, 8], backend=backend, reverse=False, optimize=True, multi_process=True)

# '''生成一些一些比特之间没有纠缠的'''
# all_qubits = list(range(n_qubits))
# for i in range(len(all_qubits) + 1):
#     for subset in combinations(all_qubits, i):
#         if len(subset) <= 1 or len(subset) == n_qubits:
#             continue
#         n_qubit_subset = len(subset)
#         max_gate = 4**(n_qubit_subset+1)
#         print(subset, n_qubit_subset, max_gate)
#         sub_backend = backend.get_sub_backend(subset)
        
#         if len(sub_backend.coupling_map) == 0:
#             continue
        
#         circuits += gen_random_circuits(min_gate=min_gate, max_gate=max_gate, gate_num_step=max([1, (max_gate+max_gate)//50]), n_circuits=5, 
#                                        two_qubit_gate_probs=[3, 8], backend=sub_backend, reverse=False, optimize=True, multi_process=False)


# upstream_model = RandomwalkModel(
#     1, 20, backend=backend, travel_directions=('parallel', 'next'))
# upstream_model.train(circuits, multi_process=False)

# # random.shuffle(upstream_model.dataset)
# # circuits = upstream_model.dataset[:1000]
# synthesis_model = SynthesisModel(
#     upstream_model=upstream_model, name=model_name)
# synthesis_model.construct_data(circuits=circuits, multi_process=True)
# synthesis_model.construct_model()
# synthesis_model.save()

# synthesis_model: SynthesisModel = SynthesisModel.load(model_name)

# TODO: 一个model的放到一个文件夹

# while True:
# circuit = gen_random_circuits(min_gate=500, max_gate=501, gate_num_step=1, n_circuits=1, two_qubit_gate_probs=[
#                             3, 4], backend=backend, reverse=False, optimize=True, multi_process=False)[0]

# qiskit_circuit = layered_circuits_to_qiskit(
#     n_qubits, circuit['layer2gates'], barrier=False)
# print(qiskit_circuit)
# init_unitary_mat = layer_circuit_to_matrix(circuit['layer2gates'], n_qubits)

# params, dist = synthesis_model.find_parmas(n_qubits, circuits['layer2gates'], init_unitary_mat, max_epoch=50, allowed_dist=1e-2,
#                                 n_iter_no_change=5, no_change_tolerance=1e-3)

init_unitary_mat = unitary_group.rvs(2**n_qubits)
start_time = time.time()
synthesized_circuit = synthesize(init_unitary_mat, backend = backend, allowed_dist=1e-2, multi_process = True, heuristic_model=None, verbose=True)
# print(synthesized_circuit)
print('Synthesis costs', time.time() - start_time, 's')

# # 还要再递归的解决

qiskit_circuit = layered_circuits_to_qiskit(
    n_qubits, synthesized_circuit, barrier=False)
# qiskit_circuit = transpile(qiskit_circuit, coupling_map=backend.coupling_map, optimization_level=3, basis_gates=[
#                            'u', 'cz'], initial_layout=[qubit for qubit in range(n_qubits)])


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
# global phase: 2.9292
#          ┌───────────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐   ┌──────────────────────────┐
# q_0 -> 0 ┤ U(0.20947,-1.9891,3.0689) ├─■─┤ U(0.036089,-π/2,π/2) ├─■─┤ U(0.0080176,0,π/2) ├─■─┤ U(2.6347,-2.705,0.46604) ├
#          └┬──────────────────────────┤ │ └─┬──────────────────┬─┘ │ └─┬────────────────┬─┘ │ ├──────────────────────────┤
# q_1 -> 1 ─┤ U(1.7976,0.08717,1.6688) ├─■───┤ U(π/2,-2.1623,0) ├───■───┤ U(π/2,-π/2,-π) ├───■─┤ U(1.4675,-2.6786,1.4432) ├
#           └──────────────────────────┘     └──────────────────┘       └────────────────┘     └──────────────────────────┘
