from circuit import gen_random_circuits
from circuit.formatter import qiskit_to_my_format_circuit, layered_circuits_to_qiskit
from upstream import RandomwalkModel
from collections import defaultdict

import math
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, gen_linear_topology, get_linear_neighbor_info
# from utils.backend import default_basis_single_gates, default_basis_two_gates

from qiskit import QuantumCircuit
import pennylane as qml
import numpy as np
from downstream.synthesis.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op import layer_circuit_to_matrix

from scipy.stats import unitary_group

from functools import lru_cache
import time
from qiskit import transpile
import random


from itertools import combinations_with_replacement, product

def generate_subcircuits(num_qubits, topology, num_gates, basic_gates):
    # Create a list of all possible combinations of basic gates
    all_gates = list(product(basic_gates, repeat=num_qubits))
    
    # Create a list of all possible combinations of gates
    all_combinations = list(combinations_with_replacement(all_gates, num_gates))
    
    # Create a list of all possible sub-circuits
    subcircuits = []
    for combination in all_combinations:
        for subcircuit in product(*combination):
            # Check if sub-circuit satisfies the topology
            valid = True
            for i in range(num_qubits):
                for j in range(i+1, num_qubits):
                    if topology[i][j] and subcircuit[i][j] not in topology[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                subcircuits.append(subcircuit)
    
    return subcircuits

num_qubits = 3
topology = np.ones((3,3))
num_gates = 2
basic_gates = ['x', 'y', 'z', 'cx']
# 'rx', 'ry', 'rz'
subcircuits = generate_subcircuits(num_qubits, topology, num_gates, basic_gates)


# print(subcircuits)
# n_qubits = 5
# topology = gen_linear_topology(n_qubits)
# neigh_info = get_linear_neighbor_info(n_qubits, 1)

# backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['rx', 'ry', 'rz'],  # i是idle
#                   basis_two_gates=['cz'], divide=False, decoupling=False)

# max_step = 2
# all_qubits = list(range(n_qubits))

# class Gate():
#     def __init__(self, op, qubits, layer):
#         self.op = op
#         self.qubits = qubits
#         self.layer = layer
        
#         # self._tuple = tuple([self.op] + self.qubits + ['-'] + [self.layer])
#         self._str = f'{self.op}{",".join(self.qubits)}-{self.layer}'
#         self._hash = hash(self._str)
    
#     def __str__(self):
#         return self._str
    
#     def __hash__(self) -> int:
#         return self._hash 
    
# def gate(op, qubits, layer):
#     qubits = list(qubits)
#     qubits.sort()
#     qubits = tuple(qubits) 
#     return _gen_gate(op, qubits, layer)

# @lru_cache
# def _gen_gate(op, qubits, layer):
#     return Gate(op, qubits, layer)


# '''假设neighbor_info就是topology'''

# def _is_occp(gate_path, gate): 
#     return

# def _scan(head_gate: Gate, gate_paths: set, traveled_qubits: set, now_step: int):
#     '''
#         traveled_qubits: 之前走过不会再去的qubits
#     '''
    
#     now_layer = head_gate.layer
#     candidates = set()
#     head_qubits = head_gate.qubits
    
#     for op in backend.basis_single_gates:
#         for _layer in range(now_layer-1, now_layer+2):
#             candidates.add(gate(op, [head_qubit], now_layer))
        
#     for op in backend.basis_two_gates:
#         for qubit in backend.topology[head_qubit]:
#             candidates.add(gate(op, [qubit, head_qubit], now_layer))
            
#     '''本层的candidate
#         1. 不能出现path中包含的比特
#         2. 
#     '''
#     return

# def gen_pathtable(head_qubit: int, backend: Backend):
    
#     now_layer = 0  # 假设几点的layer是0
    
#     candidates = set()
#     for op in backend.basis_single_gates:
#         candidates.add(gate(op, [head_qubit], now_layer))
        
#     for op in backend.basis_two_gates:
#         for qubit in backend.topology[head_qubit]:
#             candidates.add(gate(op, [qubit, head_qubit], now_layer))
    
#     traveled_qubits = set()
#     traveled_qubits.add(head_qubit)
#     gate_paths = set()
    
#     now_gate_path = []
#     for candidate in candidates:
#         # 深度优先
#         _scan(set(now_gate_path + candidate), traveled_qubits, 0)
    
#     return gate_paths