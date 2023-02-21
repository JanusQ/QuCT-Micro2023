from circuit import gen_random_circuits
from circuit.formatter import qiskit_to_my_format_circuit, layered_circuits_to_qiskit
from upstream import RandomwalkModel
from collections import defaultdict

import math
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, gen_linear_topology, get_linear_neighbor_info
# from utils.backend import default_basis_single_gates, default_basis_two_gates


import pennylane as qml
import numpy as np
from downstream.synthesis.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op import layer_circuit_to_matrix
from downstream.synthesis import matrix_distance_squared
from downstream.synthesis.synthesis_model import pkl_dump, pkl_load

from downstream.synthesis import SynthesisModel
from scipy.stats import unitary_group
from qiskit import transpile
# gird topological information
# grid_num = 2
# topology = gen_grid_topology(grid_num) # 3x3 9 qubits
# neigh_info = get_grid_neighbor_info(grid_num, 1)
# n_qubits = grid_num ** 2

# 2-qubit topological information
n_qubits = 2
topology = gen_linear_topology(n_qubits)
neigh_info = get_linear_neighbor_info(n_qubits, 1)


backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = ['u'], 
                  basis_two_gates = ['cz'], divide = False, decoupling=False)


'''生成用于测试的模板电路'''
# max_gate = 100 #4**n_qubits
# circuits = gen_random_circuits(min_gate = 10, max_gate = max_gate, gate_num_step = 20, n_circuits = 10, two_qubit_gate_probs=[4, 8], backend = backend, reverse = False, optimize = True, multi_process = False)
# upstream_model_path = f'./temp_data/upstream_model_q{n_qubits}_0220.pkl'
# upstream_model = RandomwalkModel(1, 20, backend = backend, travel_directions=('parallel', 'next'))
# upstream_model.train(circuits, multi_process = False)
# # pkl_dump(upstream_model ,upstream_model_path)


# # upstream_model: RandomwalkModel = pkl_load(upstream_model_path)

model_name = f'q{n_qubits}_0220'
# synthesis_model = SynthesisModel(upstream_model = upstream_model, name = model_name)
# synthesis_model.construct_data(upstream_model.dataset, multi_process=True)
# synthesis_model.construct_model()
# synthesis_model.save()


synthesis_model: SynthesisModel = SynthesisModel.load(model_name)

# TODO: 一个model的放到一个文件夹
init_unitary_mat = unitary_group.rvs(2**n_qubits)
layer2gates = synthesis_model.synthesize(init_unitary_mat)
qiskit_circuit = layered_circuits_to_qiskit(n_qubits, layer2gates, barrier = False)
qiskit_circuit = transpile(qiskit_circuit, coupling_map=backend.coupling_map, optimization_level=3, basis_gates=['u', 'cz'], initial_layout=[qubit for qubit in range(n_qubits)])

print(qiskit_circuit)
# qiskit_to_my_format_circuit, layered_circuits_to_qiskit
print('finish')

# global phase: 2.9292
#          ┌───────────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐   ┌──────────────────────────┐
# q_0 -> 0 ┤ U(0.20947,-1.9891,3.0689) ├─■─┤ U(0.036089,-π/2,π/2) ├─■─┤ U(0.0080176,0,π/2) ├─■─┤ U(2.6347,-2.705,0.46604) ├
#          └┬──────────────────────────┤ │ └─┬──────────────────┬─┘ │ └─┬────────────────┬─┘ │ ├──────────────────────────┤
# q_1 -> 1 ─┤ U(1.7976,0.08717,1.6688) ├─■───┤ U(π/2,-2.1623,0) ├───■───┤ U(π/2,-π/2,-π) ├───■─┤ U(1.4675,-2.6786,1.4432) ├
#           └──────────────────────────┘     └──────────────────┘       └────────────────┘     └──────────────────────────┘
