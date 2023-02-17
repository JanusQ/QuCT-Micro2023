from circuit import gen_random_circuits
from upstream import RandomwalkModel
from collections import defaultdict

import math
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend
# from utils.backend import default_basis_single_gates, default_basis_two_gates


import pennylane as qml
import numpy as np
from downstream.synthesis.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op import layer_circuit_to_matrix
from downstream.synthesis import matrix_distance_squared

# layer2gates = [
#     [{
#         'name': 'u',
#         'params': [np.pi/2, np.pi/2, np.pi/2,],
#         'qubits': [0]
#     }],
# ]

# n_qubits = 1
# U = layer_circuit_to_pennylane_tape(layer2gates)

# qml_m = qml.matrix(U)
# tc_m = layer_circuit_to_matrix(layer2gates, n_qubits)


# print(np.allclose(tc_m, qml_m))   # 现在是true了
# print(matrix_distance_squared(tc_m, qml_m))

# topological information
grid_num = 2
topology = gen_grid_topology(grid_num) # 3x3 9 qubits
neigh_info = get_grid_neighbor_info(grid_num, 1)

# print(topology)
# print(coupling_map)
# print(neigh_info)

n_qubits = grid_num ** 2
backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = ['u'], 
                  basis_two_gates = ['cz'], divide = False, decoupling=False)


# upstream_model.load_reduced_vecs()

from downstream.synthesis import SynthesisModel
from scipy.stats import unitary_group


synthesis_model = SynthesisModel(backend=backend)
synthesis_model.construct_model()

init_unitary_mat = unitary_group.rvs(2**n_qubits)
print('finish')