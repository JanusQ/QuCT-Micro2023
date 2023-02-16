from circuit import gen_random_circuits
from upstream import RandomwalkModel
from downstream import FidelityModel
from collections import defaultdict

import math
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates

# topological information
topology = gen_grid_topology(3)  # 3x3 9 qubits
neigh_info = get_grid_neighbor_info(3, 1)

print(topology)
print(neigh_info)

n_qubits = 9
backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info,
                  basis_single_gates=default_basis_single_gates,
                  basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

train_dataset = gen_random_circuits(min_gate=10, max_gate=100, n_circuits=2, two_qubit_gate_probs=[4, 8],
                                    backend=backend)

upstream_model = RandomwalkModel(1, 20, backend=backend)
upstream_model.train(train_dataset, multi_process=True)

# upstream_model.load_reduced_vecs()

test_dataset = gen_random_circuits(min_gate=10, max_gate=100, n_circuits=1, two_qubit_gate_probs=[4, 8],
                                    backend=backend)
circuit_info = upstream_model.vectorize(test_dataset[0])



print('finish')
