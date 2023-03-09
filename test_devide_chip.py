from circuit.dataset_loader import gen_random_circuits
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map, devide_chip
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle 



topology = gen_grid_topology(6) # 3x3 9 qubits
neigh_info = get_grid_neighbor_info(6, 1)
coupling_map = topology_to_coupling_map(topology)
n_qubits = 36

backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = default_basis_single_gates,
                  basis_two_gates = default_basis_two_gates, divide = False, decoupling=False)

ret_backend = devide_chip(backend,5)

dataset = gen_random_circuits(min_gate = 120, max_gate = 150, n_circuits = 10, two_qubit_gate_probs=[4, 8],backend = ret_backend,multi_process=False)
print("")