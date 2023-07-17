
from upstream import RandomwalkModel

from utils.backend import  Backend, topology_to_coupling_map, gen_washington_topology, get_washington_neighbor_info
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle
import numpy as np 
import copy
from circuit import gen_random_circuits


n_qubits = 127
topology = gen_washington_topology(n_qubits)  # 3x3 9 qubits
neighbor_info = get_washington_neighbor_info(topology, 1)
coupling_map = topology_to_coupling_map(topology)

# n_qubits = 7
# topology = {0: [1], 1: [0, 2, 3], 2: [1], 3: [1, 5], 4: [5], 5: [3, 4, 6], 6: [5]}
# neighbor_info = copy.deepcopy(topology)
# coupling_map = topology_to_coupling_map(topology)

backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                  basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

dataset = gen_random_circuits(min_gate=n_qubits *10, max_gate=n_qubits*20, n_circuits=40, two_qubit_gate_probs=[
                                  2, 5], gate_num_step=10, backend=backend, multi_process=True)



upstream_model = RandomwalkModel(1, 20, backend=backend, travel_directions=('parallel, former'))
upstream_model.train(dataset, multi_process=True, process_num = 20, remove_redundancy=False)
upstream_model.max_table_size

size = []
for table in upstream_model.device2path_table.values():
    size.append(len(table))
# np.array(size).mean(),np.array(size).min(),np.array(size).max(),
print(n_qubits,'step1', np.array(size).mean())

upstream_model = RandomwalkModel(2, 20, backend=backend, travel_directions=('parallel, former'))
upstream_model.train(dataset, multi_process=True, process_num = 20, remove_redundancy=False)
upstream_model.max_table_size

size = []
for table in upstream_model.device2path_table.values():
    size.append(len(table))
# np.array(size).mean(),np.array(size).min(),np.array(size).max(),
print(n_qubits,'step2', np.array(size).mean())