import pickle
from utils.backend import *
from upstream import RandomwalkModel
from circuit import gen_random_circuits
import numpy as  np


for n_qubits in range(5,130,10):

    topology = gen_washington_topology(n_qubits)
    neigh_info = get_washington_neighbor_info(topology, 1)
    coupling_map = topology_to_coupling_map(topology)

    backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = default_basis_single_gates,
                    basis_two_gates = default_basis_two_gates, divide = False, decoupling=False)

    dataset = gen_random_circuits(min_gate = 10, max_gate = 120, n_circuits = n_qubits // 3 +5, two_qubit_gate_probs=[4, 8],backend = backend)
    
    for step  in range(1,3):
        upstream_model = RandomwalkModel(step, 20, backend = backend, travel_directions=('parallel', 'former'))
        print(len(dataset),"circuit generated")
        upstream_model.train(dataset, multi_process = True, remove_redundancy = True)
        device_size =[]
        for device, path_table in upstream_model.device2path_table.items():
            device_size.append((device,len(path_table)))
        
        with open(f'washington/qubit{n_qubits}_step{step}.pkl', "wb") as f:
            pickle.dump(device_size,f)



