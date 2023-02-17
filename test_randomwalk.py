from circuit import gen_random_circuits
from upstream import RandomwalkModel
from upstream.randomwalk_model import extract_device
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

train_dataset = gen_random_circuits(min_gate=10, max_gate=100, n_circuits=1, two_qubit_gate_probs=[4, 8],
                                    backend=backend)

upstream_model = RandomwalkModel(3, 30, backend=backend)
upstream_model.train(train_dataset, multi_process=True)

# upstream_model.load_reduced_vecs()

test_dataset = gen_random_circuits(min_gate=10, max_gate=100, n_circuits=1, two_qubit_gate_probs=[4, 8],
                                    backend=backend)

from circuit.formatter import qiskit_to_my_format_circuit, layered_instructions_to_circuit,  get_layered_instructions, layered_circuits_to_qiskit
for circuit_info in test_dataset:
    circuit_info = upstream_model.vectorize(circuit_info)

    print(circuit_info['qiskit_circuit'])

    for i in range(len(circuit_info['gates'])):
        gate, vec =  circuit_info['gates'][i], circuit_info['vecs'][i]
        print(upstream_model.extract_paths_from_vec(gate, vec))
        
        layer2gates = upstream_model.reconstruct(gate, vec)
        qiskit_circuit = layered_circuits_to_qiskit(n_qubits, layer2gates)
        
        print(layer2gates)
        print(qiskit_circuit)
        print()
    
    print('----------------------------------------------------------------')



print('finish')
