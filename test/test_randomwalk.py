from circuit import gen_random_circuits
from upstream.randomwalk_model import RandomwalkModel
from upstream.randomwalk_model import extract_device
from downstream import FidelityModel
from collections import defaultdict

import math
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates

# topological information
n_qubits = 2
topology = gen_grid_topology(n_qubits)  # 3x3 9 qubits
neigh_info = get_grid_neighbor_info(n_qubits, 1)

print(topology)
print(neigh_info)

n_qubits = n_qubits ** 2
backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info,
                  basis_single_gates=default_basis_single_gates,
                  basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

train_dataset = gen_random_circuits(min_gate=10, max_gate=100, n_circuits=5, two_qubit_gate_probs=[4, 8],
                                    backend=backend, multi_process=True)


print(train_dataset[0]['layer2gates'])

upstream_model = RandomwalkModel(4, 200, backend=backend)
upstream_model.train(train_dataset, multi_process=True)

# 2
# random walk finish device size =  8
# 0 path table size =  738
# 1 path table size =  734
# 2 path table size =  741
# 3 path table size =  737
# (1, 3) path table size =  971
# (0, 1) path table size =  952
# (0, 2) path table size =  973
# (2, 3) path table size =  976

# 3
# random walk finish device size =  8
# 0 path table size =  7376
# 1 path table size =  7472
# 2 path table size =  7420
# 3 path table size =  7186
# (1, 3) path table size =  7096
# (0, 1) path table size =  7062
# (0, 2) path table size =  7460
# (2, 3) path table size =  7096

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
