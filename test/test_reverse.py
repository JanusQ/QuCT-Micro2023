
from circuit.formatter import layered_circuits_to_qiskit
from circuit.parser import get_circuit_duration, qiskit_to_layered_circuits
from circuit.random_circuit import random_circuit
from collections import defaultdict
import copy
from utils.backend import  gen_grid_topology, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle
from utils.backend import default_basis_single_gates, default_basis_two_gates


size = 5
n_qubits = 18
topology = gen_grid_topology(size)  # 3x3 9 qubits
new_topology = defaultdict(list)
for qubit in topology.keys():
    if qubit < n_qubits:
        for ele in topology[qubit]:
            if ele < n_qubits:
                new_topology[qubit].append(ele)
topology =  new_topology      
neighbor_info = copy.deepcopy(topology)
coupling_map = topology_to_coupling_map(topology)     


n_gates = 50
res  = random_circuit(n_qubits, n_gates, two_qubit_prob = 0.5, reverse = True, coupling_map = coupling_map, basis_single_gates = default_basis_single_gates, basis_two_gates = default_basis_two_gates,)


circuit_info = {}

circuit_info = qiskit_to_layered_circuits(res, False, False)

circuit_info['duration'] = get_circuit_duration(circuit_info['layer2gates'], 30, 60)
circuit_info['gate_num'] = len(circuit_info['gates'])


circuit_info['qiskit_circuit'] = res

reverse_circuit_info = copy.deepcopy(circuit_info)

reverse_circuit_info['layer2gates'].reverse()

max_layer = len(reverse_circuit_info['layer2gates'])

reverse_circuit_info['gate2layer'] = [max_layer - layer for layer in reverse_circuit_info['gate2layer'] ] 

print(layered_circuits_to_qiskit(n_qubits,reverse_circuit_info['layer2gates']))
