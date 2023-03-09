import random
import time

import numpy.random

from circuit import gen_random_circuits, label_ground_truth_fidelity
from circuit.formatter import get_layered_instructions, layered_circuits_to_qiskit, layered_instructions_to_circuit, qiskit_to_my_format_circuit
from circuit.parser import divide_layer, dynamic_decoupling, dynamic_decoupling_divide, get_circuit_duration
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle
import numpy as np


n_qubits = 5
topology= {0: [1, 3], 1: [0, 2, 4], 2: [1], 3: [0, 4], 4: [1, 3]}
coupling_map= [[0, 1], [1, 2], [3, 4], [0, 3], [1, 4]]
neigh_info= {0: [1, 3], 1: [0, 2, 4], 2: [1], 3: [0, 4], 4: [1, 3]}

backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = default_basis_single_gates,
                  basis_two_gates = default_basis_two_gates, divide = False, decoupling=False)

dataset = []
dataset += gen_random_circuits(min_gate=10, max_gate=30, n_circuits=4,
                                   two_qubit_gate_probs=[4, 8], backend=backend, multi_process=True)
upstream_model = RandomwalkModel(1, 20, backend = backend, travel_directions=('parallel', 'former'))
upstream_model.train(dataset, multi_process = True)

simulator = NoiseSimulator(backend)

# start2 = time.time()
# erroneous_pattern = simulator.get_error_results(dataset, upstream_model, multi_process=False)
# execute_time2 = time.time() - start2

start = time.time()
erroneous_pattern = simulator.get_error_results(dataset, upstream_model, multi_process=True)
execute_time = time.time() - start



print(execute_time, execute_time2)
