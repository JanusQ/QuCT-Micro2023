# Import general libraries (needed for functions)
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# Import the RB Functions
import qiskit.ignis.verification.randomized_benchmarking as rb
from simulator import NoiseSimulator, get_random_erroneous_pattern
# Import Qiskit classes 
import qiskit
from utils.backend import devide_chip, gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from collections import defaultdict
from utils.backend import default_basis_single_gates, default_basis_two_gates
import copy
import os
from upstream import RandomwalkModel

# Generate RB circuits (2Q RB)

# number of qubits
nQ = 2 
rb_opts = {}
#Number of Cliffords in the sequence
rb_opts['length_vector'] = [1, 10, 20, 50, 75, 100, 125, 150, 175, 200]
# Number of seeds (random sequences)
rb_opts['nseeds'] = 5
# Default pattern
rb_opts['rb_pattern'] = [[0, 1]]

rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)


size = 3
n_qubits = 5
n_steps = 1

topology = gen_grid_topology(size)  # 3x3 9 qubits
new_topology = defaultdict(list)
for qubit in topology.keys():
    if qubit < n_qubits:
        for ele in topology[qubit]:
            if ele < n_qubits:
                new_topology[qubit].append(ele)
topology = new_topology
neighbor_info = copy.deepcopy(topology)
coupling_map = topology_to_coupling_map(topology)

backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                  basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)


dir_size = 'temp_data'
dataset_path = os.path.join(dir_size, f"dataset_{n_qubits}.pkl")
upstream_model_path = os.path.join(dir_size, f"upstream_model_{n_qubits}.pkl")

with open(upstream_model_path, "rb")as f:
    upstream_model: RandomwalkModel = pickle.load(f)

simulator = NoiseSimulator(backend)
erroneous_pattern = get_random_erroneous_pattern(upstream_model, error_pattern_num_per_device=3)