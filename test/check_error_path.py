from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.parser import get_circuit_duration
from circuit.utils import make_circuitlet
from generate_dataset import gen_train_dataset
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path
import random
import numpy as np

from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator
from utils.backend import devide_chip, gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle

  
size = 3
n_qubits = 8
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



backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                  basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

dataset = gen_train_dataset(n_qubits, topology, neighbor_info, coupling_map, dataset_size = 500, devide_size= 4)

upstream_model = RandomwalkModel(1, 20, backend=backend, travel_directions=('parallel', 'former'))
print(len(dataset), "circuit generated")
upstream_model.train(dataset, multi_process=True)



print("original",len(dataset))
dataset = make_circuitlet(dataset)
print("cutted",len(dataset))


simulator = NoiseSimulator(backend)
erroneous_pattern = simulator.get_error_results(dataset, upstream_model, multi_process=True)
upstream_model.erroneous_pattern = erroneous_pattern

for idx, cir in enumerate(dataset):
    print(cir['n_erroneous_patterns'],cir['n_erroneous_patterns']/ len(cir['gates']), len(cir['gates']))