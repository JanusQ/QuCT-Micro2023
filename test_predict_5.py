import random
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


# n_qubits = 5
# topology= {0: [1, 3], 1: [0, 2, 4], 2: [1], 3: [0, 4], 4: [1, 3]}
# coupling_map= [[0, 1], [1, 2], [3, 4], [0, 3], [1, 4]]
# neigh_info= {0: [1, 3], 1: [0, 2, 4], 2: [1], 3: [0, 4], 4: [1, 3]}

with open("5qubit_data/dataset_split.pkl", "rb") as f:
    train_dataset, test_dataset = pickle.load(f)

# backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = default_basis_single_gates,
#                   basis_two_gates = default_basis_two_gates, divide = False, decoupling=False)
# upstream_model = RandomwalkModel(2, 20, backend = backend, travel_directions=('parallel', 'former')) ### step
# upstream_model.train(train_dataset, multi_process = True)


# with open("upstream_model_5_step2.pkl", "wb") as f: ### step
#     pickle.dump(upstream_model, f)


    
# downstream_model = FidelityModel(upstream_model)
# downstream_model.train(upstream_model.dataset)

# with open("downstream_model_5_step2.pkl", "wb") as f:### step
#     pickle.dump(downstream_model, f)

with open("downstream_model_5_step2.pkl", 'rb') as  f:
    downstream_model = pickle.load(f)
    
upstream_model = downstream_model.upstream_model
for idx,cir in enumerate(test_dataset):
    cir = upstream_model.vectorize(cir)
    if idx % 100 == 0:
        print(idx,"predict finished!")
    predict = downstream_model.predict_fidelity(cir)
    
from plot.plot import plot_duration_fidelity

import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(20,6)) # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index  = plot_duration_fidelity(fig, axes,test_dataset,1000,18000)
fig.savefig("duration_fidelity_step2.svg") ### step
