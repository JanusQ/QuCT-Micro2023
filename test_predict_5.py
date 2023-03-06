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


n_qubits = 5
topology= {0: [1, 3], 1: [0, 2, 4], 2: [1], 3: [0, 4], 4: [1, 3]}
coupling_map= [[0, 1], [1, 2], [3, 4], [0, 3], [1, 4]]
neigh_info= {0: [1, 3], 1: [0, 2, 4], 2: [1], 3: [0, 4], 4: [1, 3]}

with open("5qubit_data/dataset_split.pkl", "rb") as f:
    train_dataset, test_dataset = pickle.load(f)
    
# print('Finish loading')

# def simplify(dataset):
#     for circuit_info in dataset:
#         circuit_info['qiskit_circuit'] = None
#         circuit_info['gate_paths'] = None
#         circuit_info['path_indexs'] = None
#         circuit_info['vecs'] = None

# simplify(train_dataset), simplify(test_dataset)
# with open("5qubit_data/dataset_split_simp.pkl", "wb") as f:
#     pickle.dump((train_dataset, test_dataset), f)

backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = default_basis_single_gates,
                  basis_two_gates = default_basis_two_gates, divide = False, decoupling=False)

# with open("5qubit_data/dataset_split_simp.pkl", "rb") as f:
#     train_dataset, test_dataset = pickle.load(f)
    
# print('Finish loading')

# upstream_model: RandomwalkModel = RandomwalkModel(1, 20, backend = backend) ### step
# upstream_model.train(list(train_dataset) + list(test_dataset), multi_process = True, remove_redundancy = False, process_num = 20)

# with open("upstream_model_5_step3.pkl", "wb") as f:
#     pickle.dump(upstream_model, f)

# with open("upstream_model_5_step3.pkl", "rb") as f:
#     upstream_model = pickle.load(f)

# print(len(train_dataset))
# train_dataset_size = len(train_dataset)
# train_dataset = upstream_model.dataset[:len(train_dataset)]
# test_dataset = upstream_model.dataset[len(train_dataset):]

# downstream_model = FidelityModel(upstream_model)
# downstream_model.train(train_dataset)

# with open("downstream_model_5_step1.pkl", "wb") as f:### step
#     pickle.dump(downstream_model, f)

with open("downstream_model_5_step1.pkl", "rb") as f:### step
    downstream_model: FidelityModel = pickle.load(f)

upstream_model: RandomwalkModel = downstream_model.upstream_model

for idx,cir in enumerate(test_dataset):
    cir = upstream_model.vectorize(cir)
    if idx % 100 == 0:
        print(idx,"predict finished!")
    predict = downstream_model.predict_fidelity(cir)
    
from plot.plot import plot_duration_fidelity

fig, axes, duration_X, duration2circuit_index  = plot_duration_fidelity(test_dataset,500,0) 
fig.savefig("duration_fidelity_step1.svg") ### step
