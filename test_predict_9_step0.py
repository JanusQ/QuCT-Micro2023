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

topology = gen_grid_topology(3) # 3x3 9 qubits
neigh_info = get_grid_neighbor_info(3, 1)
coupling_map = topology_to_coupling_map(topology)
n_qubits = 9

with open("dataset_split.pkl", "rb") as f:
    train_dataset, test_dataset = pickle.load(f)

backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = default_basis_single_gates,
                  basis_two_gates = default_basis_two_gates, divide = False, decoupling=False)
upstream_model = RandomwalkModel(4, 20, backend = backend) ### step
upstream_model.train(train_dataset, multi_process = True)

with open("upstream_model_9_step4.pkl", "wb") as f: ### step
    pickle.dump(upstream_model, f)


    
downstream_model = FidelityModel(upstream_model)
downstream_model.train(upstream_model.dataset)

with open("downstream_model_9_step4.pkl", "wb") as f:### step
    pickle.dump(downstream_model, f)

for idx,cir in enumerate(test_dataset):
    cir = upstream_model.vectorize(cir)
    if idx % 100 == 0:
        print(idx,"predict finished!")
    predict = downstream_model.predict_fidelity(cir)
    
from plot.plot import plot_duration_fidelity

plot_duration_fidelity(test_dataset,500,0,"duration_fidelity_9_step4.svg")### step
