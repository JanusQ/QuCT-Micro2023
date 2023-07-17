from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.parser import get_circuit_duration
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
from simulator.noise_simulator import get_random_erroneous_pattern

n_qubits = 50
with open(f"upstream_model_{n_qubits}_6error.pkl", "rb")as f:
    upstream_model = pickle.load(f)

with open(f"split_dataset_{n_qubits}.pkl", "rb")as f:
    train_dataset, test_dataset = pickle.load(f)

backend = upstream_model.backend

simulator = NoiseSimulator(backend)

dataset = list(train_dataset)

erroneous_pattern = upstream_model.erroneous_pattern

simulator.get_error_results(dataset, upstream_model, erroneous_pattern=erroneous_pattern, multi_process=True)


with open(f"dataset_6error.pkl", "wb")as f:
    pickle.dump((dataset,test_dataset), f)
