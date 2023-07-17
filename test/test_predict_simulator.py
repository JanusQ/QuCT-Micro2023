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


with open("upstream_model_18.pkl","rb")as f:
    upstream_model = pickle.load(f)
    
# backend = upstream_model.backend

# dataset = []

# dataset += gen_random_circuits(min_gate=20, max_gate=160, n_circuits=2,
#                                    two_qubit_gate_probs=[4, 8], backend=backend, multi_process=True)
# for idx, cir in enumerate(dataset):
#     cir = upstream_model.vectorize(cir)
    
# simulator = NoiseSimulator(backend)
# simulator.get_error_results(dataset, upstream_model, erroneous_pattern = upstream_model.erroneous_pattern, multi_process=True)

# with open("validate_dataset_18.pkl","wb")as f:
#     pickle.dump(dataset,f)
    
with open("validate_dataset_18.pkl","rb")as f:
    dataset = pickle.load(f)


downstream_model = FidelityModel(upstream_model)

with open("error_params_predicts_18.pkl","rb")as f:
    error_params, _ = pickle.load(f)
    
downstream_model.error_params = error_params

predicts = []
for idx, cir in enumerate(dataset):
    cir = upstream_model.vectorize(cir)
    if idx % 100 == 0:
        print(idx, "predict finished!")
    predict = downstream_model.predict_fidelity(cir)
    print(predict, cir['ground_truth_fidelity'])
    

fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = plot_duration_fidelity(fig, axes, dataset)
fig.savefig("duration_fidelity_validate_18.svg")  # step