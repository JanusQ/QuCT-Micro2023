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


n_qubits = 50
with open(f"upstream_model_{n_qubits}.pkl","rb")as f:
    step1 = pickle.load(f)


    
dataset = step1.dataset
backend = step1.backend

upstream_model = RandomwalkModel(0, 20, backend=backend, travel_directions=('parallel', 'former'))
upstream_model.train(dataset, multi_process=True)

upstream_model.erroneous_pattern = step1.erroneous_pattern
with open(f"upstream_model_{n_qubits}_step0_error.pkl","wb")as f:
    pickle.dump(upstream_model,f)



with open(f"dataset_6error.pkl","rb")as f:
    train_dataset, test_dataset = pickle.load(f)
# upstream_model = downstream_model.upstream_model
    
for idx, cir in enumerate(train_dataset):
    cir = upstream_model.vectorize(cir)

downstream_model = FidelityModel(upstream_model)
downstream_model.train(train_dataset)

predicts, reals, durations = [], [], []
for idx, cir in enumerate(test_dataset):
    cir = upstream_model.vectorize(cir)
    if idx % 100 == 0:
        print(idx, "predict finished!")
    predict = downstream_model.predict_fidelity(cir)

    predicts.append(cir['circuit_predict'])
    # reals.append(cir['ground_truth_fidelity'])
    durations.append(cir['duration'])
    # print(predict, cir['ground_truth_fidelity'])
    
    
    # print(predict)
with open(f"error_params_predicts_{n_qubits}_step0_error.pkl","wb")as f:
    pickle.dump((downstream_model.error_params, predicts, reals, durations), f)
    
find_error_path(upstream_model, downstream_model.error_params)


fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = plot_duration_fidelity(fig, axes, test_dataset)
fig.savefig(f"duration_fidelity_{n_qubits}_step0_6error.svg")  # step
