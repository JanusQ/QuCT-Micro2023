from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.parser import get_circuit_duration
from circuit.utils import make_circuitlet, stitching
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path
import random
import numpy as np

from circuit import gen_random_circuits, label_ground_truth_fidelity
from simulator.noise_simulator import get_random_erroneous_pattern
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator
from utils.backend import devide_chip, gen_grid_topology, get_devide_qubit, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle

size = 6
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



backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                  basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)
max_cut_type = 0

dataset = gen_random_circuits(min_gate=20, max_gate=170, n_circuits=20, two_qubit_gate_probs=[
                                            1, 5], gate_num_step=10, backend=backend, multi_process=True,circuit_type='random')


upstream_model = RandomwalkModel(1, 20, backend=backend, travel_directions=('parallel', 'former'))
print(len(dataset), "circuit generated")
upstream_model.train(dataset, multi_process=True)

with open(f"upstream_model_{max_cut_type}.pkl","wb")as f:
    pickle.dump(upstream_model,f)



simulator = NoiseSimulator(backend)
erroneous_pattern = get_random_erroneous_pattern(upstream_model)
erroneous_pattern = simulator.get_error_results(dataset, upstream_model, erroneous_pattern, multi_process=True)
upstream_model.erroneous_pattern = erroneous_pattern

with open(f"upstream_model_{max_cut_type}.pkl","wb")as f:
    pickle.dump(upstream_model,f)


random.shuffle(dataset)

train_dataset, test_dataset  = dataset[:1000], dataset[-1000:]
with open(f"train_test_{max_cut_type}.pkl","wb")as f:
    pickle.dump((train_dataset, test_dataset),f)
    
downstream_model = FidelityModel(upstream_model)
downstream_model.train(train_dataset)

predicts, reals, durations = [], [], []
for idx, cir in enumerate(test_dataset):
    if idx % 100 == 0:
        print(idx, "predict finished!")
    predict = downstream_model.predict_fidelity(cir)

    predicts.append(cir['circuit_predict'])
    reals.append(cir['ground_truth_fidelity'])
    durations.append(cir['duration'])
print(np.abs(np.array(predicts) - np.array(reals)).mean())

with open(f"res_{max_cut_type}.pkl","wb")as f:
    pickle.dump((downstream_model, test_dataset, predicts, reals, durations), f)
    
