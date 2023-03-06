import random
import numpy.random

from circuit import gen_random_circuits, label_ground_truth_fidelity
from circuit.formatter import get_layered_instructions, layered_circuits_to_qiskit, layered_instructions_to_circuit, qiskit_to_my_format_circuit
from circuit.parser import divide_layer, dynamic_decoupling, dynamic_decoupling_divide, get_circuit_duration
from circuit.utils import get_xeb_fidelity
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

# with open("5qubit_data/dataset_split.pkl", "rb") as f:
#     train_dataset, test_dataset = pickle.load(f)

# backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = default_basis_single_gates,
#                   basis_two_gates = default_basis_two_gates, divide = False, decoupling=False)
# upstream_model = RandomwalkModel(0, 20, backend = backend)
# upstream_model.train(train_dataset, multi_process = True)

# with open("upstream_model_step0.pkl", "wb") as f:
#     pickle.dump(upstream_model, f)

with open("5qubit_data/upstream_model_step1.pkl", "rb") as f:
    upstream_model = pickle.load(f)
    
downstream_model = FidelityModel(upstream_model)
downstream_model.train(upstream_model.dataset)

# with open("5qubit_data/downstream_model.pkl", "rb") as f:
#     downstream_model = pickle.load(f)

with open("5qubit_data/downstream_model_step1.pkl", "wb") as f:
    pickle.dump(downstream_model,f)

with open("5qubit_data/dataset_split.pkl", "rb") as f:
    _, test_dataset = pickle.load(f)
upstream_model = downstream_model.upstream_model
lt150_test_dataset = []
for idx,cir in enumerate(test_dataset):
    if len(cir["gates"]) > 150:
        continue
    cir = upstream_model.vectorize(cir)
    if idx % 100 == 0:
        print(idx,"finished!")
    predict = downstream_model.predict_fidelity(cir)
    lt150_test_dataset.append(cir)
    
from plot.plot import plot_duration_fidelity

fig, axes, duration_X, duration2circuit_index = plot_duration_fidelity(lt150_test_dataset,500,0)

xebs = get_xeb_fidelity(lt150_test_dataset)
xebs_y = []
for circuit_index in duration2circuit_index:
        xebs_y.append(xebs[circuit_index].mean())
axes.plot(duration_X, xebs_y ,markersize = 12,linewidth = 2, label='xeb',marker = '^' )
fig.savefig("duration_fidelity_step1_xeb.svg")
