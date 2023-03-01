import random
import numpy as np

from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle 

topology = gen_grid_topology(3) # 3x3 9 qubits
neigh_info = get_grid_neighbor_info(3, 1)
coupling_map = topology_to_coupling_map(topology)
n_qubits = 9

backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates = default_basis_single_gates,
                  basis_two_gates = default_basis_two_gates, divide = False, decoupling=False)
upstream_model = RandomwalkModel(1, 20, backend = backend)

dataset = gen_random_circuits(min_gate = 10, max_gate = 150, n_circuits = 10, two_qubit_gate_probs=[4, 8],backend = backend)
print(len(dataset),"circuit generated")
upstream_model.train(dataset, multi_process = True)


simulator = NoiseSimulator(backend)
erroneous_pattern = simulator.get_error_results(dataset, upstream_model)
# label_ground_truth_fidelity(train_dataset,numpy.random.rand(64))

upstream_model.erroneous_pattern = erroneous_pattern
import pickle
with open("upstream_model_9.pkl","wb") as f:
     pickle.dump(upstream_model, f)


import pickle

def load_upstream_model() -> RandomwalkModel:
    with open("upstream_model_9.pkl", "rb") as f:
        upstream_model = pickle.load(f)
    return upstream_model

# upstream_model = load_upstream_model()

backend = upstream_model.backend
dataset = upstream_model.dataset

index = np.arange(len(dataset))
random.shuffle(index)
train_index, test_index = index[:-1500],index[-1500:]
train_dataset, test_dataset =  np.array(dataset)[train_index], np.array(dataset)[test_index]
with open("split_index_9.pkl", "wb") as f:
    pickle.dump((train_dataset, test_dataset), f)
    
    
downstream_model = FidelityModel(upstream_model)
downstream_model.train(upstream_model.dataset, )

with open("downstream_model_9.pkl", "wb") as f:
    pickle.dump(downstream_model, f)

# with open("downstream_model_9.pkl", "rb") as f:
#     downstream_model = pickle.load(f)

for idx, cir in enumerate(test_dataset):
    # cir = upstream_model.vectorize(cir)
    if idx % 100 == 0:
        print(idx," predict finished!")
    predict = downstream_model.predict_fidelity(cir)
    # print(predict)

from plot.plot import plot_duration_fidelity

plot_duration_fidelity(test_dataset,500,0)