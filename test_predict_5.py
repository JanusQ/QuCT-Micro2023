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
topology= {0: [1], 1: [0, 2], 2: [1, 3], 3: [4, 2], 4: [3]}
coupling_map = [[0, 1], [1, 2], [2, 3], [3, 5]]
neigh_info= topology
# (2, 3)
# with open("5qubit_data/dataset_split.pkl", "rb") as f:
#     train_dataset, test_dataset = pickle.load(f)
    
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
<<<<<<< HEAD
upstream_model = RandomwalkModel(1, 20, backend = backend) ### step
upstream_model.train(train_dataset, multi_process = True)

with open("upstream_model_5_step3.pkl", "wb") as f: ### step
    pickle.dump(upstream_model, f)


=======

with open("5qubit_data/dataset_split_simp.pkl", "rb") as f:
    train_dataset, test_dataset = pickle.load(f)
print(len(train_dataset))

print('Finish loading')

upstream_model: RandomwalkModel = RandomwalkModel(1, 20, backend = backend, travel_directions=('parallel', 'former')) ### step
upstream_model.train(list(train_dataset) + list(test_dataset), multi_process = True, remove_redundancy = False, process_num = 100)

# with open("upstream_model_5_step3.pkl", "wb") as f:
#     pickle.dump(upstream_model, f)

# with open("upstream_model_5_step3.pkl", "rb") as f:
#     upstream_model = pickle.load(f)


train_dataset_size = len(train_dataset) # 15000
train_dataset = upstream_model.dataset[:len(train_dataset)]
test_dataset = upstream_model.dataset[len(train_dataset):]
upstream_model.dataset = None

>>>>>>> 60f9be79e6181cbdb6bc2eaa29e2da0cb2029f24
downstream_model = FidelityModel(upstream_model)
downstream_model.train(train_dataset, test_dataset = None)

with open("downstream_model_5_step1.pkl", "wb") as f:### step
    pickle.dump(downstream_model, f)


error_params = downstream_model.error_params
upstream_model = downstream_model.upstream_model
device_index2device = {}
for device  in upstream_model.device2path_table.keys():
    device_index = list(upstream_model.device2path_table.keys()).index(device)
    device_index2device[device_index] = device
device_index2device
import numpy as np 

error_params_path_weight = {}
error_params_path = {}
for idx, device_error_param in enumerate(error_params):
    device = device_index2device[idx]
    sort = np.argsort(device_error_param)
    sort = sort[::-1]
    device_error_params_path_weight = []
    device_error_params_path = []
    for i in sort:
        if int(i) in upstream_model.device2reverse_path_table[device].keys():
            path = upstream_model.device2reverse_path_table[device][int(i)]
            if isinstance(path,str):
                device_error_params_path_weight.append((path, float(device_error_param[i]), downstream_model.path_count[path]))
                device_error_params_path.append(path)
    error_params_path_weight[device] = device_error_params_path_weight
    error_params_path[device] = device_error_params_path

error_params_path_weight

from openpyxl import Workbook
wb = Workbook()


for device, device_error_params_path_weight in  error_params_path_weight.items():
    ws = wb.create_sheet(str(device))
    for row in device_error_params_path_weight:
        ws.append(row)
    
# Save the file
wb.save("sample.xlsx")    
# with open("downstream_model_5_step1.pkl", "rb") as f:### step
#     downstream_model: FidelityModel = pickle.load(f)

# upstream_model: RandomwalkModel = downstream_model.upstream_model

# upstream_model = downstream_model.upstream_model
# for idx,cir in enumerate(test_dataset):
#     cir = upstream_model.vectorize(cir)
#     if idx % 100 == 0:
#         print(idx,"predict finished!")
#     predict = downstream_model.predict_fidelity(cir)
    
# from plot.plot import plot_duration_fidelity

# fig, axes, duration_X, duration2circuit_index  = plot_duration_fidelity(test_dataset, 500, 0, 1000,18000)
# fig.savefig("duration_fidelity_step1.svg") ### step
