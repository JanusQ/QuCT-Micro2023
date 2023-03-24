from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.parser import get_circuit_duration
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path, plot_real_predicted_fidelity
import random
import numpy as np

from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream.qubit_dependent_model import QubitDependentModel
from downstream import FidelityModel
from simulator import NoiseSimulator
from utils.backend import devide_chip, gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle
import  os 

n_qubits = 5
dir_size = 'temp_data'
dataset_path = os.path.join(dir_size, f"dataset_{n_qubits}.pkl")
upstream_model_path = os.path.join(dir_size, f"upstream_model_{n_qubits}.pkl")

retrain = False
if retrain:
    with open(dataset_path, "rb")as f:
        train_dataset, validation_dataset, test_dataset = pickle.load(f)

    with open(upstream_model_path, "rb")as f:
        step1 = pickle.load(f)

    dataset = step1.dataset
    backend = step1.backend

    upstream_model = QubitDependentModel(0, 20, backend=backend, travel_directions=('parallel', 'former'))
    upstream_model.train(dataset, multi_process=True)


    for idx, cir in enumerate(train_dataset):
        cir = upstream_model.vectorize(cir)

    for idx, cir in enumerate(validation_dataset):
        cir = upstream_model.vectorize(cir)

    for idx, cir in enumerate(test_dataset):
        cir = upstream_model.vectorize(cir)
        
    downstream_model = FidelityModel(upstream_model)
    downstream_model.train(train_dataset, validation_dataset = validation_dataset, epoch_num = 100)

    predicts, reals, durations = [], [], []
    for idx, cir in enumerate(test_dataset):
        cir = upstream_model.vectorize(cir)
        if idx % 100 == 0:
            print(idx, "predict finished!")
        predict = downstream_model.predict_fidelity(cir)

        predicts.append(cir['circuit_predict'])
        reals.append(cir['ground_truth_fidelity'])
        durations.append(cir['duration'])
        # print(predict, cir['ground_truth_fidelity'])
            
    reals = np.array(reals)
    predicts = np.array(predicts)
    durations = np.array(durations)

        # print(predict)
    with open(f"error_params_predicts_{n_qubits}_qubit_dependent_error.pkl","wb")as f:
        pickle.dump((downstream_model, predicts, reals, durations, test_dataset), f)
else:
    with open(f"error_params_predicts_{n_qubits}_qubit_dependent_error.pkl","rb")as f:
        downstream_model, predicts, reals, durations, test_dataset = pickle.load(f)
        # downstream_model.error_params, predicts, reals, durations = pickle.load(f)
    # upstream_model = downstream_model.upstream_model

# 画duration - fidelity
# predicts, reals, durations = [], [], []
# for idx, cir in enumerate(test_dataset):
#     if cir['gate_num'] > 2000:
#         continue
#     predicts.append(cir['circuit_predict'])
#     reals.append(cir['ground_truth_fidelity'])
#     durations.append(cir['duration'])
# reals = np.array(reals)
# predicts = np.array(predicts)
# durations = np.array(durations)

print('average inaccuracy = ', np.abs(predicts - reals).mean())
print('average variance', np.var(np.abs(predicts - reals), ddof = 1) )

fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = plot_duration_fidelity(fig, axes, test_dataset)
fig.savefig(f"duration_fidelity_{n_qubits}_qubit_dependent.svg")  # step

fig, axes = plt.subplots(figsize=(10, 10))  # 创建一个图形对象和一个子图对象
plot_real_predicted_fidelity(fig, axes, test_dataset)
fig.savefig(f"real_predictedy_{n_qubits}_qubit_dependent.svg")  # step

# 画duration - inaccuracy
from plot.plot import get_duration2circuit_infos

fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
# duration_X, duration2circuit_index = get_duration2circuit_infos(durations,100,0)

delta = []
for circuit_index in duration2circuit_index:
    delta.append(np.abs(reals[circuit_index] - predicts[circuit_index]).mean())
    
axes.plot(duration_X, delta ,markersize = 12,linewidth = 2, label='delta',marker = '^' )
axes.set_xlabel('duration ')
axes.set_ylabel('fidelity')
axes.legend() # 添加图例
fig.savefig(f"inaccuracy_fidelity_{n_qubits}_qubit_dependent.svg")
print(np.array(delta).mean())

