from openpyxl import Workbook
from downstream.fidelity_predict.evaluate_tools import plot_top_ratio
from plot.plot import get_duration2circuit_infos
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
from circuit.parser import get_circuit_duration
from circuit.utils import make_circuitlet
from generate_dataset import gen_train_dataset
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path, plot_correlation, plot_real_predicted_fidelity
import random
import numpy as np

from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import NoiseSimulator, get_random_erroneous_pattern
from utils.backend import devide_chip, gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from utils.backend import default_basis_single_gates, default_basis_two_gates
import pickle
import os
from sklearn.model_selection import train_test_split

n_qubits = 18
with open(f"execute_18bit/split_dataset_execute_18bits_train_0_2500.pkl","rb")as f:
    train_dataset, test_dataset = pickle.load(f)
    
with open(f"error_params_predicts_execute_18bits_train_0_2500_step0.pkl", "rb")as f:
    downstream_model, predicts, reals, durations, test_dataset = pickle.load(f)
    
    
upstream_model = downstream_model.upstream_model
   
   
# for idx, cir in enumerate(test_dataset):
#     cir = upstream_model.vectorize(cir)
#     predict = downstream_model.predict_fidelity(cir)
#     if idx % 100 == 0:
#         print(idx, "predict finished!")   

predicts = np.array(predicts)
reals = np.array(reals)
print('average inaccuracy = ', np.abs(predicts - reals).mean())

# 画找到path的数量
# find_error_path(upstream_model, downstream_model.error_params['gate_params'])

fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = plot_duration_fidelity(
    fig, axes, test_dataset)
fig.savefig(f"duration_fidelity_{n_qubits}__step0.svg")  # step
plt.close(fig)

# 画x: real fidelity, y: predicted fidelity
fig, axes = plt.subplots(figsize=(10, 10))  # 创建一个图形对象和一个子图对象
plot_real_predicted_fidelity(fig, axes, test_dataset)
fig.savefig(f"real_predictedy_{n_qubits}__step0.svg")  # step

# 画x: real fidelity, y: inaccuracy
fig, axes = plt.subplots(figsize=(20, 6))  # 创建一个图形对象和一个子图对象
duration_X, duration2circuit_index = get_duration2circuit_infos(
    durations, 100, 0)

delta = []
for circuit_index in duration2circuit_index:
    delta.append(np.abs(reals[circuit_index] - predicts[circuit_index]).mean())

axes.plot(duration_X, delta, markersize=12, linewidth=2, label='delta', marker='^')
axes.set_xlabel('duration')
axes.set_ylabel('fidelity')
axes.legend()  # 添加图例
fig.savefig(f"inaccuracy_fidelity_{n_qubits}__step0.svg")
print(np.array(delta).mean())


# 存path error 到 excel
error_params = downstream_model.error_params['gate_params']
device_index2device = {}
for device  in upstream_model.device2path_table.keys():
    device_index = list(upstream_model.device2path_table.keys()).index(device)
    device_index2device[device_index] = device

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
            if isinstance(path, str):
                device_error_params_path_weight.append(
                    (path, float(device_error_param[i]), downstream_model.path_count[path]))
                device_error_params_path.append(path)
    error_params_path_weight[device] = device_error_params_path_weight
    error_params_path[device] = device_error_params_path

wb = Workbook()

for device, device_error_params_path_weight in error_params_path_weight.items():
    ws = wb.create_sheet(str(device))
    for row in device_error_params_path_weight:
        row = list(row)
        # if row[0] in upstream_model.erroneous_pattern[device]:
        #     row.append("true")
        # else:
        #     row.append("false")
        ws.append(row)

# Save the file
wb.save("sample_18qubits_step0.xlsx")