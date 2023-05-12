import os
# import cloudpickle as pickle
from openpyxl import Workbook
from downstream.fidelity_predict.evaluate_tools import plot_top_ratio
from downstream.fidelity_predict.fidelity_analysis import get_n_instruction2circuit_infos
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
import ray

from utils.ray_func import wait


def eval(filepath, filename):
    additional_info = [elm.replace('.pkl', '') for elm in filename.split('_')[1:]]
    
    n_qubits = int(additional_info[0])
    start = int(additional_info[1])
    end = int(additional_info[2])
    n_data = int(additional_info[3])    
    if len(additional_info) == 4:
        gen_type = 'rand'
    else:
        gen_type = additional_info[4]
    
    additional_info = '_'.join(additional_info)
    # print('draw', additional_info)
    # return [n_qubits, start, end, n_data, 0 ,gen_type]
    with open(filepath, "rb")as f:
        downstream_model, predicts, reals, durations, train_dataset, validation_dataset, test_dataset = pickle.load(f)
        upstream_model = downstream_model.upstream_model

    # 画找到path的数量
    find_error_path(
        upstream_model, downstream_model.error_params['gate_params'], name = f"temp_data/find_ratio_{additional_info}.png")

    fig, axes = plt.subplots(figsize=(20, 6))
    duration_X, duration2circuit_index = plot_duration_fidelity(
        fig, axes, test_dataset)
    fig.savefig(f"temp_data/duration_fidelity_{additional_info}.png")  # step
    plt.close(fig)

    # 画x: real fidelity, y: predicted fidelity
    fig, axes = plt.subplots(figsize=(10, 10))
    plot_real_predicted_fidelity(fig, axes, test_dataset)
    fig.savefig(f"temp_data/real_predictedy_{additional_info}.png")  # step

    # 画x: real fidelity, y: inaccuracy
    fig, axes = plt.subplots(figsize=(20, 6))
    duration_X, duration2circuit_index = get_duration2circuit_infos(
        durations, 100, 0)

    delta = []
    for circuit_index in duration2circuit_index:
        delta.append(np.abs(reals[circuit_index] -
                    predicts[circuit_index]).mean())

    axes.plot(duration_X, delta, markersize=12,
            linewidth=2, label='delta', marker='^')
    axes.set_xlabel('duration')
    axes.set_ylabel('fidelity')
    axes.legend()  # 添加图例
    fig.savefig(f"temp_data/duration_inaccuracy_{additional_info}.png")
    
    print('draw', additional_info)
    
    return [n_qubits, start, end, n_data, np.abs(predicts-reals).mean() ,gen_type]
            

@ray.remote
def eval_remote(filepath, filename):
    return eval(filepath, filename)

futures = []
for root, directories, files in os.walk('temp_data'):
    for filename in files:
        filepath = os.path.join(root, filename)
        if 'fidelitymodel_' not in filepath:
            continue
        print(filepath)
        # futures.append(eval(filepath, filename))
        futures.append(eval_remote.remote(filepath, filename))
        # break
        
futures = wait(futures) # , show_progress = True
data = [
    elm[:-1]
    for elm in futures
]
labels = [
    elm[-1]
    for elm in futures
]
plot_correlation(data, ['n_qubits', 'start', 'end', 'n_data', 'inaccuracy'], color_features=labels, name='eval_data')


