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

size = 3
n_qubits = 4
n_steps = 1

dataset_path = os.path.join('temp_data', f"eval_datasize_{n_qubits}.pkl")

regen = False
if regen:
    topology = gen_grid_topology(size)  # 3x3 9 qubits
    new_topology = defaultdict(list)
    for qubit in topology.keys():
        if qubit < n_qubits:
            for ele in topology[qubit]:
                if ele < n_qubits:
                    new_topology[qubit].append(ele)
    topology = new_topology
    neighbor_info = copy.deepcopy(topology)
    coupling_map = topology_to_coupling_map(topology)
    backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                    basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

    for gen_type in ['cycle', 'random']:
        dataset_cycle = gen_random_circuits(min_gate=20, max_gate=150, n_circuits=5, two_qubit_gate_probs=[
            3, 7], gate_num_step=40, backend=backend, multi_process=True, circuit_type='cycle')
        dataset_random = gen_random_circuits(min_gate=20, max_gate=150, n_circuits=5, two_qubit_gate_probs=[
            3, 7], gate_num_step=40, backend=backend, multi_process=True, circuit_type='random')

    for elm in dataset_cycle:
        elm['label'] = 'train_cycle'

    for elm in dataset_random:
        elm['label'] = 'train_random'

    '''TODO: 门可以少一些'''
    test_dataset_cycle = gen_random_circuits(min_gate=20, max_gate=1500, n_circuits=2, two_qubit_gate_probs=[
        3, 7], gate_num_step=100, backend=backend, multi_process=True, circuit_type='cycle')  # random
    test_dataset_random = gen_random_circuits(min_gate=20, max_gate=1500, n_circuits=2, two_qubit_gate_probs=[
        3, 7], gate_num_step=100, backend=backend, multi_process=True, circuit_type='random')  #

    for elm in test_dataset_cycle:
        elm['label'] = 'test_cycle'

    for elm in test_dataset_random:
        elm['label'] = 'test_random'

    train_dataset = np.array(dataset_cycle + dataset_random)
    test_dataset = np.array(test_dataset_cycle + test_dataset_random)

    test_dataset, validation_dataset = train_test_split(test_dataset, test_size=.2)
    all_dataset = np.concatenate([train_dataset, validation_dataset, test_dataset])

    print('train dataset size = ', len(train_dataset))
    print('test dataset size = ', len(test_dataset))

    upstream_model = RandomwalkModel(
        n_steps, 20, backend=backend, travel_directions=('parallel', 'former'))

    upstream_model.train(all_dataset, multi_process=True,
                        remove_redundancy=n_steps > 1)


    ata_backend = Backend(n_qubits=n_qubits, topology=None, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                        basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

    simulator = NoiseSimulator(ata_backend)
    erroneous_pattern = get_random_erroneous_pattern(
        upstream_model, error_pattern_num_per_device=3)

    simulator.get_error_results(all_dataset, upstream_model,
                                multi_process=True, erroneous_pattern=erroneous_pattern)
    upstream_model.erroneous_pattern = erroneous_pattern


    with open(dataset_path, "wb")as f:
        pickle.dump((upstream_model, train_dataset,
                    validation_dataset, test_dataset), f)
    print('erroneous patterns = ', upstream_model.erroneous_pattern)
else:
    with open(dataset_path, "rb")as f:
        upstream_model, train_dataset, validation_dataset, test_dataset = pickle.load(f)
    
def sumarize_datasize(dataset, name):
    data = []
    labels = []
    for circuit_info in dataset:
        data.append([circuit_info['n_erroneous_patterns'], len(circuit_info['gates']),
                     circuit_info['n_erroneous_patterns'] /
                     len(circuit_info['gates']
                         ), circuit_info['two_qubit_prob'],
                     circuit_info['ground_truth_fidelity'], circuit_info['independent_fidelity'], circuit_info[
            'ground_truth_fidelity'] - circuit_info['independent_fidelity'],
        ])
        labels.append(circuit_info['label'])

    random.shuffle(data)
    data = data[:3000]  # 太大了画出来的图太密了
    data = np.array(data)
    plot_correlation(data, ['n_erroneous_patterns',
                            'n_gates', 'error_prop', 'two_qubit_prob', 'ground_truth_fidelity', 'independent_fidelity', 
                            'ground_truth_fidelity - independent_fidelity'], color_features=labels, name=name)


sumarize_datasize(train_dataset, f'train_dataset_{n_qubits}')
sumarize_datasize(test_dataset, f'test_dataset_{n_qubits}')


def eval(train_dataset, start, end):
    print(start, end, len(test_dataset))
    
    downstream_model = FidelityModel(upstream_model)
    downstream_model.train(
        train_dataset, validation_dataset, epoch_num=200, verbose = False)

    predicts, reals, durations = [], [], []
    for idx, cir in enumerate(test_dataset):
        cir = upstream_model.vectorize(cir)
        predict = downstream_model.predict_fidelity(cir)
        predicts.append(predict)
        reals.append(cir['ground_truth_fidelity'])
        durations.append(cir['duration'])

    reals = np.array(reals)
    predicts = np.array(predicts)
    durations = np.array(durations)

    additional_info = f'{n_qubits}_{start}_{end}_{len(train_dataset)}'
    print(additional_info, np.array(delta).mean())

    with open(f"temp_data/fidelitymodel_{additional_info}.pkl", "wb")as f:
        pickle.dump((downstream_model, predicts, reals, durations,
                    train_dataset, validation_dataset, test_dataset), f)

    # 画找到path的数量
    find_error_path(
        upstream_model, downstream_model.error_params['gate_params'])

    fig, axes = plt.subplots(figsize=(20, 6))
    duration_X, duration2circuit_index = plot_duration_fidelity(
        fig, axes, test_dataset)
    fig.savefig(f"temp_data/duration_fidelity_{additional_info}.svg")  # step
    plt.close(fig)

    # 画x: real fidelity, y: predicted fidelity
    fig, axes = plt.subplots(figsize=(10, 10))
    plot_real_predicted_fidelity(fig, axes, test_dataset)
    fig.savefig(f"temp_data/real_predictedy_{additional_info}.svg")  # step

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
    fig.savefig(f"temp_data/duration_inaccuracy_{additional_info}.svg")


@ray.remote
def eval_remote(dataset, start, end):
    return eval(dataset, start, end)

# TODO: for 不同的数据量
maximum_datasize = 2000

n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(
    train_dataset)

futures = []
for start in range(20, 150):
    for end in range(start + 150, 1500, 50):
        avl_gate_nums = [
            gate_num
            for gate_num in gate_nums
            if gate_num <= end and gate_num > start
        ]

        aval_n_instruction2circuit_infos = {
            gate_num: list(n_instruction2circuit_infos[gate_num])
            for gate_num in avl_gate_nums
        }

        _dataset = []

        while len(_dataset) < maximum_datasize:
            for gate_num in avl_gate_nums:
                if len(aval_n_instruction2circuit_infos[gate_num]) != 0:
                    circuit_info = random.choice(
                        aval_n_instruction2circuit_infos[gate_num])
                    aval_n_instruction2circuit_infos[gate_num].remove(
                        circuit_info)
                    _dataset.append(circuit_info)

                    if len(_dataset) > maximum_datasize:
                        break
                else:
                    aval_n_instruction2circuit_infos.pop(gate_num)

            avl_gate_nums = list(aval_n_instruction2circuit_infos.keys())
            if len(avl_gate_nums) == 0:
                break

        # futures.append(eval(_dataset, start, end))
        futures.append(eval_remote.remote(_dataset, start, end))

        # if len(futures) > 10:
        #     print()

wait(futures)
