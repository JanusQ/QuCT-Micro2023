import copy
import pickle
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from circuit.utils import make_circuitlet
from downstream import FidelityModel
from downstream.fidelity_predict.baseline.rb import get_errors
from circuit.generate_dataset import gen_cut_dataset
from simulator import NoiseSimulator, get_random_erroneous_pattern
from upstream import RandomwalkModel
from upstream.randomwalk_model import extract_device
from utils.backend import default_basis_single_gates, default_basis_two_gates
from utils.backend import gen_grid_topology, Backend, topology_to_coupling_map

n_steps = 1
qubit_list = [i for i in range(50, 400, 50)]
size_list = [8, 10, 13, 15, 16, 18, 19]
# qubit_list = [i for i in range(5, 7)]
# size_list = [3] * 2

err, qubit_num, method = [], [], []
for size, n_qubits in zip(size_list, qubit_list):

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

    backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info,
                      basis_single_gates=default_basis_single_gates, basis_two_gates=default_basis_two_gates,
                      divide=False, decoupling=False)

    dataset = gen_cut_dataset(n_qubits, topology, neighbor_info, coupling_map, dataset_size=5000, min_cut_qubit=3,
                              devide_size=5)

    test_dataset = gen_cut_dataset(n_qubits, topology, neighbor_info, coupling_map, dataset_size=1000, min_cut_qubit=3,
                                   devide_size=5)

    print('train dataset size = ', len(dataset))
    print('test dataset size = ', len(test_dataset))

    upstream_model = RandomwalkModel(n_steps, 20, backend=backend, travel_directions=('parallel', 'former'))

    upstream_model.train(dataset + test_dataset, multi_process=True, remove_redundancy=n_steps > 1)

    dataset_cut = make_circuitlet(dataset)
    print("cutted", len(dataset_cut))

    test_dataset_cut = make_circuitlet(test_dataset)
    print("cutted", len(test_dataset_cut))

    all_to_all_backend = Backend(n_qubits=5, topology=None, neighbor_info=neighbor_info,
                                 basis_single_gates=default_basis_single_gates, basis_two_gates=default_basis_two_gates,
                                 divide=False, decoupling=False)

    simulator = NoiseSimulator(all_to_all_backend)
    erroneous_pattern = get_random_erroneous_pattern(upstream_model, error_pattern_num_per_device=3)

    # 每个subcircuit是单独的NoiseSimulator，backend对应不对
    erroneous_pattern = simulator.get_error_results(dataset_cut + test_dataset_cut, upstream_model, multi_process=True,
        erroneous_pattern=erroneous_pattern)
    upstream_model.erroneous_pattern = erroneous_pattern

    with open(f"upstream_model_{n_qubits}qubit.pkl", "wb") as f:
        pickle.dump(upstream_model, f)

    train_dataset, validation_dataset = train_test_split(dataset_cut, test_size=.2)

    test_dataset = np.array(test_dataset_cut)

    with open(f"dataset_{n_qubits}qubit.pkl", "wb") as f:
        pickle.dump((train_dataset, validation_dataset, test_dataset), f)

    total_dataset = np.concatenate([train_dataset, validation_dataset, test_dataset])

    downstream_model = FidelityModel(upstream_model)
    downstream_model.train(train_dataset, validation_dataset, epoch_num=50)

    predicts, reals, durations = [], [], []
    for idx, cir in enumerate(test_dataset):
        cir = upstream_model.vectorize(cir)
        if idx % 100 == 0:
            print(idx, "predict finished!")
        predict = downstream_model.predict_fidelity(cir)

        predicts.append(predict)
        reals.append(cir['ground_truth_fidelity'])
        durations.append(cir['duration'])

    reals = np.array(reals)
    predicts = np.array(predicts)
    durations = np.array(durations)

    #     # print(predict)
    with open(f"error_params_predicts_{n_qubits}qubit.pkl", "wb") as f:
        pickle.dump((downstream_model, predicts, reals, durations, test_dataset), f)

    print('average inaccuracy = ', np.abs(predicts - reals).mean())

    all_errors = get_errors(backend, simulator, upstream_model=None, multi_process=True)
    print(all_errors)
    single_average_error = {}
    couple_average_error = {}
    for q, e in enumerate(all_errors[0]):
        single_average_error[q] = e
    for c, e in zip(list(backend.coupling_map), all_errors[1]):
        couple_average_error[tuple(c)] = e

    print(single_average_error, couple_average_error)
    with open(f"rb_error_{n_qubits}qubit.pkl", "wb") as f:
        pickle.dump((single_average_error, couple_average_error), f)


    def get_rb_fidelity(circuit_info):
        fidelity = 1
        for gate in circuit_info['gates']:
            device = extract_device(gate)
            if isinstance(device, tuple):
                device = (circuit_info['map'][device[0]], circuit_info['map'][device[1]])
                fidelity = fidelity * (1 - couple_average_error[device])
            else:
                device = circuit_info['map'][device]
                fidelity = fidelity * (1 - single_average_error[device])
        # * np.product((measure0_fidelity + measure1_fidelity) / 2)
        return fidelity


    rbs = []
    for cir in test_dataset:
        rbs.append(get_rb_fidelity(cir))

    rbs = np.array(rbs)

    print('average inaccuracy = ', np.abs(rbs - reals).mean())

    err += np.abs(predicts - reals).tolist()
    qubit_num += [n_qubits] * len(reals)
    method += ['quct'] * len(reals)

    err += np.abs(rbs - reals).tolist()
    qubit_num += [n_qubits] * len(reals)
    method += ['rb'] * len(reals)

import seaborn as sns
import pandas as pd

df = pd.DataFrame({'inc': err, 'n_qubit': qubit_num, 'type': method})

fig = sns.boxplot(df, x='n_qubit', y='inc', hue='type').get_figure()
fig.savefig('simulate50-350.svg')
