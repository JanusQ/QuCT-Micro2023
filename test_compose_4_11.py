import copy
import pickle
from jax import numpy as jnp
from jax import vmap
import jax
import numpy as np
from numpy import random
import time
from qiskit.quantum_info.analysis import hellinger_fidelity
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import ray
from circuit.dataset_loader import gen_algorithms, gen_random_circuits
from circuit.formatter import layered_circuits_to_qiskit, get_layered_instructions
from downstream.fidelity_predict.fidelity_analysis import smart_predict, error_param_rescale
from simulator.noise_free_simulator import simulate_noise_free
from simulator.noise_simulator import NoiseSimulator
from test_opt import count_error_path_num
from upstream.randomwalk_model import RandomwalkModel, extract_device
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_circuit_layout
from collections import defaultdict
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path, plot_correlation, plot_real_predicted_fidelity
from circuit.parser import get_circuit_duration
from utils.backend import Backend, gen_grid_topology, gen_linear_topology, get_linear_neighbor_info, topology_to_coupling_map
from qiskit.transpiler.preset_passmanagers import (
    level_0_pass_manager,
    level_1_pass_manager,
    level_2_pass_manager,
    level_3_pass_manager,
)
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.compiler.transpiler import _parse_coupling_map, _parse_initial_layout
from utils.ray_func import wait

def my_routing(qiskit_circuit, backend, initial_layout):
    coupling_map = _parse_coupling_map(backend.coupling_map, backend)
    if initial_layout is None:
        initial_layout = list(range(n_qubits))
        random.shuffle(initial_layout)
        initial_layout = _parse_initial_layout(
            initial_layout, [qiskit_circuit])[0]
    else:
        initial_layout = _parse_initial_layout(
            initial_layout, [qiskit_circuit])[0]
    # pass_manager_config = PassManagerConfig(initial_layout=initial_layout, basis_gates=['rx', 'ry', 'rz', 'cz'],
    #                                         routing_method = 'sabre', coupling_map = coupling_map)

    pass_manager_config = PassManagerConfig(initial_layout=initial_layout, basis_gates=['rx', 'ry', 'rz', 'cz'],
                                            routing_method='stochastic', coupling_map=coupling_map)
    pass_manager = level_2_pass_manager(pass_manager_config)
    transpiled_circuit = pass_manager.run(qiskit_circuit)
    return transpiled_circuit, pass_manager.property_set['final_layout']


def opt_trans(circuit_info, downstream_model, max_layer=10):

    n_qubits = circuit_info['num_qubits']

    upstream_model = downstream_model.upstream_model
    backend = upstream_model.backend
    # v2p: qr: int

    init_qc = layered_circuits_to_qiskit(circuit_info['num_qubits'], circuit_info['layer2gates'], barrier=False)
    res_qcs = []
    for _ in range(100):
        block_circuit = {}
        initial_layout = None  # list(range(backend.n_qubits))
        res_qc = QuantumCircuit(circuit_info['num_qubits'])
        res_qc, final_layout = my_routing(init_qc, backend, initial_layout)
        
        if len(res_qc) == 0:
            continue
        
        res_qcs.append(res_qc)

        
    best_baseline_qc, best_quct_qc = None, None
    best_baseline_predict, best_quct_predict = 0, 0 
    for res_qc in res_qcs:
        baseline_predict = .999 ** len(res_qc)

        circuit_info = upstream_model.vectorize(res_qc)
        quct_predict = downstream_model.predict_fidelity(circuit_info)
        
        if baseline_predict > best_baseline_predict:
            best_baseline_predict = baseline_predict
            best_baseline_qc = res_qc
        
        if quct_predict > best_quct_predict:
            best_quct_predict = quct_predict
            best_quct_qc = res_qc
        
    sim_result = simulate_noise_free(best_quct_qc)
    assert len(sim_result) == 1 and '00000' in sim_result, sim_result

    sim_result = simulate_noise_free(best_baseline_qc)
    assert len(sim_result) == 1 and '00000' in sim_result, sim_result

    # print(sim_result)
    # print(res_qc)
    return best_quct_qc, best_baseline_qc


@ray.remote
def opt_trans_remote(circuit_info, downstream_model, max_layer):
    return opt_trans(circuit_info, downstream_model, max_layer)


def sumarize_datasize(dataset, name):
    data = []
    labels = []
    for circuit_info in dataset:
        data.append([circuit_info['n_erroneous_patterns'], len(circuit_info['gates']), len(circuit_info['layer2gates']),
                    circuit_info['n_erroneous_patterns'] /
                    len(circuit_info['gates']), circuit_info['two_qubit_prob'],
                    circuit_info['ground_truth_fidelity'], circuit_info['independent_fidelity'], circuit_info[
                    'ground_truth_fidelity'] - circuit_info['independent_fidelity'],
        ])
        labels.append(circuit_info['label'])

    random.shuffle(data)
    data = data[:3000]  # 太大了画出来的图太密了
    data = np.array(data)
    plot_correlation(data, ['n_erroneous_patterns',
                            'n_gates', 'depth', 'error_prop', 'two_qubit_prob', 'ground_truth_fidelity', 'independent_fidelity',
                            'ground_truth_fidelity - independent_fidelity'], color_features=labels, name=name)


if __name__ == '__main__':
    size = 3
    n_qubits = 5
    n_steps = 1

    with open(f"opt_5bit/error_params_predicts_5.pkl", "rb")as f:
        downstream_model, predicts, reals, props, _ = pickle.load(f)
    upstream_model = downstream_model.upstream_model
    backend = upstream_model.backend
    simulator = NoiseSimulator(upstream_model.backend)

    all_to_all_backend = Backend(n_qubits=5, topology=None)
    algos = gen_algorithms(5, all_to_all_backend, mirror=True, trans=True)

    print(upstream_model.erroneous_pattern)
    
    max_layer = 1000
    futures, futures_baseline = [], []

    # for circuit_info in algos:
    #     opt_trans(circuit_info, downstream_model, max_layer)
    
    for circuit_info in algos:
        futures.append(opt_trans_remote.remote(
            circuit_info, downstream_model, max_layer))
    
    
    futures = wait(futures)

    algos_routing, algos_baseline = [], []
    for quct, baseline in futures:
        quct = upstream_model.vectorize(quct)
        downstream_model.predict_fidelity(quct)

        baseline = upstream_model.vectorize(baseline)
        downstream_model.predict_fidelity(baseline)

        algos_routing.append(quct)
        algos_baseline.append(baseline)

    simulator.get_error_results(algos_baseline, upstream_model, multi_process=True,
                                erroneous_pattern=upstream_model.erroneous_pattern)
    simulator.get_error_results(algos_routing, upstream_model, multi_process=True,
                                erroneous_pattern=upstream_model.erroneous_pattern)

    for circuit_info, new_circuit, algo in zip(algos_baseline, algos_routing, algos):
        print(algo['id'], 'depth:', len(circuit_info['layer2gates']), '---->', len(new_circuit['layer2gates']),
              'gate_num:', len(
                  circuit_info['gates']), '---->', len(new_circuit['gates']),
              'predict:', circuit_info['circuit_predict'], '---->', new_circuit['circuit_predict'],
              'ground_truth_fidelity:', circuit_info[
                  'ground_truth_fidelity'], '---->', new_circuit['ground_truth_fidelity'],
              'n_erroneous_patterns:', circuit_info['n_erroneous_patterns'], '---->', new_circuit['n_erroneous_patterns'],)
