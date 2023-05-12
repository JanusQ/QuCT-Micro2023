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

def my_routing(qiskit_circuit, backend, initial_layout):
    
    # n_qubits = circuit_info['num_qubits']
    # qiskit_circuit = layered_circuits_to_qiskit(n_qubits, circuit_info['layer2gates'], barrier = False)
    # transpiled_circuit = transpile(qiskit_circuit, 
    #                            coupling_map = backend.coupling_map, optimization_level=3, 
    #                            basis_gates=['rx', 'ry', 'rz', 'cz'], routing_method = 'sabre')

    coupling_map = _parse_coupling_map(backend.coupling_map, backend)
    if initial_layout is None:
        initial_layout = list(range(n_qubits))
        random.shuffle(initial_layout)
        initial_layout = _parse_initial_layout(initial_layout, [qiskit_circuit])[0]
    else:
        initial_layout = _parse_initial_layout(initial_layout, [qiskit_circuit])[0]
    # pass_manager_config = PassManagerConfig(initial_layout=initial_layout, basis_gates=['rx', 'ry', 'rz', 'cz'], 
    #                                         routing_method = 'sabre', coupling_map = coupling_map)
    
    pass_manager_config = PassManagerConfig(initial_layout=initial_layout, basis_gates=['rx', 'ry', 'rz', 'cz'], 
                                            routing_method = 'stochastic', coupling_map = coupling_map)
    pass_manager = level_1_pass_manager(pass_manager_config)
    transpiled_circuit = pass_manager.run(qiskit_circuit, str(random.random()))
    return transpiled_circuit, pass_manager.property_set['final_layout']


def opt_trans(circuit_info, downstream_model, max_layer = 10, baseline = False):
    res_qc = QuantumCircuit(circuit_info['num_qubits'])
    
    n_qubits = circuit_info['num_qubits']
    
    upstream_model = downstream_model.upstream_model
    backend = upstream_model.backend
    # v2p: qr: int
    
    block_circuit = {}
    initial_layout = None #list(range(backend.n_qubits))
    for start in range(0, len(circuit_info['layer2gates']), max_layer):
        block_circuit['num_qubits'] = circuit_info['num_qubits']
        block_circuit['layer2gates'] = circuit_info['layer2gates'][start: start + max_layer]
        block_qc = layered_circuits_to_qiskit(block_circuit['num_qubits'], block_circuit['layer2gates'], barrier= False)

        best_predict = 0
        for _ in range(1000):
            # block_qc_trans = transpile(block_qc, coupling_map = backend.coupling_map, optimization_level=3, basis_gates=['rx', 'ry', 'rz', 'cz'], initial_layout=initial_layout)
            block_qc_trans, final_layout = my_routing(block_qc, backend, initial_layout)
            
            # print(block_qc)
            # print(block_qc_trans)
            new_circuit = copy.deepcopy(res_qc)
            new_circuit = new_circuit.compose(block_qc_trans)
            
            if len(block_qc_trans) == 0:
                print('empty')
                continue
            
            if baseline:
                new_predict = .999 ** len(new_circuit)
            else:
                new_circuit = upstream_model.vectorize(new_circuit)
                new_predict = downstream_model.predict_fidelity(new_circuit)
                # print(new_predict)
                assert new_predict < 1
                
            if new_predict > best_predict:
                best_block_qc = block_qc_trans
                best_predict = new_predict
                if final_layout is None:
                    next_initial_layout = initial_layout
                else:
                    v2p = list(final_layout._v2p.values())
                    if initial_layout is None:
                        next_initial_layout = v2p
                    else:
                        next_initial_layout = [v2p[i] for i in initial_layout]
                        
            # if baseline:
            #     break
        initial_layout = next_initial_layout
                
        res_qc = res_qc.compose(best_block_qc)
    
    # print(res_qc)
    # print(layered_circuits_to_qiskit(circuit_info['num_qubits'], circuit_info['layer2gates']))
    # res_qc = transpile(res_qc, coupling_map = backend.coupling_map, basis_gates= backend.basis_gates, optimization_level = 2)
    
    sim_result = simulate_noise_free(res_qc)
    assert len(sim_result) == 1 and '00000' in sim_result
    
    # print(sim_result)
    # print(res_qc)
    return res_qc         

@ray.remote
def opt_trans_remote(circuit_info, downstream_model, max_layer, baseline = False):
    return opt_trans(circuit_info, downstream_model, max_layer, baseline= baseline)



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

    # topology = gen_grid_topology(size)  # 3x3 9 qubits
    # new_topology = defaultdict(list)
    # for qubit in topology.keys():
    #     if qubit < n_qubits:
    #         for ele in topology[qubit]:
    #             if ele < n_qubits:
    #                 new_topology[qubit].append(ele)
    # topology = new_topology
    # neighbor_info = copy.deepcopy(topology)
    # coupling_map = topology_to_coupling_map(topology)
    

    
    
    with open(f"opt_5bit/error_params_predicts_5.pkl", "rb")as f:
        downstream_model, predicts, reals, durations, _ = pickle.load(f)
    upstream_model = downstream_model.upstream_model
    backend = upstream_model.backend
    simulator = NoiseSimulator(upstream_model.backend)

    # dataset = gen_random_circuits(min_gate=20, max_gate=400, n_circuits=5, two_qubit_gate_probs=[
    #     2, 5], gate_num_step=20, backend=backend, multi_process=True, circuit_type='random')
    # for circuit_info in dataset:
    #     upstream_model.vectorize(circuit_info)
    # for elm in dataset:
    #     elm['label'] = 'train'
    # simulator.get_error_results(dataset, upstream_model, multi_process=True, erroneous_pattern=upstream_model.erroneous_pattern)
    # sumarize_datasize(dataset, 'q5_new_compose')
        

    all_to_all_backend = Backend(n_qubits=5, topology=None)
    # all_to_all_backend.basis_single_gates = None
    # all_to_all_backend.basis_two_gates = None
    # all_to_all_backend.basis_gates = None
    algos = gen_algorithms(5, all_to_all_backend, mirror = True, trans = True)

    max_layer = 1000
    futures, futures_baseline = [],[]
    
    # for circuit_info in algos:
    #     # opt_trans(circuit_info, downstream_model, max_layer, baseline = False)
    #     opt_trans(circuit_info, downstream_model, max_layer, baseline = True)
    
    for circuit_info in algos:
        # futures.append(opt_trans_remote.remote(circuit_info, downstream_model, max_layer))
        futures.append(opt_trans_remote.remote(circuit_info, downstream_model, max_layer, baseline = False ))
        futures_baseline.append(opt_trans_remote.remote(circuit_info, downstream_model, max_layer, baseline = True ))
    
    _algos_routing, _algos_baseline = [], []
    for future, future_baseline in zip(futures,futures_baseline):
        _algos_routing.append(ray.get(future))
        _algos_baseline.append(ray.get(future_baseline))
    
    algos_routing, algos_baseline =[], []
    for cir in _algos_baseline:
        # print(simulate_noise_free(cir))
        veced_cir = upstream_model.vectorize(cir)
        downstream_model.predict_fidelity(veced_cir)
        algos_baseline.append(veced_cir)
        
    for cir in _algos_routing:
        # print(simulate_noise_free(cir))
        veced_cir = upstream_model.vectorize(cir)
        downstream_model.predict_fidelity(veced_cir)
        algos_routing.append(veced_cir)
    
        
    simulator.get_error_results(algos_baseline, upstream_model, multi_process=True, erroneous_pattern=upstream_model.erroneous_pattern)
    simulator.get_error_results(algos_routing, upstream_model, multi_process=True, erroneous_pattern=upstream_model.erroneous_pattern)
    
    for circuit_info, new_circuit, algo in zip(algos_baseline, algos_routing, algos):
        print(algo['id'], 'depth:', len(circuit_info['layer2gates']), '---->', len(new_circuit['layer2gates']),
              'gate_num:', len(circuit_info['gates']), '---->', len(new_circuit['gates']),
              'predict:', circuit_info['circuit_predict'], '---->', new_circuit['circuit_predict'],
              'ground_truth_fidelity:', circuit_info['ground_truth_fidelity'], '---->', new_circuit['ground_truth_fidelity'],
              'n_erroneous_patterns:', circuit_info['n_erroneous_patterns'], '---->', new_circuit['n_erroneous_patterns'],)
