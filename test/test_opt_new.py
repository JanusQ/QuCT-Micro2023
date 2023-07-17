import copy
import pickle
from jax import numpy as jnp
from jax import vmap
import jax
import numpy as np
from numpy import random
import time
from qiskit.quantum_info.analysis import hellinger_fidelity
import ray
import matplotlib.pyplot as plt
from copy import deepcopy
from circuit.dataset_loader import gen_random_circuits
from circuit.formatter import layered_circuits_to_qiskit
from downstream.fidelity_predict.fidelity_analysis import smart_predict, error_param_rescale
from simulator.noise_simulator import NoiseSimulator
from test_opt import count_error_path_num
from upstream.randomwalk_model import RandomwalkModel, extract_device
from qiskit import QuantumCircuit, transpile
from collections import defaultdict
from circuit.parser import get_circuit_duration
from utils.backend import Backend, gen_grid_topology, gen_linear_topology, topology_to_coupling_map
from circuit.dataset_loader import gen_algorithms

def opt_move(circuit_info, downstream_model, threshold):
    upstream_model = downstream_model.upstream_model 
    
    
    new_circuit = {}         
    new_circuit['gates'] = []
    new_circuit['layer2gates'] = []
    new_circuit['gate2layer'] = []
    new_circuit['id'] = circuit_info['id']
    new_circuit['num_qubits'] = circuit_info['num_qubits']
    new_circuit['gate_num'] = len(circuit_info['gates'])
    new_circuit['ground_truth_fidelity'] = circuit_info['ground_truth_fidelity']
    # new_circuit['two_qubit_prob'] = circuit_info['two_qubit_prob']
    
    
    cur_layer = [0 for i in range(circuit_info['num_qubits'])]
    
    pre_fidelity = 1
    id = 0
    cnt = 0
    for layergates in circuit_info['layer2gates']:
        for _gate in layergates:
            
            gate = copy.deepcopy(_gate)
            new_circuit['gates'].append(gate)
            new_circuit['gate2layer'].append(-1)
            gate['id'] = id
            id += 1
            qubits = gate['qubits']
            
            offset = 0
            while True:
                if len(qubits) == 1:
                    qubit = qubits[0]
                    insert_layer = cur_layer[qubit] + offset
                    new_circuit['gate2layer'][-1] = insert_layer
                    if insert_layer >= len(new_circuit['layer2gates']):
                        assert insert_layer == len(new_circuit['layer2gates'])
                        new_circuit['layer2gates'].append([gate])
                    else:
                        new_circuit['layer2gates'][insert_layer].append(gate)
                        
                else:
                    qubit0 = qubits[0]
                    qubit1 = qubits[1]
                    insert_layer = max(cur_layer[qubit0],cur_layer[qubit1]) + offset
                    new_circuit['gate2layer'][-1] = insert_layer
                    if insert_layer >= len(new_circuit['layer2gates']):
                        assert insert_layer == len(new_circuit['layer2gates'])
                        new_circuit['layer2gates'].append([gate])
                    else:
                        new_circuit['layer2gates'][insert_layer].append(gate)
                
                
                
                new_circuit = upstream_model.vectorize(new_circuit)
                new_circuit['n_erroneous_patterns'] = count_error_path_num(new_circuit,upstream_model)
                # new_circuit['duration'] = get_circuit_duration(new_circuit['layer2gates'])
                new_fidelity = downstream_model.predict_fidelity(new_circuit)
                new_fidelity = new_fidelity if new_fidelity < 1 else 1
                # print(pre_fidelity,new_fidelity)
                
                if offset > 5 or pre_fidelity - new_fidelity < threshold:
                    if offset > 5:
                        print('threshold too small')
                    pre_fidelity = new_fidelity
                    if len(qubits) == 1:
                        cur_layer[qubit] = insert_layer + 1
                    else:
                        cur_layer[qubit0] = insert_layer+ 1
                        cur_layer[qubit1] = insert_layer+ 1
                    break
                else:
                    new_circuit['layer2gates'][insert_layer].remove(gate)
                    offset += 1
                    cnt += 1
                    
                    
    # print(layered_circuits_to_qiskit(circuit_info['num_qubits'], circuit_info['layer2gates']))                
    # print(layered_circuits_to_qiskit(new_circuit['num_qubits'], new_circuit['layer2gates']))
    print(new_circuit['id'], 'predict:', downstream_model.predict_fidelity(circuit_info), '--->', new_fidelity)
    return new_circuit, cnt

@ray.remote
def opt_move_remote(circuit_info, downstream_model, threshold):
    return opt_move(circuit_info, downstream_model, threshold)


                            
def test_opt_move(downstream_model, test_dataset):
    upstream_model = downstream_model.upstream_model

    print(upstream_model.erroneous_pattern)
    error_params = downstream_model.error_params

    # test_dataset = test_dataset[300:310]

    simulator = NoiseSimulator(upstream_model.backend)
    
    veced = []
    for cir in test_dataset:
        veced_cir = upstream_model.vectorize(cir)
        downstream_model.predict_fidelity(veced_cir)
        veced.append(veced_cir)
    test_dataset = veced
    
    simulator.get_error_results(test_dataset, upstream_model, multi_process=True, erroneous_pattern=upstream_model.erroneous_pattern)
    
    threshold = 0.005
        
    # for circuit_info in test_dataset:
    #     opt_move(circuit_info, downstream_model, threshold)
    
    futures = []
    for circuit_info in test_dataset:
        futures.append(opt_move_remote.remote(circuit_info, downstream_model, threshold))
    
    new_circuits = []
    cnts = []
    for future in futures:
        new_circuit, cnt = ray.get(future)
        new_circuits.append(new_circuit)
        cnts.append(cnt)
    
    simulator.get_error_results(new_circuits, upstream_model, multi_process=True, erroneous_pattern=upstream_model.erroneous_pattern)
    
    for circuit_info, new_circuit, cnt in zip(test_dataset, new_circuits, cnts):
        print(circuit_info['id'], 'depth:', len(circuit_info['layer2gates']), '---->', len(new_circuit['layer2gates']),
              'predict:', circuit_info['circuit_predict'], '---->', new_circuit['circuit_predict'],
              'ground_truth_fidelity:', circuit_info['ground_truth_fidelity'], '---->', new_circuit['ground_truth_fidelity'],
              'n_erroneous_patterns:', circuit_info['n_erroneous_patterns'], '---->', new_circuit['n_erroneous_patterns'],
              'opt_count:', cnt)

   
if __name__ == '__main__':
    size = 3
    n_qubits = 5
    n_steps = 1

    topology = gen_linear_topology(n_qubits)
    coupling_map = topology_to_coupling_map(topology)
    neighbor_info = copy.deepcopy(topology)
    
    algos = gen_algorithms(n_qubits, coupling_map, mirror = True, trans= True)
    
    
    with open(f"opt_5bit/error_params_predicts_5.pkl", "rb")as f:
        downstream_model, predicts, reals, durations, _ = pickle.load(f)
    test_opt_move(downstream_model,algos)