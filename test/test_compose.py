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
from circuit.dataset_loader import gen_random_circuits
from circuit.formatter import layered_circuits_to_qiskit, get_layered_instructions
from downstream.fidelity_predict.fidelity_analysis import smart_predict, error_param_rescale
from simulator.noise_free_simulator import simulate_noise_free
from simulator.noise_simulator import NoiseSimulator
from test_opt import count_error_path_num
from upstream.randomwalk_model import RandomwalkModel, extract_device
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_circuit_layout
from collections import defaultdict
from circuit.parser import get_circuit_duration
from utils.backend import Backend, gen_linear_topology, get_linear_neighbor_info
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
        initial_layout = _parse_initial_layout(list(range(n_qubits)), [qiskit_circuit])[0]
    pass_manager_config = PassManagerConfig(initial_layout=initial_layout, basis_gates=['rx', 'ry', 'rz', 'cz'], 
                                            routing_method = 'sabre', coupling_map = coupling_map)
    pass_manager = level_3_pass_manager(pass_manager_config)
    transpiled_circuit = pass_manager.run(qiskit_circuit, str(random.random()))
    return transpiled_circuit, pass_manager.property_set['final_layout']


def opt_trans(circuit_info, backend: Backend, max_layer = 10):
    res_qc = QuantumCircuit(circuit_info['num_qubits'])
    
    n_qubits = circuit_info['num_qubits']
        
    # v2p: qr: int
    
    # print(qiskit_circuit)
    # print(pass_manager.property_set['final_layout'])
    # print(transpiled_circuit)

    # for gate in transpiled_circuit.data:
    #     gate.operation.label = '-'.join([str(qubit.index) for qubit in gate[1]])
    
    # print(transpileed_circuit)
    # for layer in get_layered_instructions(transpileed_circuit):
    #     for 
        
    # for qr in transpileed_circuit.data[:circuit_info['num_qubits']]:
    #     print(qr[1])
    
    block_circuit = {}
    initial_layout = None #list(range(backend.n_qubits))
    for start in range(0, len(circuit_info['layer2gates']), max_layer):
        block_circuit['num_qubits'] = circuit_info['num_qubits']
        block_circuit['layer2gates'] = circuit_info['layer2gates'][start: start + max_layer]
        block_qc = layered_circuits_to_qiskit(block_circuit['num_qubits'], block_circuit['layer2gates'])

        # print(block_qc)
        for i in range(10):
            # block_qc_trans = transpile(block_qc, coupling_map = backend.coupling_map, optimization_level=3, basis_gates=['rx', 'ry', 'rz', 'cz'], initial_layout=initial_layout)
            block_qc_trans, final_layout = my_routing(block_qc, backend, initial_layout)
            
            # print(block_qc_trans)
            
            if random.random() < 0.4:
                best_block_qc = block_qc_trans
                # plot_circuit_layout()
                next_initial_layout = final_layout  #block_qc_trans._layout.initial_layout
                
        initial_layout = next_initial_layout
                
        res_qc = res_qc.compose(best_block_qc)
    
    # print(res_qc)
    # print(layered_circuits_to_qiskit(circuit_info['num_qubits'], circuit_info['layer2gates']))
    
    return res_qc         

if __name__ == '__main__':
    all_to_all_backend = Backend(n_qubits=5, topology=None)
    test_dataset = gen_random_circuits(min_gate=40, max_gate=170, n_circuits=5, two_qubit_gate_probs=[
                                  2, 5], gate_num_step=10, backend=all_to_all_backend, multi_process=True)
    test_dataset = test_dataset[20:30]

    
    max_layer = 10

    n_qubits = 5
    backend = Backend(n_qubits=5, topology = gen_linear_topology(n_qubits), neighbor_info =  get_linear_neighbor_info(n_qubits, 1))
    
    futures, non_opts = [],[]
    for circuit_info in test_dataset:
        # futures.append(opt_trans_remote.remote(circuit_info, downstream_model, max_layer))
        circuit = opt_trans(circuit_info, backend, max_layer)
        print(simulate_noise_free(layered_circuits_to_qiskit(circuit_info['num_qubits'], circuit_info['layer2gates'])))
        print(simulate_noise_free(circuit))
    