from circuit import gen_random_circuits
from circuit.algorithm.get_data_sys import get_dataset_alg_component
from circuit.formatter import qiskit_to_my_format_circuit, layered_circuits_to_qiskit, get_layered_instructions
from upstream import RandomwalkModel
from collections import defaultdict

import math
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, gen_linear_topology, get_linear_neighbor_info, gen_fulllyconnected_topology

from qiskit import QuantumCircuit
import pennylane as qml
import numpy as np
from downstream.synthesis.pennylane_op_jax import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op_jax import layer_circuit_to_matrix

from scipy.stats import unitary_group

from downstream.synthesis.synthesis_model_pca_unitary_jax import SynthesisModelNN, SynthesisModelRandom, find_parmas, pkl_dump, pkl_load, matrix_distance_squared, SynthesisModel, synthesize
from itertools import combinations
import time
from qiskit import transpile
import random
import cloudpickle as pickle
from circuit.formatter import layered_circuits_to_qiskit, to_unitary
from qiskit.quantum_info import Operator
import ray
from utils.ray_func import wait
from utils.unitaries import qft_U, grover_U, optimal_grover




def cnot_count(qc: QuantumCircuit):
    count_ops = qc.count_ops()
    if 'cx' in count_ops:
        return count_ops['cx']
    return 0


def cz_count(qc: QuantumCircuit):
    count_ops = qc.count_ops()
    if 'cz' in count_ops:
        return count_ops['cz']
    return 0


# def to_circuit_info(circuit, name, backend: Backend):
#     divide, decoupling, coupling_map, n_qubits = backend.divide, backend.decoupling, backend.coupling_map, backend.n_qubits
#     basis_single_gates, basis_two_gates = backend.basis_single_gates, backend.basis_two_gates
    
#     circuit = transpile(circuit, coupling_map=backend._true_coupling_map, optimization_level=3, basis_gates=(
#             basis_single_gates+basis_two_gates), initial_layout=[qubit for qubit in range(n_qubits)])
#     # except Exception as e:
#     #     traceback.print_exc()

#     # print(circuit)

#     circuit_info = {
#         'id': f'rc_{n_qubits}_{n_gates}_{two_qubit_prob}_{_}',
#         'qiskit_circuit': circuit
#     }

#     new_dataset = []
#     # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的
#     circuit_info = qiskit_to_layered_circuits(
#         _circuit_info['qiskit_circuit'], divide, decoupling)
#     circuit_info['id'] = _circuit_info['id']

#     circuit_info['duration'] = get_circuit_duration(
#         circuit_info['layer2gates'], backend.single_qubit_gate_time, backend.two_qubit_gate_time)
#     circuit_info['gate_num'] = len(circuit_info['gates'])
#     circuit_info['devide_qubits'] = backend.devide_qubits
#     circuit_info['two_qubit_prob'] = two_qubit_prob

#     # fig = circuit_info['qiskit_circuit'].draw('mpl')
#     # fig.savefig("devide_figure/"+str(_circuit_info['id'])+".svg")
    
#     # 减少模型大小
#     circuit_info['qiskit_circuit'] = None

#     new_dataset.append(circuit_info)
    
#     return 
    

def eval(n_qubits):
    
    topology = gen_fulllyconnected_topology(n_qubits)
    neigh_info = gen_fulllyconnected_topology(n_qubits)
    backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                    basis_two_gates=['cz'], divide=False, decoupling=False)
    n_step = n_qubits
    
    synthesis_model_name = f'synthesis_{n_qubits}_{n_step}NN'
    # synthesis_model: SynthesisModel = SynthesisModel.load(synthesis_model_name)
    # backend: Backend = synthesis_model.backend
    print('synthesize grover', backend.n_qubits)
    
    init_unitary_mat = grover_U(n_qubits)
    
    # min_gate, max_gate = max([2**n_qubits - 200, 10]), 2**n_qubits
    # dataset = gen_random_circuits(min_gate=min_gate, max_gate=max_gate, gate_num_step=max_gate//20, n_circuits=10,
    #                             two_qubit_gate_probs=[2, 5], backend=backend, reverse=False, optimize=True, multi_process=True)
    
    dataset = get_dataset_alg_component(n_qubits, backend)
    
    
    # grover_circuit = optimal_grover(n_qubits)
    
    upstream_model = RandomwalkModel(n_step, 100, backend)
    upstream_model.train(dataset, multi_process=True, remove_redundancy=False, full_vec=False)

    
    synthesis_model = SynthesisModelNN(upstream_model, synthesis_model_name)
    data = synthesis_model.construct_data(dataset, multi_process=False)
    print(f'data size of {synthesis_model_name} is {len(data[0])}')
    synthesis_model.construct_model(data)
    synthesis_model.save()

    synthesis_model: SynthesisModel = SynthesisModel.load(synthesis_model_name)
    backend: Backend = synthesis_model.backend
    print('synthesize', backend.n_qubits)
    
    for use_heuristic in [True]:
        start_time = time.time()

        # TODO:给他直接喂算法的电路
        synthesis_log = {}
        synthesized_circuit, cpu_time = synthesize(init_unitary_mat, backend=backend, allowed_dist=1e-2,
                                                multi_process=True, heuristic_model=synthesis_model if use_heuristic else None,
                                                verbose=False, lagre_block_penalty=4, synthesis_log = synthesis_log)
        synthesis_time = time.time() - start_time

        qiskit_circuit = layered_circuits_to_qiskit(
            n_qubits, synthesized_circuit, barrier=False)

        result = {
            'n_qubits': n_qubits,
            'U': init_unitary_mat,
            'qiskit circuit': qiskit_circuit,
            '#gate': len(qiskit_circuit),
            '#two-qubit gate': cnot_count(qiskit_circuit) + cz_count(qiskit_circuit),
            'depth': qiskit_circuit.depth(),
            # 'synthesis time': synthesis_time,  # 直接函数里面会存的
            'cpu time': cpu_time,
            'use heuristic': use_heuristic,
        }        
        result.update(synthesis_log)
        
        print({ key: item for key, item in result.items() if key not in ('print', 'qiskit circuit', 'U')})
        print('\n')

        with open(f'temp_data/synthesis_result/3_29/{n_qubits}_{use_heuristic}_grover_result.pkl', 'wb') as f:
            pickle.dump(result, f)

@ray.remote
def eval_remote(*args):
    return eval(*args)

# n_qubits = 3
for n_qubits in range(4, 7):

    futures = []
    # futures.append(eval_remote.remote(n_qubits))
    futures.append(eval(n_qubits))
        
wait(futures)