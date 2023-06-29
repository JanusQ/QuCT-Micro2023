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

from downstream.synthesis.synthesis_model_pca_unitary_jax import SynthesisModelNN, SynthesisModelRandom, create_unitary_gate, find_parmas, pkl_dump, pkl_load, matrix_distance_squared, SynthesisModel, synthesize
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
import os
import json

ray.init(num_cpus = 40)
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


def eval(index, datasize, U, n_qubits, synthesis_model: SynthesisModel, n_unitary_candidates, n_neighbors):
    if isinstance(synthesis_model, str):
        synthesis_model: SynthesisModel = SynthesisModel.load(synthesis_model)

    target_backend = synthesis_model.backend

    print('synthesize', target_backend.n_qubits)

    use_heuristic = True
    # TODO:给他直接喂算法的电路
    synthesis_log = {}


    synthesized_circuit, cpu_time = synthesize(U, backend=target_backend, allowed_dist=1e-1,
                                                multi_process=True, heuristic_model=synthesis_model if use_heuristic else None,
                                                verbose=True, lagre_block_penalty=3, synthesis_log=synthesis_log, n_unitary_candidates=n_unitary_candidates, timeout=7*24*3600)

    qiskit_circuit = layered_circuits_to_qiskit(
        n_qubits, synthesized_circuit, barrier=False)

    result = {
        'n_qubits': n_qubits,
        'U': U,
        'qiskit circuit': qiskit_circuit,
        '#gate': len(qiskit_circuit),
        '#two-qubit gate': cnot_count(qiskit_circuit) + cz_count(qiskit_circuit),
        'depth': qiskit_circuit.depth(),
        'cpu time': cpu_time,
        'use heuristic': use_heuristic,
        'n_unitary_candidates': n_unitary_candidates,
        # 'baseline_name': filename,
        # 'baseline_dir': dirpath,
        'n_neighbors': n_neighbors,
        'datasize': datasize,
    }
    result.update(synthesis_log)

    print('RESULT')
    print({key: item for key, item in result.items()
            if key not in ('print', 'qiskit circuit', 'U')})
    print('\n')

    with open(f'eval_datasize/{use_heuristic}_{n_qubits}_{n_unitary_candidates}_{index}_{n_neighbors}_{datasize}.pkl', 'wb') as f:
        pickle.dump(result, f)

@ray.remote
def eval_remote(*args):
    return eval(*args)

n_qubits = 5
topology = gen_fulllyconnected_topology(n_qubits)
neigh_info = gen_fulllyconnected_topology(n_qubits)
target_backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                        basis_two_gates=['cz'], divide=False, decoupling=False)
                        
min_gate, max_gate = max([4**n_qubits - 100, 10]), 4**n_qubits

dataset = gen_random_circuits(min_gate=max_gate-20, max_gate=max_gate, gate_num_step=10, n_circuits=10,
                            two_qubit_gate_probs=[4, 7], backend=target_backend, reverse=False, optimize=False, multi_process=True, circuit_type='random')

n_step = 5
upstream_model = RandomwalkModel(
    n_step, 50, target_backend, travel_directions=('parallel', 'next'))
upstream_model.train(dataset, multi_process=True,
                    remove_redundancy=False, full_vec=False)

start_construct_data = time.time()
synthesis_model = SynthesisModelNN(upstream_model)
data = synthesis_model.construct_data(dataset, multi_process=True, n_random=10)
time_per_pair = (time.time() - start_construct_data) / len(data[0])

print('totoal data:',  len(data[0]), 'time_per_pair', time_per_pair)

random_index = list(range(len(data[0])))
random.shuffle(random_index)

datasize =100000
for n_neighbors in range(20, 40, 5):
    print(datasize)
    print('Path table size is', upstream_model.max_table_size)
    print('Device', len(upstream_model.device2path_table))
    synthesis_model_name = f'synthesis_{n_qubits}_{n_step}_{n_neighbors}_datasize{datasize}NN'


    device_gate_vecs, Us, sub_circuits = [], [], []
    for i in random_index[:datasize]:
        Us.append(data[0][i])
        device_gate_vecs.append(data[1][i])  
        sub_circuits.append(data[2][i])

    Us = np.array(Us)
    start_model_constuct = time.time()
    synthesis_model = SynthesisModelNN(upstream_model, synthesis_model_name)
    synthesis_model.construct_model((Us, device_gate_vecs, sub_circuits), n_neighbors)

    synthesis_model.data_constrcution_time = time_per_pair * datasize
    synthesis_model.model_constrcution_time = time.time() - start_model_constuct
    print('datasize', datasize, 'data_constrcution_time', synthesis_model.data_constrcution_time, 'model_constrcution_time', synthesis_model.model_constrcution_time)

    synthesis_model.save()

    print('\n\n\n\n')


# list(range(6, 10, 2))

# Us = [
#     unitary_group.rvs(2**n_qubits)
#     for _ in range(3)
# ]
with open('us.pkl','rb')as f:
    Us = pickle.load(f) 

n_unitary_candidates = 5
datasize =100000
for n_neighbors in range(20, 40, 5):
    synthesis_model_name = f'synthesis_{n_qubits}_{n_step}_{n_neighbors}_datasize{datasize}NN'
    futures = []
    for index, U in enumerate(Us):
        futures.append(eval_remote.remote(index, datasize, U, n_qubits, synthesis_model_name, n_unitary_candidates, n_neighbors))
        time.sleep(10)
    wait(futures)

