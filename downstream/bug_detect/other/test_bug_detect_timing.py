from collections import defaultdict
from ctypes import pointer
from email.policy import default
from operator import index
from statistics import mode
from matplotlib.pyplot import axes, get
from qiskit import QuantumCircuit
from sklearn import neighbors
from sklearn.cluster import KMeans
from upstream.randomwalk_model import RandomwalkModel
import os
from jax import numpy as jnp, vmap
from sklearn.cluster import KMeans
import numpy as np
from dataset.dataset_loader import parse_circuit, my_format_circuit_to_qiskit
from dataset.get_data import get_data, get_dataset, get_dataset_bug_detection
from simulator.hardware_info import max_qubit_num
from qiskit.tools.visualization import circuit_drawer
from upstream.sparse_dimensionality_reduction import batch
from upstream.sparse_dimensionality_reduction import sp_mds_reduce, sp_multi_constance, sp_pluse, sp_MDS, pad_to, sp_dist, \
    sp_cos_dist
import matplotlib.pyplot as plt
import copy
import random
import statistics
import time
###### 定义BUG的类型如下：
# 1. 某一个子电路重复出现
# 2. 大小端错误
# 3. 某一个门错误
# 4. 分解电路出错


# from algorithm.get_data_bug import get_bug_circuit
import ray
ray.init()


total_bug_num = 0
success_num = 0

error_num = 0
totoal_identify_num = 0

time2info = []

def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    return np.argpartition(np.array(matrix), K, axis=axis)[:K]
    # a_part = np.argpartition(np.array(matrix), K, axis=axis)[:K]
    # if axis == 0:
    #     row_index = np.arange(len(a_part))
    #     a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
    #     return a_part[0:K, :][a_sec_argsort_K, row_index]
    # else:
    #     column_index = np.arange(matrix.shape[1 - axis])[:, None]
    #     a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
    #     return a_part[:, 0:K][column_index, a_sec_argsort_K]


@ray.remote
def identify_gate_usage(vecs, analyzed_vecs):
    results = []
    for analyzed_vec in analyzed_vecs:
        dists = vmap(func_dist, in_axes=(0, None), out_axes=0)(vecs, analyzed_vec)
        # dist_indexs = np.argsort(dists)[:3]  # 小的在前面
        dist_indexs = partition_arg_topK(dists, 3)
        nearest_dists = dists[dist_indexs]
        
        dist_indexs = dist_indexs[nearest_dists < 2]
        nearest_dists = dists[dist_indexs]
            
        results.append((dists[dist_indexs], dist_indexs))
    
    return results

def func_dist(vec1, vec2):
    return jnp.sqrt(sp_dist(vec1, vec2) / (RandomwalkModel.reduced_scaling**2))

def scan(max_step, path_per_node, bug_num, min_qubit_num, max_qubit_num):
    dataset = get_dataset_bug_detection(min_qubit_num, max_qubit_num)

    id2circuit_info = {
        circuit_info['id']: circuit_info
        for circuit_info in dataset
    }
    
    model = RandomwalkModel(max_step, path_per_node)  #max-step=2 会有14000维，max-step 也有10000维，减少生成的特征的数量``
    model.batch_train(dataset, 3)
    model.load_vecs()

    positive_vecs = np.array(model.all_vecs, dtype=np.int64)
    all_qubit_nums = np.array([circuit_info['num_qubits'] for _,_,circuit_info in model.all_instructions], dtype=np.int64)

    num_qubits2positive_vecs = defaultdict(list)
    num_qubits2instructions = defaultdict(list)

    for elm_index, (instruction_index, instruction, circuit_info) in enumerate(model.all_instructions):
        num_qubits2positive_vecs[circuit_info['num_qubits']].append(positive_vecs[elm_index])
        num_qubits2instructions[circuit_info['num_qubits']].append((instruction_index, instruction, circuit_info))

    for gate_num in num_qubits2positive_vecs:
        num_qubits2positive_vecs[gate_num] = jnp.array(num_qubits2positive_vecs[gate_num], dtype=jnp.int64)

    THREADSHOLD = None

    global total_bug_num, success_num, error_num, totoal_identify_num
    
    total_bug_num = 0
    success_num = 0

    error_num = 0
    totoal_identify_num = 0
    
    alg2n_qubit2success_num = defaultdict(lambda : defaultdict(int))
    alg2n_qubit2bug_num = defaultdict(lambda : defaultdict(int))
    alg2n_qubit2error_num = defaultdict(lambda : defaultdict(int))
    alg2n_qubit2identify_num = defaultdict(lambda : defaultdict(int))

    
    
    def find_bug(circuit_info, bug_instructions):  # bug_instructions是正样本
        start_time = time.time()
        
        global total_bug_num, success_num, error_num, totoal_identify_num
        alg_name = circuit_info['alg_id']
        circuit_info = model.vectorize(circuit_info)
        gate_vecs = circuit_info['sparse_vecs'] #.reshape(-1, 100)
        bug_circuit = circuit_info['qiskit_circuit']
        num_qubits = circuit_info['num_qubits']
        
        # assert len(bug_instructions) == 1

        instruction2nearest_circuits = []

        print('start', circuit_info['id'])
        
        futures = []
        batch_size = len(dataset) // 30
        if batch_size < 200:
            batch_size = 200
        for _gate_vecs in batch(gate_vecs, batch_size=200, should_shffule = False,):
            future = identify_gate_usage.remote(num_qubits2positive_vecs[num_qubits], _gate_vecs)
            futures.append(future)

        results = []
        for future in futures:
            results += ray.get(future)
        
        for nearest_dists, dist_indexs in results:
            nearest_positive_instructions = [
                num_qubits2instructions[num_qubits][_index]
                for _index in dist_indexs
            ]
            nearest_circuits = [
                elm[2]['id']
                for elm in nearest_positive_instructions
            ]

            nearest_circuits = set(nearest_circuits)
            
            instruction2nearest_circuits.append(nearest_circuits)
            
        for index, nearest_circuits in enumerate(instruction2nearest_circuits):

            neighbor_nearest_circuits = []
            for nearest_circuit_set in instruction2nearest_circuits[index-6:index] + instruction2nearest_circuits[index+1: index+6]: 
                neighbor_nearest_circuits += list(nearest_circuit_set)
            
            neighbor_mode_nearest_circuit1 = statistics.mode(neighbor_nearest_circuits)
            neighbor_nearest_circuits = [elm for elm in neighbor_nearest_circuits if elm != neighbor_mode_nearest_circuit1]
            
            if len(neighbor_nearest_circuits) > 0:
                neighbor_mode_nearest_circuit2 = statistics.mode(neighbor_nearest_circuits)
            else:
                neighbor_mode_nearest_circuit2 = None

            if neighbor_mode_nearest_circuit1 not in nearest_circuits:
                if neighbor_mode_nearest_circuit2 is None:
                    if index not in bug_instructions:
                        error_num += 1
                        alg2n_qubit2error_num[alg_name][num_qubits] += 1
                elif neighbor_mode_nearest_circuit2 not in nearest_circuits:
                    if index not in bug_instructions:
                        error_num += 1
                        alg2n_qubit2error_num[alg_name][num_qubits] += 1

            totoal_identify_num += 1
            alg2n_qubit2identify_num[alg_name][num_qubits] += 1

        
        total_bug_num += 0 if len(bug_instructions) == 0 else 1
        alg2n_qubit2bug_num[alg_name][num_qubits] += 0 if len(bug_instructions) == 0 else 1
        for index in bug_instructions:
            if circuit_info['original_id'] not in instruction2nearest_circuits[index]:
                success_num += 1
                alg2n_qubit2success_num[alg_name][num_qubits] += 1
                break
        
        spend_time = time.time() - start_time
        print('grove, len(gates):', len(circuit_info['instructions']), 'num_qubits', circuit_info['num_qubits'], 'time:', spend_time, )
        time2info.append((circuit_info['id'], len(circuit_info['instructions']), circuit_info['num_qubits'], spend_time))
        
        return bug_circuit

    from dataset.dataset1 import hamiltonian_simulation, ising, swap, QAOA_maxcut, qknn, qsvm
    from dataset.get_data import get_bitstr

    def construct_negative(circuit_info, bug_num):
        '''bug的宽度'''
        
        circuit = QuantumCircuit(circuit_info['num_qubits'])

        start_instruction = random.randint(0, len(circuit_info['instructions'])-1-bug_num)
        end_instruction = start_instruction + bug_num
        bug_instructions = list(range(start_instruction, end_instruction))

        for layer, layer_instructions in enumerate(circuit_info['layer2instructions']):
            for instruction in layer_instructions:
                name = instruction['name']
                qubits = instruction['qubits']
                params = instruction['params']
                id = instruction['id']

                add_bug = False
                if id in bug_instructions:
                    name = random.choice(['rx', 'ry', 'rz', 'h', 'cz', 'cx'])
                    add_bug = True

                    params = [random.random()]
                    qubit1 = random.randint(0, circuit_info['num_qubits']-1)
                    qubit2 = random.choice([qubit for qubit in range(circuit_info['num_qubits']) if qubit != qubit1])
                    qubits = [qubit1, qubit2]

                if name in ('rx', 'ry', 'rz'):
                    circuit.__getattribute__(name)(params[0], qubits[0])
                elif name in ('cz', 'cx'):
                    circuit.__getattribute__(name)(qubits[0], qubits[1])
                elif name in ('h'):
                    circuit.__getattribute__(name)(qubits[0])
                else:
                    circuit.__getattribute__(name)(*(params + qubits))


                circuit[len(circuit)-1].operation.label = f'bug-{name}' if add_bug else None
            circuit.barrier()
        
        bug_circuit_info = parse_circuit(circuit)
        for bug_instruction in bug_instructions:
            bug_circuit_info['instructions'][bug_instruction]['original'] = circuit_info['instructions'][bug_instruction]
        bug_circuit_info['id'] = circuit_info['id'] + f'_bug_{bug_instructions}'
        bug_circuit_info['original_id'] = circuit_info['id']

        return bug_circuit_info, bug_instructions

    al_names = ['hamiltonian_simulation']
    for al_name in al_names:
        for n_qubit in range(min_qubit_num, max_qubit_num):
            circuit_id = f'{al_name}_{n_qubit}'

            positive_circuit_info = id2circuit_info[circuit_id]
            # 这里就只管随机替换吧
            bug_circuit_info, bug_instructions = construct_negative(positive_circuit_info, bug_num = bug_num) 

            bug_circuit_info['alg_id'] = al_name

            find_bug(bug_circuit_info, bug_instructions)

    return alg2n_qubit2success_num, alg2n_qubit2bug_num, alg2n_qubit2error_num, alg2n_qubit2identify_num


step_num = 3
max_step = 2
path_per_node = max_step * 16
bug_num = 3
for min_qubit_num in range(10, 50, step_num):
    max_qubit_num = min_qubit_num + step_num
    scan(max_step, path_per_node, bug_num, min_qubit_num, max_qubit_num)
    
print(time2info)
