from collections import defaultdict
# from multiprocessing.pool import Pool
# ProcessPoolExecutor
from operator import index
from qiskit import QuantumCircuit
from sklearn import neighbors
from sklearn.cluster import KMeans
from upstream.randomwalk_model import RandomwalkModel
import os
from jax import numpy as jnp, vmap
import numpy as np
from dataset.dataset_loader import parse_circuit
from dataset.get_data import get_data, get_dataset, get_dataset_bug_detection
from analysis.cricuit_operation import assign_barrier, dynamic_decoupling
from qiskit.tools.visualization import circuit_drawer
from upstream.sparse_dimensionality_reduction import sp_multi_constance, sp_dist, construct_dense
import matplotlib.pyplot as plt
import random
import statistics
from ray.util.multiprocessing import Pool
import math
###### 定义BUG的类型如下：
# 1. 某一个子电路重复出现
# 2. 大小端错误
# 3. 某一个门错误
# 4. 分解电路出错


# from algorithm.get_data_bug import get_bug_circuit
import ray
# ray.init()


total_bug_num = 0
success_num = 0

error_num = 0
totoal_identify_num = 0

def scan(max_step, path_per_node, bug_num, min_qubit_num, max_qubit_num):
    dataset = get_dataset_bug_detection(min_qubit_num, max_qubit_num)
    # algorithm = algorithm * 3
    id2circuit_info = {
        circuit_info['id']: circuit_info
        for circuit_info in dataset
    }

    # algs = list(set(id2circuit_info.keys()))

    model = RandomwalkModel(max_step, path_per_node)  #max-step=2 会有14000维，max-step 也有10000维，减少生成的特征的数量``
    # model.batch_train(algorithm, 3)
    model.train(dataset)
    model.load_vecs()

    positive_vecs = np.array(model.all_vecs, dtype=np.int64)
    all_qubit_nums = np.array([circuit_info['num_qubits'] for _,_,circuit_info in model.all_instructions], dtype=np.int64)
    # neigative_dataset = copy.deepcopy(algorithm)

    num_qubits2positive_vecs = defaultdict(list)
    num_qubits2instructions = defaultdict(list)

    for elm_index, (instruction_index, instruction, circuit_info) in enumerate(model.all_instructions):
        num_qubits2positive_vecs[circuit_info['num_qubits']].append(positive_vecs[elm_index])
        num_qubits2instructions[circuit_info['num_qubits']].append((instruction_index, instruction, circuit_info))

    for gate_num in num_qubits2positive_vecs:
        num_qubits2positive_vecs[gate_num] = jnp.array(num_qubits2positive_vecs[gate_num], dtype=jnp.int64)

    # 构造一个训练集
    # X, Y = [], []
    # for circuit_info in algorithm:
    #     gate_vecs = circuit_info['sparse_vecs']
    #     X += gate_vecs
        
    THREADSHOLD = None

    def func_dist(vec1, vec2):
        return jnp.sqrt(sp_dist(vec1, vec2) / (RandomwalkModel.reduced_scaling**2))
        # return sp_cos_dist(vec1, vec2)

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
        global total_bug_num, success_num, error_num, totoal_identify_num
        alg_name = circuit_info['alg_id']
        circuit_info = model.vectorize(circuit_info)
        gate_vecs = circuit_info['sparse_vecs'] #.reshape(-1, 100)
        bug_circuit = circuit_info['qiskit_circuit']
        num_qubits = circuit_info['num_qubits']
        
        # print(bug_circuit)
        # assert len(bug_instructions) == 1

        instruction2nearest_circuits = []

        print('start', circuit_info['id'])
        
        for analyzed_vec_index, analyzed_vec in enumerate(gate_vecs):
            this_instruction = circuit_info['gates'][analyzed_vec_index]
            has_bug = analyzed_vec_index in bug_instructions
            
            # if has_bug:
            #     print('-----------next has bug-------')
            
            dists = vmap(func_dist, in_axes=(0, None), out_axes=0)(num_qubits2positive_vecs[num_qubits], analyzed_vec)
            # dists = vmap(func_dist, in_axes=(0, None), out_axes=0)(positive_vecs, analyzed_vec)
            # dists += (1- (all_qubit_nums == circuit_info['num_qubits'])) * 20 # 先mask个提高下准确度
            
            dist_indexs = np.argsort(dists)[:3]  # 小的在前面
            nearest_dists = dists[dist_indexs]
            
            dist_indexs = dist_indexs[nearest_dists < 2]
            nearest_dists = dists[dist_indexs]
            
            nearest_positive_instructions = [
                num_qubits2instructions[num_qubits][_index]
                for _index in dist_indexs
            ]
            nearest_circuits = [
                elm[2]['id']
                for elm in nearest_positive_instructions
            ]
            
            nearest_circuit_instruction_index = [
                elm[0]
                for elm in nearest_positive_instructions
            ]
            
            # print(analyzed_vec_index, this_instruction['name'], this_instruction['qubits'], nearest_dists, nearest_circuits, nearest_circuit_instruction_index)

            if np.abs(analyzed_vec_index - bug_instructions[0]) <= 2:

                if analyzed_vec_index in bug_instructions:
                    print('next is bug') 
                
                print(circuit_info['id'], nearest_dists)
                print('this', circuit_info['gates'][analyzed_vec_index])
                # if analyzed_vec_index in bug_instructions:
                #     print(circuit_info['gates'][analyzed_vec_index]['original'])
                

                print('nearest_circuits', nearest_circuits) # 相近的电路
                print('nearest_positive_instructions', nearest_circuit_instruction_index)
                # print(nearest_positive_instructions[0]['sparse_vecs'][nearest_positive_instructions[0]['id']], circuit_info['sparse_vecs'][analyzed_vec_index])

                print('\n')

            nearest_circuits = set(nearest_circuits)
            
            instruction2nearest_circuits.append(nearest_circuits)
            
        # print('finish\n\n')
        #     nearest_index = np.argmin(dists) 
        #     nearest_dist = dists[nearest_index]
        #     # nearest_positive_instruction = model.all_instructions[nearest_index]
        #     nearest_positive_instruction = num_qubits2instructions[num_qubits][nearest_index]
        #     nearest_circuit = nearest_positive_instruction[2]
        #     nearest_circuits.append(nearest_circuit['id'])

        # global total_bug_num, success_num, error_num, totoal_identify_num

        for index, nearest_circuits in enumerate(instruction2nearest_circuits):

            neighbor_nearest_circuits = []
            for nearest_circuit_set in instruction2nearest_circuits[index-6:index] + instruction2nearest_circuits[index+1: index+6]: 
                neighbor_nearest_circuits += list(nearest_circuit_set)
            
            if len(neighbor_nearest_circuits) == 0:
                if index in bug_instructions:
                    success_num += 1
                    alg2n_qubit2success_num[alg_name][num_qubits] += 1
                # else:
                #     error_num += 1
                #     alg2n_qubit2error_num[alg_name][num_qubits] += 1
                continue
            
            neighbor_mode_nearest_circuit1 = statistics.mode(neighbor_nearest_circuits)
            neighbor_nearest_circuits = [elm for elm in neighbor_nearest_circuits if elm != neighbor_mode_nearest_circuit1]
            
            if len(neighbor_nearest_circuits) > 0:
                neighbor_mode_nearest_circuit2 = statistics.mode(neighbor_nearest_circuits)
            else:
                neighbor_mode_nearest_circuit2 = None
            # print(neighbor_mode_nearest_circuit, nearest_circuit, circuit_info['id'], index in bug_instructions)

            if neighbor_mode_nearest_circuit1 not in nearest_circuits:
                if neighbor_mode_nearest_circuit2 is None:
                    if index not in bug_instructions:
                        error_num += 1
                        alg2n_qubit2error_num[alg_name][num_qubits] += 1
                elif neighbor_mode_nearest_circuit2 not in nearest_circuits:
                    if index not in bug_instructions:
                        error_num += 1
                        alg2n_qubit2error_num[alg_name][num_qubits] += 1
                # else:
                #     success_num += 1
                    
            totoal_identify_num += 1
            alg2n_qubit2identify_num[alg_name][num_qubits] += 1
        # # nearest_circuits, 
        # print(circuit_info['id'], [nearest_circuits[index] for index in bug_instructions], '\n')
        
        
        total_bug_num += 0 if len(bug_instructions) == 0 else 1#len(bug_instructions)
        alg2n_qubit2bug_num[alg_name][num_qubits] += 0 if len(bug_instructions) == 0 else 1
        for index in bug_instructions:
            if circuit_info['original_id'] not in instruction2nearest_circuits[index]:
                success_num += 1
                alg2n_qubit2success_num[alg_name][num_qubits] += 1
                break

        print(bug_num, min_qubit_num, max_qubit_num, 'success rate:', success_num / total_bug_num)
        print(bug_num, min_qubit_num, max_qubit_num, 'error rate:', error_num / totoal_identify_num)
        return bug_circuit
        # return potential_bugs

    from dataset.get_data import get_bitstr

    def construct_negative(circuit_info, bug_num):
        '''bug的宽度'''
        
        circuit = QuantumCircuit(circuit_info['num_qubits'])

        start_instruction = random.randint(0, len(circuit_info['gates'])-1-bug_num)
        end_instruction = start_instruction + bug_num
        bug_instructions = list(range(start_instruction, end_instruction))
        # bug_instructions = list(range(len(circuit_info['gates'])))
        # random.shuffle(bug_instructions)
        # bug_instructions = bug_instructions[:bug_num]

        for layer, layer_instructions in enumerate(circuit_info['layer2gates']):
            for instruction in layer_instructions:
                name = instruction['name']
                qubits = instruction['qubits']
                params = instruction['params']
                id = instruction['id']

                add_bug = False
                if id in bug_instructions: #and name in ('rx', 'ry', 'rz', 'h', 'cz', 'cx'):
                    # if name in ('rx', 'ry', 'rz', 'h'):
                    #     name = random.choice(['rx', 'ry', 'rz','h'])
                    # if name in ('cz', 'cx'):
                    #     name = random.choice(['cz', 'cx'])
                    name = random.choice(['rx', 'ry', 'rz', 'h', 'cz', 'cx'])
                    add_bug = True

                    params = [random.random()]
                    qubit1 = random.randint(0, circuit_info['num_qubits']-1)
                    qubit2 = random.choice([qubit for qubit in range(circuit_info['num_qubits']) if qubit != qubit1])
                    qubits = [qubit1, qubit2]

                    # bug_num -= 1

                if name in ('rx', 'ry', 'rz'):
                    circuit.__getattribute__(name)(params[0], qubits[0])
                elif name in ('cz', 'cx'):
                    circuit.__getattribute__(name)(qubits[0], qubits[1])
                elif name in ('h'):
                    circuit.__getattribute__(name)(qubits[0])
                else:
                    if name in ('c4z', 'mcx_gray'):
                        circuit.__getattribute__('mct')(qubits[:-1], qubits[-1])
                    else:
                        circuit.__getattribute__(name)(*(params + qubits))


                circuit[len(circuit)-1].operation.label = f'bug-{name}' if add_bug else None
            circuit.barrier()
        
        bug_circuit_info = parse_circuit(circuit)
        for bug_instruction in bug_instructions:
            bug_circuit_info['gates'][bug_instruction]['original'] = circuit_info['gates'][bug_instruction]
        bug_circuit_info['id'] = circuit_info['id'] + f'_bug_{bug_instructions}'
        bug_circuit_info['original_id'] = circuit_info['id']

        return bug_circuit_info, bug_instructions


    def bug_detect(al_name, n_qubit):
        circuit_id = f'{al_name}_{n_qubit}'

        positive_circuit_info = id2circuit_info[circuit_id]
        
        if len(positive_circuit_info['gates']) > 600000:
            print('warnning', al_name, n_qubit, 'has',  len(positive_circuit_info['gates']), 'gates')
            return
        # print(positive_circuit_info['qiskit_circuit'])
        
        # 这里就只管随机替换吧
        bug_circuit_info, bug_instructions = construct_negative(positive_circuit_info, bug_num = bug_num) 

        bug_circuit_info['alg_id'] = al_name
        # print(al_name)
        identified_circuit_info = find_bug(bug_circuit_info, bug_instructions)


    # bug_detect('hamiltonian_simulation', 4)

    al_names = ['hamiltonian_simulation', 'qknn', 'qsvm', 'vqc','ising','qft','ghz','qft_inverse']
    for al_name in al_names:
        for n_qubit in range(min_qubit_num, max_qubit_num):
                bug_detect(al_name, n_qubit)

    al_name = 'swap'
    odd = math.ceil(min_qubit_num / 2) * 2 + 1
    for n_qubits in range(odd, max_qubit_num, 2):  #[3, 5, 7, 9]:  # 大于3，奇数
        bug_detect(al_name, n_qubits)

    al_names = ['qnn', 'qugan']
    for al_name in al_names:
        for n_qubits in range(odd, max_qubit_num, 2):
            bug_detect(al_name, n_qubits)

    # al_name = 'simon'
    # even = math.ceil(min_qubit_num / 2) * 2
    # for n_qubits in range(even, max_qubit_num, 2):
    #     bug_detect(al_name, n_qubits)

    al_name = 'grover'
    for n_qubit in range(min_qubit_num, max_qubit_num):
        if n_qubit <= 20:
            bug_detect(al_name, n_qubit)
    
    al_names = ['deutsch_jozsa','bernstein_vazirani']
    for al_name in al_names:
        for n_qubits in range(min_qubit_num, max_qubit_num):
            bug_detect(al_name, n_qubits)

    # al_name = 'square_root'
    # triple = math.ceil(min_qubit_num / 3) * 3
    # for n_qubits in range(triple, max_qubit_num, 3):
    #     bug_detect(al_name, n_qubits)

    al_names = ['multiplier', 'qec_5_x']
    forth = math.ceil(min_qubit_num / 5) * 5
    for al_name in al_names:
        for n_qubits in range(forth, max_qubit_num, 5): #[1, 2]: #5的倍数
            bug_detect(al_name, n_qubits)


    print('bug_size', bug_num)
    print('alg2n_qubit2success_num=', alg2n_qubit2success_num)
    print('alg2n_qubit2bug_num=', alg2n_qubit2bug_num)
    print('alg2n_qubit2error_num=', alg2n_qubit2error_num)
    print('alg2n_qubit2identify_num=', alg2n_qubit2identify_num)
    
    return alg2n_qubit2success_num, alg2n_qubit2bug_num, alg2n_qubit2error_num, alg2n_qubit2identify_num

#### 创建dataset并训练
# min_qubit_num, max_qubit_num = 10, 20 # 10-20, 20-30, ...., 90-100
# bug_num = 3  # 1,2,3,4,5

@ray.remote(max_calls=12)
def scan_remote(max_step, path_per_node, bug_num, min_qubit_num, max_qubit_num):
    return scan(max_step, path_per_node, bug_num, min_qubit_num, max_qubit_num)

def to_dict(obj):
    new_obj = {}
    for key, elm in obj.items():
        if isinstance(elm, (dict, defaultdict)):
            new_obj[key] = to_dict(elm)
        else:
            new_obj[key] = elm
    return new_obj

# scan(4, 60, 3, 6, 7)

results = {}
step_num = 3
for min_qubit_num in range(10, 50, step_num):
    max_qubit_num = min_qubit_num + step_num
    for max_step in range(5, 7):
        path_per_node = max_step * 16
        for bug_num in range(1, 5):
            key = (min_qubit_num, max_qubit_num, bug_num, max_step)
            # if os.path.exists(f'./temp/debug_result_{key}.pkl'):
            #     continue
            future = scan_remote.remote(max_step, path_per_node, bug_num, min_qubit_num, max_qubit_num)
            # future = scan(max_step, path_per_node, bug_num, min_qubit_num, max_qubit_num)
            results[(min_qubit_num, max_qubit_num, bug_num, max_step)] = future
            # pass
# future = scan_remote.remote(max_step, path_per_node, bug_num, min_qubit_num, max_qubit_num)
# future = scan(3, 50, 2, 10, 15)
# results[(min_qubit_num, max_qubit_num, bug_num, max_step)] = future

import pickle
for key, future in results.items():
    # alg2n_qubit2success_num, alg2n_qubit2bug_num, alg2n_qubit2error_num, alg2n_qubit2identify_num = future.result()
    # result = results[key].get()
    result = ray.get(results[key])
    print(result)
    alg2n_qubit2success_num, alg2n_qubit2bug_num, alg2n_qubit2error_num, alg2n_qubit2identify_num = result
    # ray.get(results[key])
    
    result  = {
        'key': key,
        'alg2n_qubit2success_num': to_dict(alg2n_qubit2success_num),
        'alg2n_qubit2bug_num': to_dict(alg2n_qubit2bug_num), 
        'alg2n_qubit2error_num': to_dict(alg2n_qubit2error_num), 
        'alg2n_qubit2identify_num': to_dict(alg2n_qubit2identify_num) 
    }
    results[key] = result
    
    with open('temp_more_algo/debug_result_'+str(key)+'.pkl', 'wb') as file:
        pickle.dump(result, file)

with open('debug_result.pkl', 'wb') as file:
    pickle.dump(results, file)
    
print('finish')

# p.close()
# p.join()
