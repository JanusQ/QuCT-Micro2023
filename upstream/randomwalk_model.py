import random
from collections import defaultdict
from qiskit.circuit import Instruction
import pickle
import os
from circuit.parser import qiskit_to_layered_circuits
from circuit.formatter import instruction2str
import numpy as np
from qiskit.circuit import QuantumCircuit
from jax import grad, jit, vmap, pmap
import ray
from copy import deepcopy
from jax import numpy as jnp
import optax
from upstream.sparse_dimensionality_reduction import  sp_mds_reduce, sp_MDS, pad_to

# 利用随机游走来采样的方法

# def rephrase_gate(gate):
#     ''' The input is a gate. The output is a gate that constains its operated qubits'''

# 可能可以用 https://github.com/kerighan/graph-walker 来加速，但是看上去似乎没法指定节点

# 对于noise来说似乎需不需要label，直接区分单比特
class Step():
    '''A step contains (source gate, edge_type, target gate)'''

    def __init__(self, source, edge, target):
        self.source = instruction2str(source)  # it should not be changed in the future
        self.edge = edge
        self.target = instruction2str(target)

        self._hash_id = str(self)

        # self.  #label for noise simulation
        return

    def __hash__(self): return hash(self._hash_id)

    # self.source,
    def __str__(self): return f'{self.edge}-{self.target}'


# TODO: there should be a step factory in the future for saving memory
class Path():
    '''A path consists of a list of step'''

    def __init__(self, steps):
        self.steps = steps
        self._hash_id = str(self)

    def add(self, step):
        steps = list(self.steps)
        steps.append(step)
        return Path(steps)

    def __hash__(self): return hash(self._hash_id)

    def __str__(self): return ','.join([str(step) for step in self.steps])


def travel_instructions(circuit_info, head_instruction, path_per_node, max_step):
    traveled_paths = set()
    for _ in range(path_per_node):
        paths = randomwalk(circuit_info, head_instruction, max_step)
        for path in paths:
            # print(path)
            path_id = path._hash_id
            if path_id in traveled_paths:
                continue
            traveled_paths.add(path_id)

    op_qubits = [qubit for qubit in head_instruction['qubits']]
    op_qubits_str = "-".join([str(_q) for _q in op_qubits])
    traveled_paths.add(f'#Q{op_qubits_str}')  # 加一个比特自己的信息
    # traveled_paths.add(f'#G{head_instruction.operation.name}-{op_qubits_str}') # 加一个比特自己的信息，这个就是loop
    return traveled_paths


def train(dataset, max_step: int, path_per_node: int, offest=0):
    all_instruction2pathes = []

    for index, circuit_info in enumerate(dataset):
        if index % 100 == 0:
            print(f'train:{index}/{len(dataset)}, {offest}th offest')

        instruction2pathes = []
        for head_instruction in circuit_info['instructions']:
            traveled_paths = travel_instructions(circuit_info, head_instruction, path_per_node, max_step)
            instruction2pathes.append(traveled_paths)

        all_instruction2pathes.append(instruction2pathes)

    return all_instruction2pathes

@ray.remote
def remote_train(dataset, max_step: int, path_per_node: int, offest=0):
    return train(dataset, max_step, path_per_node, offest)

def randomwalk(circuit_info: dict, head_instruction: Instruction, max_step: int):
    # circuit = circuit_info['qiskit_circuit']
    layer2instructions = circuit_info['layer2instructions']
    instruction2layer = circuit_info['instruction2layer']
    instructions = circuit_info['instructions']
    # dagcircuit = circuit_info['dagcircuit']
    # nodes = circuit_info['nodes']

    now_instruction = head_instruction
    traveled_nodes = [head_instruction]

    now_path = Path([Step(head_instruction, 'loop', head_instruction)])  # 初始化一个指向自己的
    now_path_app = deepcopy(now_path)
    # now_path = Path(['head'])
    paths = [now_path_app]

    for _ in range(max_step):
        former_node = now_instruction
        # now_node_index = instructions.index(now_instruction)
        now_node_index = now_instruction['id']  # hard code in the mycircuit_to_dag
        now_layer = instruction2layer[now_node_index]
        parallel_gates = layer2instructions[now_layer]
        former_gates = [] if now_layer == 0 else layer2instructions[now_layer - 1]  # TODO: 暂时只管空间尺度的
        later_gates = [] if now_layer == len(layer2instructions)-1 else layer2instructions[now_layer + 1]

        # if len(parallel_gates) + len(former_gates) == 0: break
        
        step_type = None
        total_len = len(parallel_gates) + len(former_gates) + len(later_gates)
        choice = random.random()
        if choice < len(former_gates) / total_len:
            candidates = former_gates
            step_type = 'dependency'
        elif choice < (len(former_gates) + len(parallel_gates)) / total_len and len(parallel_gates) > 0:
            candidates = parallel_gates
            step_type = 'parallel'
        else:
            candidates = later_gates
            step_type = 'next'

        # candidates = parallel_gates + former_gates
        candidates = [candidate for candidate in candidates if candidate not in traveled_nodes]
        if len(candidates) == 0:
            break

        now_instruction = random.choice(candidates)
        traveled_nodes.append(now_instruction)

        # now_path = now_path.add(Step(former_node, 'parallel', now_instruction))
        now_path = now_path.add(Step(former_node, step_type, now_instruction))
        paths.append(now_path)

    return paths


# meta-path只有两种 gate-parallel-gate, gate-former-gate
# max_step: 定义了最大的步长


class RandomwalkModel():
    def __init__(self, max_step, path_per_node):
        '''
            max_step: maximum step size
        '''
        self.model = None
        self.hash_table = {}  # 存了路径(Path)到向量化后的index的映射
        self.reverse_hash_table = {}  #
        self.path_count = defaultdict(int)

        self.max_step = max_step
        self.path_per_node = path_per_node
        self.dataset = None
        self.reduced_dim = 100
        return

    def count_path(self, _hash_id):
        self.path_count[_hash_id] += 1

    def path_index(self, _hash_id):
        # _hash_id = path._hash_id

        if _hash_id not in self.hash_table:
            self.hash_table[_hash_id] = len(self.hash_table)
            self.reverse_hash_table[self.hash_table[_hash_id]] = _hash_id
        # else:
        #     print('hit')
        return self.hash_table[_hash_id]

    def has_path(self, _hash_id):
        return _hash_id in self.hash_table

    def batch_train(self, dataset, process_num=10):
        assert self.dataset is None
        self.dataset = dataset
        max_step = self.max_step
        path_per_node = self.path_per_node

        futures = []
        batch_size = len(dataset) // process_num
        for start in range(0, len(dataset), batch_size):
            sub_dataset = dataset[start: start + batch_size]
            sub_dataset_future = remote_train.remote(sub_dataset, max_step, path_per_node, start)
            futures.append(sub_dataset_future)

        all_instruction2pathes = []

        for future in futures:
            result = ray.get(future)
            for instruction2pathes in result:
                assert len(instruction2pathes) != 0
            all_instruction2pathes += result

        for circuit_info, instruction2pathes in zip(dataset, all_instruction2pathes):
            for traveled_paths in instruction2pathes:
                for path_id in traveled_paths:
                    self.count_path(path_id)

            assert len(instruction2pathes) == len(circuit_info['instructions'])
            circuit_info['instruction2pathes'] = instruction2pathes

        # unusual path can be removed
        for path, count in self.path_count.items():
            if count >= 10:
                self.path_index(path)

        for index, circuit_info in enumerate(dataset):
            circuit_info['path_indexs'] = []  # 原先叫sparse_vec
            for paths in circuit_info['instruction2pathes']:
                vec = [self.path_index(path_id) for path_id in paths if self.has_path(path_id)]
                vec.sort()
                circuit_info['path_indexs'].append(vec)

        self.all_instructions = []
        for circuit_info in self.dataset:
            for index, insturction in enumerate(circuit_info['instructions']):
                self.all_instructions.append(
                    (index, insturction, circuit_info)
                )
                
        print(len(self.hash_table))



    def train(self, dataset):
        assert self.dataset is None
        self.dataset = dataset
        max_step = self.max_step
        path_per_node = self.path_per_node

        futures = []
        batch_size = len(dataset)
        for start in range(0, len(dataset), batch_size):
            sub_dataset = dataset[start: start + batch_size]
            sub_dataset_future = train(sub_dataset, max_step, path_per_node, start)
            futures.append(sub_dataset_future)

        all_instruction2pathes = []

        for future in futures:
            result = future
            for instruction2pathes in result:
                assert len(instruction2pathes) != 0
            all_instruction2pathes += result

        for circuit_info, instruction2pathes in zip(dataset, all_instruction2pathes):
            for traveled_paths in instruction2pathes:
                for path_id in traveled_paths:
                    self.count_path(path_id)

            assert len(instruction2pathes) == len(circuit_info['instructions'])
            circuit_info['instruction2pathes'] = instruction2pathes

        # unusual path can be removed
        for path, count in self.path_count.items():
            if count >= 10:
                self.path_index(path)

        for index, circuit_info in enumerate(dataset):
            circuit_info['path_indexs'] = []  # 原先叫sparse_vec
            for paths in circuit_info['instruction2pathes']:
                vec = [self.path_index(path_id) for path_id in paths if self.has_path(path_id)]
                vec.sort()
                circuit_info['path_indexs'].append(vec)

        self.all_instructions = []
        for circuit_info in self.dataset:
            for index, insturction in enumerate(circuit_info['instructions']):
                self.all_instructions.append(
                    (index, insturction, circuit_info)
                )
                
        print(len(self.hash_table))

    # def travel_instructions(self, circuit_info, head_instruction, path_per_node, max_step):
    #     traveled_paths = set()
    #     for _ in range(path_per_node):
    #         paths = randomwalk(circuit_info, head_instruction, max_step) 
    #         for path in paths:
    #             # print(path)
    #             path_id = path._hash_id
    #             if path_id in traveled_paths:
    #                 continue
    #             traveled_paths.add(path_id)

    #     op_qubits = [qubit for qubit in head_instruction['qubits']]
    #     op_qubits_str = "-".join([str(_q) for _q in op_qubits])
    #     # traveled_paths.add(f'#Q{op_qubits_str}')  # 加一个比特自己的信息
    #     # traveled_paths.add(f'#G{head_instruction.operation.name}-{op_qubits_str}') # 加一个比特自己的信息，这个就是loop

    #     # print('path table:', len(self.hash_table))
    #     return traveled_paths

    @staticmethod
    def count_step(path_id: str) -> int:
        return len(path_id.split(','))

    def _construct_sparse_vec(self, path_indexs):
        path_values = []
        for index in path_indexs:
            path = self.reverse_hash_table[index]
            path_values.append(1 * self.reduced_scaling * (0.4 ** self.count_step(path)))
        return pad_to(path_indexs,path_values, self.path_per_node * (self.max_step + 1))  # 直接pad到最大长度安全一些

    def load_vecs(self):
        vecs = []
        print('vec...')

        for circuit_info in self.dataset:# 每一个电路
            sparse_vecs = []
            for path_indexs in circuit_info['path_indexs']:# 每一个节点
                sparse_vec = self._construct_sparse_vec(path_indexs)
                sparse_vecs.append(sparse_vec)
                vecs.append(sparse_vec)
            circuit_info['sparse_vecs'] = np.array(sparse_vecs, dtype=np.int64)
        vecs = np.array(vecs,dtype=np.int64)
        self.all_vecs = vecs
        print('load vec finish')
        return vecs
            
    reduced_scaling = 1000
    def load_reduced_vecs(self):
        '''come from  verify_randomwalk.ipynb'''
        vecs = self.load_vecs()

        print('len(vecs) = ', len(vecs))

        reudced_vecs = np.array(vecs,dtype=np.int64)
        random.shuffle(reudced_vecs)
        reudced_vecs = reudced_vecs[:200000]
        print('mds', len(reudced_vecs))
        print('reduced_dim',self.reduced_dim)

        vec_size = len(self.hash_table)
        reduced_params, _ = sp_MDS(reudced_vecs, vec_size, self.reduced_dim, epoch_num=12, print_interval=1, batch_size=10)
        self.reduced_params = reduced_params

        all_reduced_vecs = vmap(sp_mds_reduce, in_axes=(None, 0), out_axes=0)(reduced_params, vecs) / self.reduced_scaling
        
        point = 0 
        print('reducing...')
        for i,circuit_info in enumerate(self.dataset):
            circuit_reduced_vecs = []
            for instruction in circuit_info['instructions']:
                circuit_reduced_vecs.append(all_reduced_vecs[point])
                point += 1
            circuit_info['reduced_vecs'] = np.array(circuit_reduced_vecs, dtype=np.float)

        return all_reduced_vecs

    @staticmethod
    def load(path):
        path = os.path.join('model', path, )
        file = open(path, 'rb')
        model = pickle.load(file)
        file.close()
        return model

    def save(self, path):
        '''
            save hash_table and algorithm
        '''
        path = os.path.join('model', path, )
        file = open(path, 'wb')
        pickle.dump(self, file)
        file.close()
        return

    def vectorize(self, circuit):
        # assert circuit.num_qubits <=
        if isinstance(circuit, QuantumCircuit):
            circuit_info = qiskit_to_layered_circuits(circuit)
            circuit_info['qiskit_circuit'] = circuit
        elif isinstance(circuit, dict) :# and 'qiskit_circuit' in circuit
            circuit_info = circuit
        else:
            raise Exception(circuit, 'is a unexpected input')
        max_step = self.max_step
        path_per_node = self.path_per_node

        assert 'path_indexs' not in circuit_info

        circuit_info['path_indexs'] = []
        circuit_info['sparse_vecs'] = []
        for index, head_instruction in enumerate(circuit_info['instructions']):
            traveled_paths = travel_instructions(circuit_info, head_instruction, path_per_node, max_step)
            path_indexs = [self.path_index(path) for path in traveled_paths if self.has_path(path)]
            path_indexs.sort()
            circuit_info['path_indexs'].append(path_indexs)
            circuit_info['sparse_vecs'].append(self._construct_sparse_vec(path_indexs))

        circuit_info['sparse_vecs'] = np.array(circuit_info['sparse_vecs'], dtype=np.int64)

        if 'reduced_params' in self.__dict__:
            scaling = self.reduced_scaling
            reduced_params = self.reduced_params
         
            vecs = circuit_info['sparse_vecs']
            # vecs = [vec.tolist() for vec in vecs]
            # vecs = make_same_size(vecs)
            # vecs = np.array(vecs)
            circuit_info['reduced_vecs'] = np.array(vmap(sp_mds_reduce, in_axes=(None, 0), out_axes=0)(reduced_params, vecs)/scaling, dtype=np.float)
        return circuit_info
        

