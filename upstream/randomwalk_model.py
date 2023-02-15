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
from utils.backend import Backend

# 利用随机游走来采样的方法

# def rephrase_gate(gate):
#     '''The input is a gate. The output is a gate that constains its operated qubits'''

# 可能可以用 https://github.com/kerighan/graph-walker 来加速，但是看上去似乎没法指定节点

# 对于noise来说似乎需不需要label，直接区分单比特
class Step():
    '''A step contains (source gate, edge_type, target gate)'''

    def __init__(self, source, edge, target):
        self.source = instruction2str(source)  # it should not be changed in the future
        self.edge = edge
        self.target = instruction2str(target)

        self._path_id = str(self)

        # self.  #label for noise simulation
        return

    def __hash__(self): return hash(self._path_id)

    # self.source,
    def __str__(self): return f'{self.edge}-{self.target}'


# TODO: there should be a step factory in the future for saving memory
class Path():
    '''A path consists of a list of step'''

    def __init__(self, steps):
        self.steps = steps
        self._path_id = str(self)

    def add(self, step):
        steps = list(self.steps)
        steps.append(step)
        return Path(steps)

    def __hash__(self): return hash(self._path_id)

    def __str__(self): return ','.join([str(step) for step in self.steps])


def travel_instructions(circuit_info, head_gate, path_per_node, max_step, neighbor_info):
    traveled_paths = set()
    for _ in range(path_per_node):
        paths = randomwalk(circuit_info, head_gate, neighbor_info, max_step)
        for path in paths:
            # print(path)
            path_id = path._path_id
            if path_id in traveled_paths:
                continue
            traveled_paths.add(path_id)

    op_qubits = [qubit for qubit in head_gate['qubits']]
    op_qubits_str = "-".join([str(_q) for _q in op_qubits])
    op_name = head_gate['name']
    # traveled_paths.add(f'#Q{op_qubits_str}')  # 加一个比特自己的信息
    traveled_paths.add(f'{op_name}-{op_qubits_str}') # 加一个比特自己的信息，这个就是loop
    
    return traveled_paths


def train(dataset, max_step: int, path_per_node: int, neighbor_info: dict, offest=0):
    all_gate_paths = []

    for index, circuit_info in enumerate(dataset):
        # print(circuit_info['qiskit_circuit'])
        if index % 100 == 0:
            print(f'train:{index}/{len(dataset)}, {offest}th offest')

        gate_paths = []
        for head_gate in circuit_info['gates']:
            traveled_paths = travel_instructions(circuit_info, head_gate, path_per_node, max_step, neighbor_info)
            gate_paths.append(traveled_paths)

            # print('head_gate = ', head_gate)
            # for path in traveled_paths:
            #     print(path)
            # print('----------------------------------------------------------------')
            
        all_gate_paths.append(gate_paths)

    return all_gate_paths

@ray.remote
def remote_train(dataset, max_step: int, path_per_node: int, neighbor_info: dict, offest=0):
    return train(dataset, max_step, path_per_node, neighbor_info, offest)

''' TODO: 改成步骤回头路的'''
def randomwalk(circuit_info: dict, head_gate: dict, neighbor_info: dict, max_step: int):
    # circuit = circuit_info['qiskit_circuit']
    layer2gates = circuit_info['layer2gates']
    gate2layer = circuit_info['gate2layer']
    # gates = circuit_info['gates']

    now_gate = head_gate
    traveled_gates = [head_gate]

    # now_path = Path([Step(head_instruction, 'head', head_instruction)])  # 初始化一个指向自己的
    # now_path = Path(['head'])
    
    now_path = Path([])
    now_path_app = deepcopy(now_path)
    paths = [now_path_app]

    for _ in range(max_step):
        now_node_index = now_gate['id']  # hard code in the mycircuit_to_dag
        now_layer = gate2layer[now_node_index]
        parallel_gates = layer2gates[now_layer]
        former_gates = [] if now_layer == 0 else layer2gates[now_layer - 1]  # TODO: 暂时只管空间尺度的
        later_gates = [] if now_layer == len(layer2gates)-1 else layer2gates[now_layer + 1]

        '''TODO: 可以配置是否只要前后啥的'''
        candidates = [('parallel', gate) for gate in parallel_gates if gate != now_gate] + [('fromer', gate) for gate in former_gates] + [('next', gate) for gate in later_gates]
        
        ''' update: 对于gate只能到对应qubit周围的比特的门上 (neighbor_info)'''
        candidates = [
            ( step_type, candidate)
            for step_type, candidate in candidates
            if candidate not in traveled_gates and any([q1 in neighbor_info[q2] or q1 == q2  for q2 in now_gate['qubits'] for q1 in candidate['qubits']])
        ]
        
        if len(candidates) == 0:
            break

        step_type, next_gate = random.choice(candidates)
        traveled_gates.append(now_gate)
        
        now_path = now_path.add(Step(now_gate, step_type, next_gate))
        now_gate = next_gate
        paths.append(now_path)

    return paths


# meta-path只有三种 gate-parallel-gate, gate-former-gate, gate-next-gate
# max_step: 定义了最大的步长

def extract_device(gate):
    if len(gate['qubits']) == 2:
        return tuple(sorted(gate['qubits']))
    else:
        return gate['qubits'][0]

class RandomwalkModel():
    def __init__(self, max_step, path_per_node, backend: Backend):
        '''
            max_step: maximum step size
        '''
        self.model = None
        
        # 这里的device可以是device也可以是coupler
        self.device2path_table = defaultdict(dict) # 存了路径(Path)到向量化后的index的映射
        
        self.device2reverse_path_table = defaultdict(dict)  # qubit -> path -> index

        self.max_step = max_step
        self.path_per_node = path_per_node
        self.dataset = None
        self.reduced_dim = 100
        
        self.backend = backend
        return

    def path_index(self, device, path_id):
        path_table = self.device2path_table[device]
        reverse_path_table = self.device2reverse_path_table[device]
        
        if path_id not in path_table:
            path_table[path_id] = len(path_table)
            reverse_path_table[path_table[path_id]] = path_id
        # else:
        #     print('hit')
        return path_table[path_id] 

    def has_path(self, device, path_id):
        return path_id in self.device2path_table[device]

    def train(self, dataset, multi_process: bool = False, process_num: int = 10):
        # 改成一种device一个path table
        
        '''TODO: 可以枚举来生成所有的path table'''
        
        assert self.dataset is None
        self.dataset = dataset
        
        neighbor_info = self.backend.neighbor_info
        max_step = self.max_step
        path_per_node = self.path_per_node
                
        if multi_process:
            batch_size = len(dataset) // process_num
        else:
            batch_size = len(dataset)
        
        futures = []
        for start in range(0, len(dataset), batch_size):
            sub_dataset = dataset[start: start + batch_size]
            if multi_process:
                sub_dataset_future = remote_train.remote(sub_dataset, max_step, path_per_node, neighbor_info, start)
            else:
                sub_dataset_future = train(sub_dataset, max_step, path_per_node, neighbor_info, start)
            futures.append(sub_dataset_future)

        all_gate_paths = []

        for future in futures:
            if multi_process:
                result = ray.get(future)
            else:
                result = future
                
            for gate_paths in result:
                assert len(gate_paths) != 0
                
            all_gate_paths += result

        path_count = defaultdict(lambda : defaultdict(int))
        for circuit_info, gate_paths in zip(dataset, all_gate_paths):
            for gate_index, traveled_paths in enumerate(gate_paths):
                device = extract_device(circuit_info['gates'][gate_index])
                for path_id in traveled_paths:
                    path_count[device][path_id] += 1

            assert len(gate_paths) == len(circuit_info['gates'])
            circuit_info['gate_paths'] = gate_paths

        # unusual paths are not added to the path table
        for device in path_count:
            for path_id, count in path_count[device].items():
                if count >= 10:
                    self.path_index(device, path_id)

        for index, circuit_info in enumerate(dataset):
            circuit_info['path_indexs'] = []
            for gate, paths in zip(circuit_info['gates'], circuit_info['gate_paths']):
                device = extract_device(gate)
                vec = [self.path_index(device, path_id) for path_id in paths if self.has_path(device, path_id)]
                vec.sort()
                circuit_info['path_indexs'].append(vec)

        # self.all_instructions = []
        # for circuit_info in self.dataset:
        #     for index, insturction in enumerate(circuit_info['gates']):
        #         self.all_instructions.append(
        #             (index, insturction, circuit_info)
        #         )
                
        print('random walk finish device size = ', len(self.device2path_table))
        for device, path_table in self.device2path_table.items():
            print(device, 'path table size = ', len(path_table))

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
            for instruction in circuit_info['gates']:
                circuit_reduced_vecs.append(all_reduced_vecs[point])
                point += 1
            circuit_info['reduced_vecs'] = np.array(circuit_reduced_vecs, dtype=np.float)

        return all_reduced_vecs

    @staticmethod
    def load(path):
        path = os.path.join('circuit/quct/pattern_extractor/model', path, )
        file = open(path, 'rb')
        model = pickle.load(file)
        file.close()
        return model

    def save(self, path):
        '''
            save hash_table and algorithm
        '''
        path = os.path.join('circuit/quct/pattern_extractor/model', path, )
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
        for index, head_instruction in enumerate(circuit_info['gates']):
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
        

