import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import ray
from qiskit.circuit import QuantumCircuit

from circuit.formatter import instruction2str
from circuit.parser import qiskit_to_layered_circuits
from utils.backend import Backend
from utils.ray_func import wait


# 利用随机游走来采样的方法

# def rephrase_gate(gate):
#     '''The input is a gate. The output is a gate that constains its operated qubits'''

# 可能可以用 https://github.com/kerighan/graph-walker 来加速，但是看上去似乎没法指定节点

# 对于noise来说似乎需不需要label，直接区分单比特


class Step():
    '''A step contains (source gate, edge_type, target gate)'''

    def __init__(self, source, edge, target):
        # it should not be changed in the future
        self.source = instruction2str(source)
        self.edge = edge
        self.target = instruction2str(target)

        # self._path_id = str(self)

        # self.  #label for noise simulation
        return

    def __hash__(self): return hash(str(self))

    # self.source,
    def __str__(self): 
        if self.edge == 'loop':
            return str(self.target)
        else:
            return f'{self.edge}-{self.target}'


# TODO: there should be a step factory in the future for saving memory
class Path():
    '''A path consists of a list of step'''

    def __init__(self, steps):
        self.steps = steps
        self._path_id = str(self)
        # temp = 0
        # if self._path_id == 'cz,2,3-cz,2,3':
        #     temp = 0

    def add(self, step):
        steps = list(self.steps)
        steps.append(step)
        return Path(steps)

    def __hash__(self): return hash(self._path_id)

    def __str__(self): return '-'.join([str(step) for step in self.steps])


def BFS(traveled_paths, traveled_gates, path, circuit_info: dict, now_gate: dict, head_gate: dict, neighbor_info: dict, max_step: int,
        path_per_node: int, directions: list):
    if len(traveled_paths) > path_per_node:
        return

    if max_step <= 0:
        return
    layer2gates = circuit_info['layer2gates']
    gate2layer = circuit_info['gate2layer']

    now_node_index = now_gate['id']  # hard code in the mycircuit_to_dag
    now_layer = gate2layer[now_node_index]
    parallel_gates = layer2gates[now_layer]
    former_gates = [] if now_layer == 0 else layer2gates[now_layer - 1]  # TODO: 暂时只管空间尺度的
    later_gates = [] if now_layer == len(
        layer2gates) - 1 else layer2gates[now_layer + 1]

    '''TODO: 可以配置是否只要前后啥的'''
    candidates = []
    if 'parallel' in directions:
        candidates += [('parallel', gate)
                       for gate in parallel_gates if gate != now_gate]
    if 'former' in directions:
        candidates += [('former', gate) for gate in former_gates]
    if 'next' in directions:
        candidates += [('next', gate) for gate in later_gates]

    ''' 对于gate只能到对应qubit周围的比特的门上 (neighbor_info)'''
    candidates = [
        (step_type, candidate)
        for step_type, candidate in candidates
        if candidate not in traveled_gates and
        any([(q1 in neighbor_info[q2] or q1 == q2) for q2 in now_gate['qubits'] for q1 in candidate['qubits']]) and
        any([(q1 in neighbor_info[q2] or q1 == q2)
            for q2 in traveled_gates[0]['qubits'] for q1 in candidate['qubits']])
    ]

    for step_type, next_gate in candidates:
        path_app = deepcopy(path)
        path_app = path_app.add(Step(now_gate, step_type, next_gate))
        path_id = path_app._path_id
        if path_id not in traveled_paths:
            traveled_paths.add(path_id)
        traveled_gates.append(next_gate)
        BFS(traveled_paths, traveled_gates, path_app, circuit_info, next_gate,
            head_gate, neighbor_info, max_step - 1, path_per_node, directions)
        traveled_gates.remove(next_gate)


def travel_gates_BFS(circuit_info, head_gate, path_per_node, max_step, neighbor_info,
                     directions=('parallel', 'former', 'next')):
    traveled_paths = set()

    traveled_gates = [head_gate]
    BFS(traveled_paths, traveled_gates, Path([Step(head_gate,'loop',head_gate)]), circuit_info, head_gate, head_gate, neighbor_info, max_step, path_per_node,
        directions)

    op_qubits_str = instruction2str(head_gate)
    traveled_paths.add(op_qubits_str)
    
    # 单独加一个idle的next的
    # 给fidelity用的
    
    # TODO: 检查下写对了没有
    # layer2gates = circuit_info['layer2gates']
    # gate2layer = circuit_info['gate2layer']

    # now_node_index = head_gate['id']  # hard code in the mycircuit_to_dag
    # now_layer = gate2layer[now_node_index]
    # for qubit in head_gate['qubits']:
    #     n_idle = 0
    #     for n_idle, layer_gates in enumerate(layer2gates[now_layer+1:now_layer+6]):
    #         if any([qubit in gate['qubits'] for gate in layer_gates]):
    #             break
    #     if n_idle > 0:
    #         # print(f'{op_qubits_str}-idle{n_idle},{qubit}')
    #         traveled_paths.add(f'{op_qubits_str}-idle{n_idle},{qubit}')
    
    return traveled_paths


def train(dataset, max_step: int, path_per_node: int, neighbor_info: dict, offest=0,
          directions=('parallel', 'former', 'next')):
    all_gate_paths = []

    for index, circuit_info in enumerate(dataset):
        # print(circuit_info['qiskit_circuit'])
        # print(layered_circuits_to_qiskit(circuit_info['num_qubits'], circuit_info['layer2gates'],))
        if index % 100 == 0:
            print(f'train:{index}/{len(dataset)}, {offest}th offest')

        gate_paths = []
        for head_gate in circuit_info['gates']:
            traveled_paths = travel_gates_BFS(circuit_info, head_gate, path_per_node, max_step, neighbor_info,
                                              directions)
            gate_paths.append(traveled_paths)

            # print('head_gate = ', head_gate)
            # for path in traveled_paths:
            #     print(path)
            # print('----------------------------------------------------------------')

        all_gate_paths.append(gate_paths)

    return all_gate_paths


@ray.remote
def remote_train(dataset, max_step: int, path_per_node: int, neighbor_info: dict, offest=0,
                 directions=('parallel', 'former', 'next')):
    return train(dataset, max_step, path_per_node, neighbor_info, offest, directions)


# meta-path只有三种 gate-parallel-gate, gate-former-gate, gate-next-gate
# max_step: 定义了最大的步长

def extract_device(gate):
    if len(gate['qubits']) == 2:
        return tuple(sorted(gate['qubits']))
    else:
        # return gate['qubits'][0]
        '''不准改回去！！！！'''
        return gate['qubits'][0]


class RandomwalkModel():
    def __init__(self, max_step, path_per_node, backend: Backend, travel_directions=('parallel', 'former', 'next')):
        '''
            max_step: maximum step size
        '''
        self.model = None

        # 这里的device可以是device也可以是coupler
        self.device2path_table = defaultdict(dict)  # 存了路径(Path)到向量化后的index的映射

        self.device2reverse_path_table = defaultdict(
            dict)  # qubit -> path -> index
        self.device2reverse_path_table_size = defaultdict(int)
        self.max_step = max_step
        self.path_per_node = path_per_node
        self.dataset = None
        # self.reduced_dim = 100

        self.backend = backend
        self.travel_directions = travel_directions
        self.n_qubits = backend.n_qubits
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

    '''TODO: rename to construct'''

    def train(self, dataset, multi_process: bool = False, process_num: int = 10, remove_redundancy = True, is_filter_path =False, filter_path =None):
        # 改成一种device一个path table
        '''TODO: 可以枚举来生成所有的path table'''

        # assert self.dataset is None
        # self.dataset = dataset

        backend: Backend = self.backend
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
                sub_dataset_future = remote_train.remote(sub_dataset, max_step, path_per_node, neighbor_info, start,
                                                         self.travel_directions)
            else:
                sub_dataset_future = train(sub_dataset, max_step, path_per_node, neighbor_info, start,
                                           self.travel_directions)
            futures.append(sub_dataset_future)

        all_gate_paths = []

        futures = wait(futures)

        for result in futures:
            for gate_paths in result:
                assert len(gate_paths) != 0

            all_gate_paths += result

        device2path_coexist_count = {}
        # device2path_count = defaultdict(lambda : defaultdict(int))

        print('count path')
        # path_count = defaultdict(lambda: defaultdict(int))
        path_count = {} # defaultdict(lambda: defaultdict(int))
        for qubit in range(backend.n_qubits):
            path_count[qubit] = defaultdict(int)
        for coupling in backend.coupling_map:
            path_count[tuple(coupling)] = defaultdict(int)
        
        for index, (circuit_info, gate_paths ) in enumerate(zip(dataset, all_gate_paths)):
            # print(index, '/', len(dataset))
            for gate_index, traveled_paths in enumerate(gate_paths):
                traveled_paths = list(traveled_paths)
                device = extract_device(circuit_info['gates'][gate_index])
                for path_id in traveled_paths:
                    path_count[device][path_id] += 1
                circuit_info['gate_paths'] = gate_paths
                
                if remove_redundancy:
                    for i1, p1 in enumerate(traveled_paths):
                        for p2 in traveled_paths[i1+1:]:
                            if device not in device2path_coexist_count:
                                device2path_coexist_count[device] = {}
                            if p1 not in device2path_coexist_count[device]:
                                device2path_coexist_count[device][p1] =  defaultdict(int)
                            device2path_coexist_count[device][p1][p2] += 1

            # assert len(gate_paths) == len(circuit_info['gates'])
            circuit_info['gate_paths'] = gate_paths

        '''TODO: 去掉冗余的, 要不直接算correlation吧'''
        '''很慢'''
        
        device2redundant_path = defaultdict(set)
        if remove_redundancy:
            print('remove redundancy')
            for device, _path_count in path_count.items():
                print(device, 'before removment', len(_path_count))
                redundant_path = device2redundant_path[device]
                _paths = list(_path_count.keys())
                for i1, p1 in enumerate(_paths):
                    if p1 in redundant_path:
                        continue
                    for i2, p2 in enumerate(_paths[i1+1:]):
                        if p2 in redundant_path or device not in device2path_coexist_count or   p1 not in device2path_coexist_count[device] or device2path_coexist_count[device][p1][p2] == 0:
                            continue
                        if device2path_coexist_count[device][p1][p2] / _path_count[p1] > 0.9 and _path_count[p1] / _path_count[p2] > 0.9 and _path_count[p2] / _path_count[p1] > 0.9:
                            # print(
                            #     device, p1, p2, device2path_coexist_count[device][p1][p2], _path_count[p1], _path_count[p1], 'is redundant')
                            if _path_count[p1] > _path_count[p2]:
                                redundant_path.add(p1)
                            else:
                                redundant_path.add(p2)
                print('after removment', len(_path_count) - len(redundant_path))


        # unusual paths are not added to the path table
        for device in path_count:
            for path_id, count in path_count[device].items():
                if count >= 10 and path_id not in device2redundant_path[device]:
                    if is_filter_path is True and ('rz' in path_id or path_id in filter_path[device]):
                        continue
                    self.path_index(device, path_id)

        print('random walk finish device size = ', len(self.device2path_table))

        self.max_table_size = 0
        for device, path_table in self.device2path_table.items():
            # self.device2reverse_path_table_size[device] = len(path_table)
            print(device, 'path table size = ', len(path_table))
            # print(path_table.keys())
            if len(path_table) > self.max_table_size:
                self.max_table_size = len(path_table)

        for circuit_info in dataset:
            circuit_info['path_indexs'] = []
            circuit_info['vecs'] = []

            for gate, paths in zip(circuit_info['gates'], circuit_info['gate_paths']):
                device = extract_device(gate)
                _path_index = [self.path_index(device, path_id) for path_id in paths if self.has_path(device, path_id)]
                _path_index.sort()
                circuit_info['path_indexs'].append(_path_index)

                vec = np.zeros(self.max_table_size, dtype=np.int32)
                if len(_path_index) != 0:
                    vec[np.array(_path_index)] = 1.
                circuit_info['vecs'].append(vec)

        # self.all_instructions = []
        # for circuit_info in self.dataset:
        #     for index, insturction in enumerate(circuit_info['gates']):
        #         self.all_instructions.append(
        #             (index, insturction, circuit_info)
        #         )

    @staticmethod
    def count_step(path_id: str) -> int:
        return len(path_id.split(','))

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
        # self.dataset = None
        path = os.path.join('model', path, )
        file = open(path, 'wb')
        pickle.dump(self, file)
        file.close()
        return



    def vectorize(self, circuit, gates=None):

        if isinstance(circuit, QuantumCircuit):
            circuit_info = qiskit_to_layered_circuits(circuit)
            circuit_info['qiskit_circuit'] = circuit
        elif isinstance(circuit, dict):  # and 'qiskit_circuit' in circuit
            circuit_info = circuit
        else:
            raise Exception(circuit, 'is a unexpected input')
        max_step = self.max_step
        path_per_node = self.path_per_node

        # if 'vecs' in circuit_info and circuit_info['vecs'] is not None and len(circuit_info['vecs'][0]) == self.max_table_size:
        #     return circuit_info

        neighbor_info = self.backend.neighbor_info
        circuit_info['path_indexs'] = []

        circuit_info['vecs'] = []
        circuit_info['gate_paths'] = []
        
        if 'map' in circuit_info:
            path_map = {str(k): str(v) for k,v in circuit_info['map'].items()}
            
        for gate in (gates if gates is not None else circuit_info['gates']):
            paths = travel_gates_BFS(circuit_info, gate, path_per_node, max_step, neighbor_info,
                                     directions=self.travel_directions)
            device = extract_device(gate)
            if 'map' in circuit_info:
                maped_paths = []
                if isinstance(device,tuple):
                    device = (circuit_info['map'][device[0]],circuit_info['map'][device[1]])
                else:
                    device = circuit_info['map'][device]
                for path in paths:
                    maped_paths.append(path.translate(path.maketrans(path_map)))
                paths = maped_paths
            _path_index = [self.path_index(
                device, path_id) for path_id in paths if self.has_path(device, path_id)]
            _path_index.sort()
            circuit_info['path_indexs'].append(_path_index)
            circuit_info['gate_paths'].append(paths)
            
            vec = np.zeros(self.max_table_size, dtype=np.float32)
            if len(_path_index) != 0:
                vec[np.array(_path_index)] = 1.
            circuit_info['vecs'].append(vec)

        return circuit_info

    def extract_paths_from_vec(self, device, gate_vector: np.array) -> list:
        # device = extract_device(gate)
        inclued_path_indexs = np.argwhere(gate_vector == 1)[:, 0]
        paths = [
            self.device2reverse_path_table[device][index]
            for index in inclued_path_indexs
        ]
        return paths

    # TODO: 还要有一个判断是不是已经加进去了
    @staticmethod
    def parse_gate_info(gate_info):
        elms = gate_info.split(',')
        gate = {
            'name': elms[0],
            'qubits': [int(qubit) for qubit in elms[1:]]
        }
        if elms[0] in ('rx', 'ry', 'rz'):
            # gate['params'] =
            # np.array([np.pi])  #np.zeros((1,)) #
            gate['params'] = np.random.rand(1) * 2 * np.pi
        if elms[0] in ('u'):
            # np.array([np.pi] * 3)  #np.zeros((3,)) #
            gate['params'] = np.random.rand(3) * 2 * np.pi
        elif elms[0] in ('cx', 'cz'):
            gate['params'] = []

        return gate

    def reconstruct(self, device, gate_vector: np.array) -> list:
        paths = self.extract_paths_from_vec(device, gate_vector)

        def add_to_layer(layer, gate):
            for other_gate in layer2gates[layer]:
                if instruction2str(other_gate) == instruction2str(gate):
                    # tuple(_gate['qubits']) == tuple(gate['qubits']):
                    return

                # assert all([qubit not in other_gate['qubits'] for qubit in gate['qubits']])
            layer2gates[layer].append(gate)
            return

        head_gate = {
            'name': 'u',
            'qubits': [random.randint(0, self.n_qubits-1)],
            # [random.random() * 2 *jnp.pi for _ in range(3)],
            'params':  np.ones((3,)) * np.pi * 2
        }
        # [head_gate]
        layer2gates = [
            list()
            for _ in range(self.max_step * 2 + 1)
        ]
        head_layer = self.max_step
        # layer2gates[head_layer].append(head_gate)
        add_to_layer(head_layer, head_gate)

        for path in paths:
            now_layer = head_layer

            elms = path.split('-')
            if len(elms) == 1:
                head_gate.update(self.parse_gate_info(elms[0]))
                head_gate['params'] *= 3
            else:
                for index in range(0, len(elms), 2):
                    relation, gate_info = elms[index:index + 2]
                    if relation == 'next':
                        now_layer += 1
                    elif relation == 'former':
                        now_layer -= 1
                    add_to_layer(now_layer, self.parse_gate_info(gate_info))
                    # layer2gates[now_layer].append(parse_gate_info(gate_info))

        layer2gates = [
            layer
            for layer in layer2gates
            if len(layer) > 0
        ]

        return layer2gates

    # def reconstruct(self, gate, gate_vector: np.array) -> list:

    #     # device = extract_device(gate)
    #     # inclued_path_indexs = np.argwhere(gate_vector==1)[:,0]
    #     paths = self.extract_paths_from_vec(gate, gate_vector)
    #     # [
    #     #     self.device2reverse_path_table[device][index]
    #     #     for index in inclued_path_indexs
    #     # ]

    #     def parse_gate_info(gate_info):
    #         elms = gate_info.split(',')
    #         gate = {
    #             'name': elms[0],
    #             'qubits': [int(qubit) for qubit in elms[1:]]
    #         }
    #         if elms[0] in ('rx', 'ry', 'rz'):
    #             gate['params'] = np.array([np.pi])
    #         if elms[0] in ('u'):
    #             gate['params'] = np.array([np.pi] * 3)
    #         elif elms[0] in ('cx', 'cz'):
    #             gate['params'] = []

    #         return gate

    #     head_gate = {}
    #     layer2gates = [
    #         [head_gate]
    #     ]
    #     for path in paths:
    #         # print(path)
    #         for now_layer, gates in enumerate(layer2gates):
    #             if head_gate in gates:
    #                 print('head gate in ', now_layer)
    #                 break

    #         elms = path.split('-')
    #         if len(elms) == 1:
    #             head_gate.update(parse_gate_info(elms[0]))
    #             head_gate['params'] *= 3
    #         else:
    #             for index in range(0, len(elms), 2):
    #                 relation, gate_info = elms[index:index+2]
    #                 if relation == 'next':
    #                     now_layer += 1
    #                     if now_layer == len(layer2gates):
    #                         layer2gates.append([
    #                             parse_gate_info(gate_info)
    #                         ])
    #                     else:
    #                         layer2gates[now_layer].append(parse_gate_info(gate_info))
    #                 elif relation == 'parallel':
    #                     layer2gates[now_layer].append(parse_gate_info(gate_info))
    #                 elif relation == 'former':
    #                     now_layer -= 1
    #                     if now_layer == -1:
    #                         layer2gates = [[
    #                             parse_gate_info(gate_info)
    #                         ]] + layer2gates
    #                         now_layer = 0
    #                     else:
    #                         layer2gates[now_layer].append(parse_gate_info(gate_info))

    #                 assert len(layer2gates) <= 3

    #     return layer2gates
