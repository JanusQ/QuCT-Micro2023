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
from upstream.sparse_dimensionality_reduction import sp_mds_reduce, sp_MDS, pad_to
from utils.backend import Backend
from circuit.formatter import layered_circuits_to_qiskit
from utils.ray_func import wait

'''只根据比特类型来序列化'''


def travel_gates_BFS(circuit_info, head_gate, path_per_node, max_step, neighbor_info,
                     directions=('parallel', 'former', 'next')):
    return [str(head_gate['qubits'])]


def train(dataset, max_step: int, path_per_node: int, neighbor_info: dict, offest=0,
          directions=('parallel', 'former', 'next')):
    all_gate_paths = []

    for index, circuit_info in enumerate(dataset):
        if index % 100 == 0:
            print(f'train:{index}/{len(dataset)}, {offest}th offest')

        gate_paths = []
        for head_gate in circuit_info['gates']:
            traveled_paths = travel_gates_BFS(circuit_info, head_gate, path_per_node, max_step, neighbor_info,
                                              directions)
            gate_paths.append(traveled_paths)

        all_gate_paths.append(gate_paths)

    return all_gate_paths

def extract_device(gate):
    if len(gate['qubits']) == 2:
        return tuple(sorted(gate['qubits']))
    else:
        # return gate['qubits'][0]
        '''不准改回去！！！！'''
        return gate['qubits'][0]

class QubitDependentModel():
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
        self.reduced_dim = 100

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

    def train(self, dataset, multi_process: bool = False, process_num: int = 10, remove_redundancy = True):
        remove_redundancy = False
        
        # 改成一种device一个path table
        '''TODO: 可以枚举来生成所有的path table'''

        assert self.dataset is None
        self.dataset = dataset

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
            
        # unusual paths are not added to the path table
        for device in path_count:
            for path_id, count in path_count[device].items():
                if count >= 10 and path_id:
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

        if 'vecs' in circuit_info and circuit_info['vecs'] is not None and len(circuit_info['vecs'][0]) == self.max_table_size:
            return circuit_info

        neighbor_info = self.backend.neighbor_info
        circuit_info['path_indexs'] = []

        circuit_info['vecs'] = []
        circuit_info['gate_paths'] = []
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

    def reconstruct(self, device, gate_vector: np.array) -> list:
        paths = self.extract_paths_from_vec(device, gate_vector)

        # TODO: 还要有一个判断是不是已经加进去了
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
                head_gate.update(parse_gate_info(elms[0]))
                head_gate['params'] *= 3
            else:
                for index in range(0, len(elms), 2):
                    relation, gate_info = elms[index:index + 2]
                    if relation == 'next':
                        now_layer += 1
                    elif relation == 'former':
                        now_layer -= 1
                    add_to_layer(now_layer, parse_gate_info(gate_info))
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
