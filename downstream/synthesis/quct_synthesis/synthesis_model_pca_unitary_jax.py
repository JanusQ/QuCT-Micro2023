'''
    给定一个酉矩阵U，将其转化为一组门的组合
'''
# from downstream.synthesis import

import copy
import datetime
import itertools
import math
import os
import time
# from bqskit import compile
import traceback
from random import choice as random_choice, sample
from random import randint

import cloudpickle as pickle
import jax
import numpy as np
import optax
import pennylane as qml
from downstream.synthesis.quct_synthesis.pennylane_op_jax import layer_circuit_to_pennylane_circuit
# from sklearn.decomposition import IncrementalPCA as PCA
from jax import numpy as jnp
from jax import vmap
from jax.config import config
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from sklearn.neighbors import NearestNeighbors

from circuit import gen_random_circuits
from circuit.formatter import get_layered_instructions, qiskit_to_my_format_circuit
from circuit.formatter import layered_circuits_to_qiskit, to_unitary
from downstream.synthesis.quct_synthesis.tensor_network_op_jax import layer_circuit_to_matrix
from upstream import RandomwalkModel
from upstream.randomwalk_model import extract_device
from utils.backend import Backend
from utils.ray_func import *

config.update("jax_enable_x64", True)


def pkl_load(path):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            return pickle.load(file)


def pkl_dump(obj, path):
    if os.path.exists(path):
        print('Warning:', path, 'exists, which is copied to a new path')
        os.rename(path, path + str(datetime.datetime.now()))

    with open(path, 'wb') as file:
        return pickle.dump(obj, file)


def assign_params(params, layer2gates):
    layer2gates = copy.deepcopy(layer2gates)
    count = 0
    for gates in layer2gates:
        for gate in gates:
            for index, _ in enumerate(gate['params']):
                # gate['params'][index].set(params[count])
                gate['params'][index] = params[count]
                count += 1
    return layer2gates


def random_params(layer2gates):
    layer2gates = copy.deepcopy(layer2gates)
    for layer in layer2gates:
        for gate in layer:
            gate['params'] = np.random.rand(len(gate[
                                                    'params'])) * 2 * np.pi  # [  #     random.random() * 2 * jnp.pi  #     for param in gate['params']  # ]
    return layer2gates


@jax.jit
def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    # optimized implementation
    return jnp.abs(1 - jnp.abs(jnp.sum(jnp.multiply(A, jnp.conj(B)))) / A.shape[0])


identity_params = {}
with open('ae/synthesis/identity_params.pkl', 'rb') as file:
    identity_params: dict = pickle.load(file)


def create_unitary_gate(connect_qubits):
    if len(connect_qubits) == 1:
        return [[{'name': 'u', 'qubits': list(connect_qubits), 'params': np.zeros(3),  # np.random.rand(3) * 2 * np.pi,
        }]]
    # elif len(connect_qubits) == 2:
    #     return [
    #         [create_unitary_gate(connect_qubits[1:])[0][0], create_unitary_gate(connect_qubits[:1])[0][0]],
    #         [{'name': 'cz', 'qubits': list(connect_qubits), 'params': []}],
    #         [create_unitary_gate(connect_qubits[1:])[0][0], create_unitary_gate(connect_qubits[:1])[0][0]],
    #         [{'name': 'cz', 'qubits': list(connect_qubits), 'params': []}],
    #         [create_unitary_gate(connect_qubits[1:])[0][0], create_unitary_gate(connect_qubits[:1])[0][0]],
    #         [{'name': 'cz', 'qubits': list(connect_qubits), 'params': []}],
    #         [create_unitary_gate(connect_qubits[1:])[0][0], create_unitary_gate(connect_qubits[:1])[0][0]],
    #         [{'name': 'cz', 'qubits': list(connect_qubits), 'params': []}],
    #     ]
    else:
        # 现在的收敛方式似乎是有问题的
        n_connect_qubits = len(connect_qubits)
        return [[{'name': 'unitary', 'qubits': list(connect_qubits), # 不知道有没有参数量更少的方法
            'params': np.array(identity_params.get(n_connect_qubits, np.random.rand((4 ** n_connect_qubits) * 2))),
            # 'params': np.zeros((4**n_connect_qubits)*2),
        }]]


class PCA():
    def __init__(self, X, k=None, max_k=None, reduced_prop=None) -> None:
        X = np.concatenate([m.reshape((-1, m.shape[-1])) for m in X], axis=0)

        # 对 X 做中心化处理
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean

        # 计算 X 的协方差矩阵
        C = np.cov(X_centered.T)

        # 对 C 做特征值分解
        eigvals, eigvecs = jnp.linalg.eig(C)

        sorted_indices = jnp.argsort(eigvals)[::-1]
        sorted_eigen_values = eigvals[sorted_indices]

        sum_eigen_values = jnp.sum(sorted_eigen_values)

        if reduced_prop is not None:
            k = 0
            target_eigen_values = sum_eigen_values * reduced_prop
            accumulated_eigen_value = 0
            for eigen_value in sorted_eigen_values:
                accumulated_eigen_value += eigen_value
                k = k + 1
                if accumulated_eigen_value > target_eigen_values or (max_k is not None and k >= max_k):
                    break

        if k is not None:
            accumulated_eigen_value = 0
            for eigen_value in sorted_eigen_values[:k]:
                accumulated_eigen_value += eigen_value
            reduced_prop = accumulated_eigen_value / sum_eigen_values

        print('k =', k)
        print('reduced_prop =', reduced_prop)

        # 取前 k 个最大的特征值对应的特征向量

        self.k = k
        self.reduced_prop = reduced_prop
        self.V_k = eigvecs[:, sorted_indices[:k]]
        self.eigvecs = eigvecs
        self.sorted_indices = sorted_indices[:k]
        self.X_mean = X_mean
        pass

    def transform(self, X) -> jnp.array:
        reduced_matrices: jnp.array = vmap(pca_transform, in_axes=(0, None, None))(X, self.X_mean, self.V_k)
        return reduced_matrices.astype(jnp.float64)


@jax.jit
def pca_transform(m, X_mean, V_k):
    # 对每个输入矩阵都做降维，并且保持距离相似性
    m_centered = m - X_mean[jnp.newaxis, :]
    m_reduced = jnp.dot(m_centered, V_k)
    # 对降维后的矩阵做幺正正交化，保证在降维前后距离有相似性
    q, r = jnp.linalg.qr(m_reduced)
    q = q.reshape(q.size)
    q = jnp.concatenate([q.imag, q.real], dtype=jnp.float64)
    return q  # m_reduced #q


class SynthesisModel():
    def __init__(self, upstream_model: RandomwalkModel, name=None):
        self.upstream_model = upstream_model
        self.backend = upstream_model.backend
        self.n_qubits = upstream_model.n_qubits
        self.name = name

        self.pca_model = None
        self.U_to_vec_model = None

    def construct_data(self, circuits, multi_process=False, n_random=10):
        n_qubits = self.n_qubits
        n_steps = self.upstream_model.max_step

        # backend = self.backend
        # n_devices = n_qubits

        def gen_data(circuit_info, n_qubits, index):
            device_gate_vecs, Us = [], []
            sub_circuits = []
            layer2gates = circuit_info['layer2gates']

            for layer_index, layer_gates in enumerate(layer2gates):
                # print(layer_index)
                for target_gate in layer_gates:
                    if len(target_gate['qubits']) == 1:
                        continue

                    gate_vec = circuit_info['path_indexs'][target_gate['id']]  # 如果是奇奇怪怪的地方生成的这个地方可能会不对
                    if len(gate_vec) == 1:  # 只有一个门的直接过掉吧
                        continue

                    device_gate_vecs.append([extract_device(target_gate), gate_vec])
                    U = layer_circuit_to_matrix(layer2gates[layer_index:], n_qubits)  # TODO: vmap
                    # U = unitary_group.rvs(2**n_qubits)
                    Us.append(U)

                    # 只用一个就够了，不然可能还有一堆重复的
                    sub_circuit = layer2gates[layer_index:]
                    sub_circuits.append(sub_circuit[: n_steps])
                    break

            for layer_index, layer_gates in enumerate(layer2gates):
                layer_gates = [gate for gate in layer_gates if len(gate['qubits']) != 1]
                if len(layer_gates) == 0:
                    continue
                # print(layer_index)
                for _ in range(n_random):
                    target_gate_index = randint(0, len(layer_gates) - 1)
                    target_gate_index = layer_gates[target_gate_index]['id']
                    target_gate = circuit_info['gates'][target_gate_index]

                    gate_vec = circuit_info['path_indexs'][target_gate_index]  # .astype(np.int8)
                    if len(gate_vec) == 1:  # 只有一个门的直接过掉吧
                        continue

                    device_gate_vecs.append([extract_device(target_gate), gate_vec])

                    sub_circuit = random_params(layer2gates[layer_index:])

                    # U = unitary_group.rvs(2**n_qubits)
                    U = layer_circuit_to_matrix(sub_circuit, n_qubits)  # TODO: vmap
                    # assert np.allclose(U.T.conj() @ U, I)
                    Us.append(U)
                    sub_circuits.append(sub_circuit[: n_steps])

            # 本质上讲应该是通过 sub_circuits构建出来的，但是这里的construct不知道为啥没用了
            return device_gate_vecs, Us, sub_circuits

        @ray.remote
        def gen_data_remote(circuit_info, n_qubits, index):
            return gen_data(circuit_info, n_qubits, index)

        print('Start generating Us -> Vs, totoal', len(circuits))
        futures = []
        for index, circuit_info in enumerate(circuits):
            if multi_process:
                future = gen_data_remote.remote(circuit_info, n_qubits, index)
            else:
                future = gen_data(circuit_info, n_qubits, index)
            futures.append(future)

        futures = wait(futures, show_progress=True)

        Vs, Us = [], []
        sub_circuits = []
        for index, future in enumerate(futures):
            Vs += future[0]
            Us += future[1]
            sub_circuits += future[2]

        # Vs, Us = shuffle(Vs, Us)
        # Vs = np.array(Vs, dtype=np.int8)  # 加[:100000]，不然现在有点太慢了

        Us = np.array(Us, dtype=np.complex128)  # [:100000]

        print('len(Us) = ', len(Us), 'len(gate_vecs) = ', len(Vs))

        return Us, Vs, sub_circuits

    # def construct_model(self, data):
    #     print('Start construct model')

    #     assert False, '现在用sparse_vec现在跑不了了'

    #     start_time = time.time()

    #     Us, Vs = data

    #     self.pca_model = PCA(Us, reduced_prop=.7)
    #     output = self.pca_model.transform(Us)

    #     U_to_vec_model = DecisionTreeClassifier(splitter='random')
    #     U_to_vec_model.fit(output, Vs)

    #     self.U_to_vec_model = U_to_vec_model

    #     print(f'Finish construct model, costing {time.time()-start_time}s')

    # def choose(self, U, verbose = False):
    #     assert False, '现在用sparse_vec现在跑不了了'

    #     pca_model: PCA = self.pca_model
    #     U_to_vec_model: DecisionTreeClassifier = self.U_to_vec_model
    #     upstream_model = self.upstream_model
    #     backend = self.backend
    #     n_qubits = self.n_qubits

    #     device_gate_vec = pca_model.transform(jnp.array([U]))
    #     device_gate_vec = U_to_vec_model.predict(device_gate_vec)[0]
    #     ''' 基于预测的gate vector来构建子电路'''
    #     device = int(device_gate_vec[0])
    #     gate_vec = device_gate_vec[1:]
    #     '''TODO: 万一有冲突怎么办'''
    #     '''TODO:有可能找到的子电路还会使dist变大，可能需要类似残差的结构'''
    #     layer2gates = upstream_model.reconstruct(
    #         device, gate_vec)  # TODO: 如果出现全是0的怎么办
    #     # print('reconstruct', layer_gates)

    #     return [layer2gates] #, device_gate_vec

    #     involved_qubits = []
    #     connected_sets = []  # [[q1, q2], [q3, q4]]  存了在path里面连接的比特组
    #     for layer_gates in layer2gates:
    #         for gate in layer_gates:
    #             involved_qubits += gate['qubits']
    #             set_indexs = []
    #             for _index, _connected_set in enumerate(connected_sets):
    #                 if any([_qubit in _connected_set for _qubit in gate['qubits']]):
    #                     set_indexs.append(_index)

    #             if len(set_indexs) == 0:
    #                 connected_sets.append(list(gate['qubits']))
    #             elif len(set_indexs) == 1:
    #                 _index = set_indexs[0]
    #                 connected_sets[_index] += gate['qubits']
    #                 connected_sets[_index] = list(
    #                     set(connected_sets[_index]))
    #             elif len(set_indexs) > 2:
    #                 new_connected_set = []
    #                 for _index in set_indexs:
    #                     new_connected_set += connected_sets[_index]
    #                     connected_sets.remove(connected_sets[_index])
    #                 connected_sets.append(new_connected_set)

    #     return [[
    #         create_unitary_gate(connected_set)
    #         for connected_set in connected_sets
    #     ]], device_gate_vec

    def save(self, n_qubits):
        if not os.path.exists(f'ae/synthesis/quct/{n_qubits}'):
            os.makedirs(f'ae/synthesis/quct/{n_qubits}')
        with open(f'ae/synthesis/quct/{n_qubits}/{self.name}.pkl', 'wb') as f:
            pickle.dump(self, f)
        return

    @staticmethod
    def load(n_qubits, name):
        with open(f'ae/synthesis/quct/{n_qubits}/{name}.pkl', 'rb') as f:
            obj = pickle.load(f)
        return obj


class SynthesisModelRandom():
    def __init__(self, backend) -> None:
        self.backend = backend

    def choose(self, U, verbose=False):
        backend: Backend = self.backend
        n_qubits = backend.n_qubits
        circuit_infos = gen_random_circuits(min_gate=10 * n_qubits, max_gate=15 * n_qubits, backend=backend,
                                            gate_num_step=n_qubits * 2, n_circuits=1, two_qubit_gate_probs=[2, 4],
                                            optimize=True, reverse=False)

        if len(circuit_infos) > 15:
            circuit_infos = sample(circuit_infos, k=15)

        # for cirucit_info in circuit_infos:
        #     print(layered_circuits_to_qiskit(self.backend.n_qubits, cirucit_info['layer2gates'], False))

        return [cirucit_info['layer2gates'] for cirucit_info in circuit_infos# if len(cirucit_info['layer2gates']) < 5
        ]


def flatten(U):
    U = U.reshape(U.size)
    return np.concatenate([U.real, U.imag])


# @jax.jit
def unflatten(x):
    n_qubits = int(math.log2(x.size // 2) // 2)
    x_real = x[:x.size // 2]
    x_imag = x[x.size // 2:]
    return x_real.reshape((2 ** n_qubits, 2 ** n_qubits)) + 1j * x_imag.reshape((2 ** n_qubits, 2 ** n_qubits))


def hash_layer_gates(layer2gates: list[list]):
    circuit_tuple = []

    for layer in layer2gates:
        layer_tuple = []
        for gate in layer:
            layer_tuple.append(tuple(gate['qubits']))
        layer_tuple.sort()
        circuit_tuple.append(tuple(layer_tuple))

    return tuple(circuit_tuple)


'''去掉一个candidate里面前后一样的'''


def eliminate_candiate_rep(layer2gates: list[list]):
    former_tuple = []
    next_former_tuple = []
    for index, layer in enumerate(layer2gates):
        new_layer = []
        for gate in layer:
            qubits = tuple(gate['qubits'])
            if qubits not in former_tuple:
                new_layer.append(gate)
            else:
                temp = 0
            next_former_tuple.append(qubits)
        former_tuple = next_former_tuple
        next_former_tuple = []
        layer2gates[index] = new_layer
    return [layer for layer in layer2gates if len(layer) != 0]


class SynthesisModelNN(SynthesisModel):
    def __init__(self, upstream_model: RandomwalkModel, name=None):
        super().__init__(upstream_model, name)

    # @staticmethod
    # def flatten_matrix_distance_squared(x1, x2):
    #     x1 = unflatten(x1)
    #     x2 = unflatten(x2)
    #     return matrix_distance_squared(x1, x2)

    def construct_model(self, data, n_neighbors=15, reduced_prop=.6, max_k=100):
        print('Start construct SynthesisModelNN')
        start_time = time.time()

        Us, Vs, sub_circuits = data

        self.pca_model = PCA(Us, reduced_prop=reduced_prop, max_k=max_k)

        Us = self.pca_model.transform(Us)
        Us = [flatten(U) for U in Us]

        # metric= self.flatten_matrix_distance_squared,
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(Us)  # algorithm='ball_tree',
        self.Vs = Vs
        self.sub_circuits = sub_circuits

        print(f'Finish construct SynthesisModelNN, costing {time.time() - start_time}s')

    def choose(self, U, verbose=False):
        nbrs: NearestNeighbors = self.nbrs
        upstream_model = self.upstream_model
        Vs = self.Vs

        U = self.pca_model.transform(jnp.array([U]))[0]
        U = flatten(U)

        distances, indices = nbrs.kneighbors([U])
        distances, indices = distances[0], indices[0]
        ''' 基于预测的gate vector来构建子电路'''

        # if verbose:
        #     print('heuristic distance = ', distances)

        candidates = []
        # indices = indices[distances < .8]

        if len(indices) == 0: return []

        for index in indices:
            # device, sparse_gate_vec = Vs[index]
            # candidate = upstream_model.reconstruct(
            #     device, sparse_gate_vec)

            candidate = self.sub_circuits[index]

            candidate = [[create_unitary_gate(gate['qubits'])[0][0] for gate in layer_gates if len(gate['qubits']) == 2]
                for layer_gates in candidate]
            candidate = [layer_gates for layer_gates in candidate if len(layer_gates) != 0]
            if len(candidate) == 0:
                continue

            # TODO:不知道为啥没有identity
            # u = layer_circuit_to_matrix(candidate, self.backend.n_qubits)
            # if matrix_distance_squared(u, np.eye(2**self.backend.n_qubits)) > .1:
            #     print(matrix_distance_squared(u, np.eye(2**self.backend.n_qubits)))

            candidates.append(candidate)

        # hash2canditate_layers = {
        #     hash_layer_gates(candidate_layer): candidate_layer
        #     for candidate_layer in candidates
        # }
        # candidates = list(hash2canditate_layers.values())

        # for candidate in candidates:
        #     print(layered_circuits_to_qiskit(self.upstream_model.n_qubits, candidate, False))

        return candidates


# 现在的搜索没有随机性，因为每个门都是从identify开始的
def find_parmas(n_qubits, layer2gates, U, lr=1e-1, max_epoch=1000, allowed_dist=1e-2, n_iter_no_change=10,
                no_change_tolerance=1e-2, random_params=True, verbose=False):
    np.random.RandomState()

    param_size = 0
    params = []
    for layer in layer2gates:
        for gate in layer:
            param_size += len(gate['params'])
            params += list(gate['params'])

    if random_params:
        params = jax.random.normal(jax.random.PRNGKey(np.random.randint(0, 100)), (param_size,), dtype=jnp.float64)
    else:
        params = jnp.array(params, dtype=jnp.float64)

    dev = qml.device("default.qubit", wires=n_qubits * 2)
    '''lightning.qubit没法对unitary做优化'''
    # dev = qml.device("lightning.qubit", wires=n_qubits*2)
    '''TODO：hilbert_test相比于distance似乎会偏高一些，要计算一些换算公式'''
    '''TODO: 对比下hilbert_test和local hilbert_test，probs和expval的速度差异, 目前看来五比特以下差异不大 '''

    @qml.qnode(dev, interface="jax")  # interface如果没有jax就不会有梯度
    def hilbert_test(params, U):
        for q in range(n_qubits):
            qml.Hadamard(wires=q)

        for q in range(n_qubits):
            qml.CNOT(wires=[q, q + n_qubits])

        qml.QubitUnitary(U.conj(), wires=list(range(n_qubits)))
        '''TODO: jax.jit'''
        layer_circuit_to_pennylane_circuit(layer2gates, params=params, offest=n_qubits)

        for q in range(n_qubits):
            qml.CNOT(wires=[q, q + n_qubits])

        for q in range(n_qubits):
            qml.Hadamard(wires=q)

        '''hilbert_test get probs'''
        return qml.probs(list(range(n_qubits * 2)))

        '''hilbert_test get expval'''  # base = qml.PauliZ(0)  # for q in range(1, n_qubits*2):  #     base @= qml.PauliZ(q)  # return  qml.expval(base)

    def cost_hst(params, U):
        probs = hilbert_test(params, U)
        # return 1 - jnp.sqrt(probs[0]) # 会变慢
        return (1 - probs[0])

    opt = optax.adamw(learning_rate=lr)
    # opt = optax.adam(learning_rate=lr)
    # opt = optax.sgd(learning_rate=lr)
    opt_state = opt.init(params)

    min_loss = 1e2
    best_params = params

    loss_decrease_history = []
    epoch = 0
    # start_time = time.time()
    # jax.jacobian
    # value_and_grad = jax.value_and_grad(cost_hst)
    while True:
        loss_value, gradient = jax.value_and_grad(cost_hst)(params, U)  # 需要约1s
        # loss_value = 1 - math.sqrt(1-loss_value)  # 将HilbertSchmidt的distance计算变为matrix_distance_squared，HilbertSchmidt的公式具体看https://docs.pennylane.ai/en/stable/code/api/pennylane.HilbertSchmidt.html?highlight=HilbertSchmidt

        if verbose:
            circuit_U: jnp.array = layer_circuit_to_matrix(layer2gates, n_qubits, params)
            matrix_dist = matrix_distance_squared(circuit_U, U)
            print('Epoch: {:5d} | Loss: {:.5f}  | Dist: {:.5f}'.format(epoch, loss_value, matrix_dist))

        loss_decrease_history.append(min_loss - loss_value)
        if min_loss > loss_value and epoch != 0:  # 如果很开始就set了，就可能会一下子陷入全局最优
            min_loss = loss_value
            best_params = params  # print(epoch)

            # _loss_value = cost_hst(best_params, U)  # if abs(_loss_value - min_loss) > 1e-2:  #     print(min_loss, _loss_value, '????--')

        if epoch < n_iter_no_change:
            loss_no_change = False
        else:
            loss_no_change = True
            for loss_decrement in loss_decrease_history[-n_iter_no_change:]:
                if loss_decrement > no_change_tolerance:
                    loss_no_change = False
        if loss_no_change or epoch > max_epoch or loss_value < allowed_dist:
            break

        updates, opt_state = opt.update(gradient, opt_state, params)
        params = optax.apply_updates(params, updates)
        epoch += 1

    # _min_loss = cost_hst(best_params, U)
    # if abs(_min_loss - min_loss) > 1e-2:
    #     print(min_loss, _min_loss, '????')

    if min_loss > 1:  # 啥都没有优化出来
        min_loss = loss_value
    else:
        try:
            min_loss = 1 - math.sqrt(1 - min_loss)
        except:
            traceback.print_exc()
            print('min_loss', min_loss)

    return best_params, min_loss, epoch


# def hilbert_matrix_distance(n_qubits, total_layers, U, params):
#     dev = qml.device("default.qubit", wires=n_qubits*2)
#     @qml.qnode(dev, interface="jax")  # interface如果没有jax就不会有梯度
#     def hilbert_test():
#         for q in range(n_qubits):
#             qml.Hadamard(wires=q)

#         for q in range(n_qubits):
#             qml.CNOT(wires=[q, q+n_qubits])

#         qml.QubitUnitary(U.conj(), wires=list(range(n_qubits)))
#         layer_circuit_to_pennylane_circuit(
#             total_layers, params=params, offest=n_qubits)

#         for q in range(n_qubits):
#             qml.CNOT(wires=[q, q+n_qubits])

#         for q in range(n_qubits):
#             qml.Hadamard(wires=q)

#         return qml.probs(list(range(n_qubits*2)))

#     def cost_hst():
#         probs = hilbert_test()
#         return (1 - probs[0])

#     return  1 - math.sqrt(1-cost_hst())

def _optimize(now_layers, new_layers, n_optimized_layers, U, lr, n_iter_no_change, n_qubits, allowed_dist, now_dist,
              remote=False):
    if not remote:
        now_layers = copy.deepcopy(now_layers)
        new_layers = copy.deepcopy(new_layers)

    old_dist = now_dist
    start_time = time.time()

    # new_layers = random_params(new_layers)
    '''找到当前的最优参数'''
    total_layers = now_layers + new_layers
    unoptimized_layers, optimized_layers = total_layers[:-n_optimized_layers], total_layers[-n_optimized_layers:]
    unoptimized_U = layer_circuit_to_matrix(unoptimized_layers, n_qubits)

    # if len(unoptimized_layers) > 0:
    #     print(len(unoptimized_layers))

    '''TODO: 将连续的两个操作同一组qubits的unitary合并'''
    params, qml_dist, epoch_spent = find_parmas(n_qubits, optimized_layers, U @ unoptimized_U.T.conj(), max_epoch=500,
                                                allowed_dist=allowed_dist, n_iter_no_change=n_iter_no_change,
                                                no_change_tolerance=now_dist / 1000, random_params=False, lr=lr,
                                                verbose=False)
    # now_dist/10000

    # _dist =  hilbert_matrix_distance(n_qubits, optimized_layers, U @ unoptimized_U.T.conj(), params)
    # if abs(qml_dist - _dist) > 1e-3:
    #     print(qml_dist, _dist, params, '~~~')

    optimized_layers = assign_params(params, optimized_layers)
    total_layers = unoptimized_layers + optimized_layers

    epoch_finetunes = [epoch_spent]
    qml_dists = [qml_dist]

    qml_dist_finetune = qml_dist
    while len(epoch_finetunes) < 5:  # and n_qubits <= 5:
        lr = max([qml_dist / 5, 1e-2])  # 1e-1?
        params_finetune, qml_dist_finetune, epoch_finetune = find_parmas(n_qubits, total_layers, U, max_epoch=500,
                                                                         allowed_dist=allowed_dist, n_iter_no_change=5,
                                                                         no_change_tolerance=qml_dist / 100,
                                                                         random_params=False, lr=lr, verbose=False)
        epoch_finetunes.append(epoch_finetune)
        qml_dists.append(qml_dist)
        if (qml_dist - qml_dist_finetune) > qml_dist / 10:
            total_layers = assign_params(params_finetune, total_layers)
            qml_dist = qml_dist_finetune
        else:
            break

    optimize_time = time.time() - start_time
    if now_dist > old_dist:
        # optimize_time = optimize_time / sum(epoch_finetunes) * 5
        optimize_time = 0
    else:
        if qml_dist_finetune - qml_dist >= -1e-5:
            optimize_time = optimize_time / sum(epoch_finetunes) * epoch_spent
        pass

    circuit_U: jnp.array = layer_circuit_to_matrix(total_layers, n_qubits)
    remained_U = U @ circuit_U.T.conj()  # 加上这个remained_U之后circuit_U就等于U了, 如果不用heuristic的时候其实是不用算的

    now_dist = matrix_distance_squared(circuit_U, U)  # 和qml_dist一样了现在
    if now_dist - old_dist > 0.1 or abs(qml_dist - now_dist) > 1e-2:
        print(old_dist, qml_dist, now_dist, epoch_finetunes, qml_dists, len(unoptimized_layers), len(optimized_layers),
              len(total_layers), n_optimized_layers, '!!!!')

    return remained_U, total_layers, qml_dist, now_dist, epoch_finetunes + qml_dists, optimize_time


@ray.remote
def _optimize_remote(total_layers, new_layers, n_optimized_layers, invU, lr, n_iter_no_change, n_qubits, allowed_dist,
                     now_dist):
    return _optimize(total_layers, new_layers, n_optimized_layers, invU, lr, n_iter_no_change, n_qubits, allowed_dist,
                     now_dist, remote=True)


def synthesize(U, backend: Backend, allowed_dist=1e-5, heuristic_model: SynthesisModelNN = None, multi_process=True,
               verbose=False, lagre_block_penalty=2, timeout=7 * 24 * 3600, synthesis_log=None,
               n_unitary_candidates=40):
    # heuristic_ratio = .5,

    if synthesis_log is not None:
        # synthesis_log = {}
        synthesis_log['print'] = []

    start_time = time.time()
    subp_times = []

    if verbose:
        print('start synthesis', backend.n_qubits)

    # multi_process = True
    n_qubits = int(math.log2(len(U)))
    if n_qubits < 3:  # TODO: 等于 3
        gate = {'qubits': list(range(n_qubits)), 'unitary': U, }
        return synthesize_unitary_gate(gate, backend, allowed_dist, lagre_block_penalty, multi_process)

    I = jnp.eye(2 ** n_qubits)

    if matrix_distance_squared(U, I) <= allowed_dist:
        return QuantumCircuit(n_qubits)

    use_heuristic = heuristic_model is not None

    now_dist = 1
    remained_U = U
    total_layers = []
    iter_count = 0

    min_dist = 1e2

    heuristic_takeeffact_count = 0  # 记录下起heuristic作用的数量
    while now_dist > allowed_dist:
        if time.time() - start_time > timeout: break

        '''TODO: 有些电路要是没有优化动就不插了'''
        canditate_layers = []

        # device_gate_vec = None
        if use_heuristic and iter_count != 0:  # and now_dist > .5:
            # if use_heuristic and heuristic_ratio > random() and iter_count != 0 and now_dist > 0.3:
            '''相似的酉矩阵矩阵可能会导致重复插入相似的电路'''
            # canditates, device_gate_vec = heuristic_model.choose(remained_U,)
            # canditate_layers.append(new_layers)
            start_heuristic_time = time.time()
            canditates = heuristic_model.choose(remained_U, verbose=True)
            canditate_layers += canditates
            # if  time.time() - start_heuristic_time > 5:
            print('use heuristic costs', time.time() - start_heuristic_time, 's')  # 五比特花了13s

        # if True:
        # if not use_heuristic or device_gate_vec is None or (former_device_gate_vec is not None and jnp.allclose(device_gate_vec, former_device_gate_vec)):
        '''explore最优的子电路'''
        # 单比特直接搞一层算了
        new_layer = []  # 整个空的，就是单纯的findtune
        for _q in range(n_qubits):
            new_layer.append(create_unitary_gate([_q])[0][0])
        canditate_layers.append([new_layer])

        # max_n_subset_qubits = max([3, int(n_qubits//2)]) + 1
        # max_n_subset_qubits = max([3, int(n_qubits//3*2)]) + 1

        if iter_count != 0:
            unitary_candidates = []
            # max_n_subset_qubits = max([3, n_qubits - 1])
            # max_n_subset_qubits_step = max([1, (max_n_subset_qubits-3)//2])
            # TODO: 可能后面要设置成关于n_qubits的函数
            # for n_subset_qubits in [2, 3] + list(range(max_n_subset_qubits, 3, -max_n_subset_qubits_step)):

            if n_qubits >= 8:
                n_subset_qubits = 4
                for subset_qubits in backend.get_connected_qubit_sets(n_subset_qubits):
                    unitary_candidates.append(create_unitary_gate(subset_qubits))
            else:
                for n_subset_qubits in [3]:  # [2, 3]: #[2, 3]:
                    if n_qubits <= n_subset_qubits:  # 不会生成大于自己的比特数的子电路
                        continue

                    for subset_qubits in backend.get_connected_qubit_sets(n_subset_qubits):
                        unitary_candidates.append(create_unitary_gate(subset_qubits))

            # 增加多层的，尽量减少冗余的
            canditate_layers += unitary_candidates

            multi_layer_block_size = 2
            if n_qubits >= 8:
                multi_layer_block_size = 3

            #  这里如果要先全部遍历在生成会很慢

            qubit_sets_for_multilayer_unitary_candidates = list(itertools.product(
                *[backend.get_connected_qubit_sets(multi_layer_block_size) + [None] for depth in range(n_qubits - 2)]))
            if len(qubit_sets_for_multilayer_unitary_candidates) > n_unitary_candidates:
                qubit_sets_for_multilayer_unitary_candidates = sample(qubit_sets_for_multilayer_unitary_candidates,
                                                                      k=n_unitary_candidates)

            multilayer_unitary_candidates = []

            for qubit_sets in qubit_sets_for_multilayer_unitary_candidates:  #
                # TODO: 需要判断比如前后的是不是有一个是子集
                multi_layer_candiate = []
                for gate_qubits in qubit_sets:
                    if gate_qubits is None:
                        continue
                    multi_layer_candiate += create_unitary_gate(gate_qubits)
                multilayer_unitary_candidates.append(multi_layer_candiate)

            canditate_layers += multilayer_unitary_candidates

        for candidate in canditate_layers:
            eliminate_candiate_rep(candidate)

        # maximum_n_canditates = 15
        # if len(canditate_layers) > maximum_n_canditates:
        #     canditate_layers = sample(canditate_layers, k = maximum_n_canditates)

        '''去掉了哪些重复的'''
        hash2canditate_layers = {hash_layer_gates(candidate_layer): candidate_layer for candidate_layer in
            canditate_layers}
        canditate_layers = list(hash2canditate_layers.values())

        # for candidate in canditate_layers:
        #     print(layered_circuits_to_qiskit(n_qubits, candidate, False))

        if iter_count > 0:
            canditate_layers = [[]] + canditate_layers

        # canditate_layers = list(itertools.product(canditate_layers, canditate_layers))
        futures = []
        # former_device_gate_vec = device_gate_vec

        candiate_selection_start_time = time.time()
        for candidate_layer in canditate_layers:
            # TODO: 每隔一段时间进行一次finetune
            # (iter_count+1) % 10 == 0 or
            if now_dist < 1e-1:  # TODO: 需要尝试的超参数
                n_optimized_layers = len(total_layers) + len(candidate_layer)
            else:
                # n_optimized_layers = 5 + len(candidate_layer)  # TODO: 需要尝试的超参数,看下前面几层的变化大不大
                n_optimized_layers = 5 + len(candidate_layer)

            '''还没有试过效果, 似乎现在第一次和第二次的结果都差不多'''
            # if n_qubits > 5:
            #     n_optimized_layers = len(total_layers) + len(candidate_layer)

            if now_dist < 1e-2:
                n_iter_no_change = 20  # lr = .01
            else:
                n_iter_no_change = 10  # lr = .1

            lr = max([now_dist / 5, 1e-2])  # TODO: 需要尝试的超参数
            if multi_process:
                # max_cpu_num
                futures.append([candidate_layer,
                                _optimize_remote.remote(total_layers, candidate_layer, n_optimized_layers, U, lr,
                                    n_iter_no_change, n_qubits, allowed_dist, now_dist)])
            else:
                futures.append([candidate_layer,
                                _optimize(total_layers, candidate_layer, n_optimized_layers, U, lr, n_iter_no_change,
                                          n_qubits, allowed_dist, now_dist)])

        futures = wait(futures)
        max_dist_decrement = 0
        _min_dist = None

        def cal_penalty(candidate_layer):
            penalty = 1
            for index, layer in enumerate(candidate_layer):
                for gate in layer:
                    if gate['name'] != 'unitary':
                        penalty += len(gate['qubits']) / 2 ** index  # - 1
                    else:
                        penalty += lagre_block_penalty ** len(gate['qubits']) / 2 ** index  # 加一个门带来的distance减少并不是线性的
            return penalty

        # heuristic_takeeffact = False
        best_candidate = None
        for future in futures:
            candidate_layer, (c_remained_U, c_total_layers, c_qml_dist, c_now_dist, c_epoch, subp_time) = future
            subp_times.append(subp_time)

            '''TODO: 相似的距离优先比特数小的'''
            dist_decrement = now_dist - c_now_dist

            subset_penalty = cal_penalty(candidate_layer)
            if dist_decrement > 0:
                # 比特数多虽然能够减少的快，但是代价大
                dist_decrement /= subset_penalty
            else:
                dist_decrement *= subset_penalty

            if len(candidate_layer) != 0:
                candidate_layer = candidate_layer[0][0]
            else:
                candidate_layer = {'name': 'empty', 'qubits': []}

            if verbose:
                print(candidate_layer['name'], candidate_layer['qubits'], dist_decrement, now_dist, c_now_dist,
                      subset_penalty, c_epoch, subp_time)

            if synthesis_log is not None:
                synthesis_log['print'].append((candidate_layer['name'], candidate_layer['qubits'], dist_decrement,
                                               now_dist, c_now_dist, subset_penalty, c_epoch, subp_time))

            if dist_decrement > max_dist_decrement:
                max_dist_decrement = dist_decrement
                total_layers = c_total_layers
                remained_U = c_remained_U
                _min_dist = c_now_dist
                best_candidate = candidate_layer

        #         if len(c_total_layers) != 0 and c_total_layers[-1][0]['name'] != 'unitary' and len(c_total_layers) != 1:
        #             heuristic_takeeffact = True
        #         else:
        #             heuristic_takeeffact = False

        # if heuristic_takeeffact:
        #     # if verbose:
        #     #     print('heuristic takes effact')
        #     heuristic_takeeffact_count += 1

        if verbose:
            print('candiate selection cost', time.time() - candiate_selection_start_time)

        if synthesis_log is not None:
            synthesis_log['print'].append(('candiate selection cost', time.time() - candiate_selection_start_time))

        now_dist = _min_dist
        # print('choose', total_layers[-1][0]['name'], total_layers[-1][0]['qubits'], now_dist)

        if max_dist_decrement == 0:
            if verbose:
                print('Warning: no improvement in the exploration')
            future = random_choice(futures)
            best_candidate, (remained_U, total_layers, c_qml_dist, now_dist, epoch_spent, subp_time) = future

            if synthesis_log is not None:
                synthesis_log['print'].append(['Warning: no improvement in the exploration'])
        try:
            best_candidate = best_candidate[0][0]
            if best_candidate['name'] == 'empty':
                subp_times = subp_times[:-(len(futures) - 1)]  # empty是没有拟合好导致的，所以时间不算
        except:
            traceback.print_exc()

        if verbose:
            # 只显示一部分就好了，不然太乱了
            qiskit_circuit = layered_circuits_to_qiskit(n_qubits, total_layers[-8:], barrier=False)  # 0.00287s
            print(qiskit_circuit)
            print('iter_count=', iter_count, 'now_dist=', now_dist, '\n')

        if synthesis_log is not None:
            qiskit_circuit = layered_circuits_to_qiskit(n_qubits, total_layers[-8:], barrier=False)  # 0.00287s
            synthesis_log['print'].append([qiskit_circuit])
            synthesis_log['print'].append(['iter_count=', iter_count, 'now_dist=', now_dist, '\n'])

        if now_dist < min_dist:
            min_dist = now_dist

        iter_count += 1

    # multi_process = False

    if verbose:
        print('synthesize decomposed unitaries', backend.n_qubits)

    # 大于3比特的先放到别的进程去跑
    for layer_gates in total_layers:
        for gate in layer_gates:
            gate_name = gate['name']
            if gate_name == 'unitary':
                if len(gate['qubits']) >= 3:
                    if multi_process:
                        gate['synthesis_result'] = synthesize_unitary_gate_remote.remote(gate, backend,
                            allowed_dist=allowed_dist / 10, lagre_block_penalty=4 if len(gate['qubits']) == 3 else 2,
                            multi_process=multi_process)
                    else:
                        gate['synthesis_result'] = synthesize_unitary_gate(gate, backend,
                            allowed_dist=allowed_dist / 10, lagre_block_penalty=4 if len(gate['qubits']) == 3 else 2,
                            multi_process=multi_process)

    # 两比特的自己跑还快一些
    for layer_gates in total_layers:
        for gate in layer_gates:
            gate_name = gate['name']
            if gate_name == 'unitary':
                if len(gate['qubits']) == 2:
                    gate['synthesis_result'] = synthesize_unitary_gate(gate, backend, allowed_dist=allowed_dist / 10,
                        multi_process=multi_process)

    # if verbose:
    #     circuit_U: jnp.array = layer_circuit_to_matrix(total_layers, n_qubits)
    #     now_dist = matrix_distance_squared(circuit_U, U)
    #     print('before decomposition, the dist is', now_dist)

    new_total_layers = []
    for layer_gates in total_layers:
        for gate in layer_gates:
            gate_name = gate['name']
            if gate_name in ('u', 'cz', 'cx'):
                new_total_layers.append([gate])
            elif gate_name == 'unitary':
                unitary_circuit, subp_time = wait(gate['synthesis_result'])
                subp_times.append(subp_time)
                new_total_layers += unitary_circuit
            else:
                raise Exception('unknown', gate)

    '''最后finetune一下，反而变差了感觉像是hl_test对global phase也敏感的原因'''

    if verbose:
        print('cpu_time:', subp_times)

    qiskit_circuit: QuantumCircuit = layered_circuits_to_qiskit(n_qubits, new_total_layers, barrier=False)

    qiskit_circuit = transpile(qiskit_circuit, coupling_map=backend.coupling_map, optimization_level=3,
                               basis_gates=backend.basis_gates, initial_layout=[qubit for qubit in range(n_qubits)])
    layer2instructions, _, _, _, _ = get_layered_instructions(qiskit_circuit)
    synthesized_circuit, _, _ = qiskit_to_my_format_circuit(layer2instructions)

    # params, qml_dist, epoch_spent = find_parmas(n_qubits, synthesized_circuit, U, max_epoch=1e5, allowed_dist=1e-12, n_iter_no_change=20, no_change_tolerance=0, random_params=False, lr=1e-3, verbose=False)
    # assign_params(params, synthesized_circuit)

    if verbose:
        print('heuristic takes effect in', heuristic_takeeffact_count / iter_count * 100, '%')

    if synthesis_log is not None:
        synthesis_log['heuristic_takeeffect_count'] = heuristic_takeeffact_count
        synthesis_log['heuristic_takeeffect_prob'] = heuristic_takeeffact_count / iter_count
        synthesis_log['iter_count'] = iter_count
        synthesis_log['synthesis_time'] = time.time() - start_time

    return synthesized_circuit, sum(subp_times)  # time.time() - start_time + sum(subp_times)


# from qiskit.quantum_info import TwoQubitBasisDecomposer
# kak_decomposer = TwoQubitBasisDecomposer(CXGate())


def closest_unitary(matrix):
    svd_u, _, svd_v = np.linalg.svd(matrix)
    return svd_u.dot(svd_v)


def synthesize_unitary_gate(gate: dict, backend: Backend, allowed_dist=1e-10, lagre_block_penalty=2,
                            multi_process=True):
    start_time = time.time()
    subp_times = []

    gate_qubits: list = gate['qubits']

    if 'params' in gate:
        unitary_params = gate['params']
        unitary_params = (
                    unitary_params[0: 4 ** len(gate_qubits)] + 1j * unitary_params[4 ** len(gate_qubits):]).reshape(
            (2 ** len(gate_qubits), 2 ** len(gate_qubits)))
        unitary = to_unitary(unitary_params)
    elif 'unitary' in gate:
        unitary = gate['unitary']
    else:
        raise Exception('no params or unitary')

    # print(gate_qubits)
    if len(gate_qubits) <= 2:
        # start_time = time.time()
        qiskit_circuit = QuantumCircuit(len(gate_qubits))
        gate = Operator(unitary)
        qiskit_circuit.append(gate, list(range(len(gate_qubits))))

        with np.errstate(invalid="ignore"):
            qiskit_circuit = transpile(qiskit_circuit, optimization_level=3,
                                       basis_gates=backend.basis_gates)  # 会报det除以0的warning但是似乎问题不大

        layer2instructions, _, _, _, _ = get_layered_instructions(qiskit_circuit)
        synthesized_circuit, _, _ = qiskit_to_my_format_circuit(layer2instructions)

        '''特别注意qiskit和pennylane的大小端是不一样的'''
        for _layer_gates in synthesized_circuit:
            for _gate in _layer_gates:
                _gate['qubits'] = [len(gate_qubits) - _qubit - 1 for _qubit in
                    _gate['qubits']]  # print('2q', time.time() -  start_time)
    else:
        # 重新映射backend
        '''TODO: 三比特的可以换成qfast'''
        remap_topology = {
            gate_qubits.index(q1): [gate_qubits.index(q2) for q2 in backend.topology[q1] if q2 in gate_qubits] for q1 in
            gate_qubits}

        backend_3q = Backend(n_qubits=len(gate_qubits), topology=remap_topology, neighbor_info=None,
                             basis_single_gates=['u'], basis_two_gates=['cz'], divide=False, decoupling=False)

        # print('hah')

        if len(gate_qubits) <= 4:
            '''QFAST'''
            # try:
            from ..synthesis_baseline.baseline.qfast.synthesis import synthesis as qfast_synthesis

            qasm = qfast_synthesis(unitary, coupling_graph=backend_3q.coupling_map)
            qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
            qiskit_circuit = transpile(qiskit_circuit, optimization_level=3, basis_gates=backend.basis_gates,
                                       coupling_map=backend.coupling_map,
                                       initial_layout=[qubit for qubit in range(len(gate_qubits))])
            layer2instructions, _, _, _, _ = get_layered_instructions(qiskit_circuit)
            synthesized_circuit, _, _ = qiskit_to_my_format_circuit(layer2instructions)

            # print('synthesis', len(gate_qubits), 'costs' ,time.time() - start_time, 's')  # except:  #     traceback.print_exc()  #     synthesized_circuit, subp_time = synthesize(  #         unitary, backend_3q, allowed_dist, multi_process=multi_process, lagre_block_penalty=lagre_block_penalty, verbose=False)  #     subp_times.append(subp_time)

        # elif len(gate_qubits) == 3:
        #     # 三比特还是会报错
        #     try:
        #         synthesized_circuit = compile(
        #             unitary, max_synthesis_size=len(gate_qubits))
        #         qasm = synthesized_circuit.to('qasm')
        #         qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
        #         qiskit_circuit = transpile(qiskit_circuit, optimization_level=3, basis_gates=backend.basis_gates,
        #                                    coupling_map=backend.coupling_map, initial_layout=[qubit for qubit in range(len(gate_qubits))])
        #         layer2instructions, _, _, _, _ = get_layered_instructions(
        #             qiskit_circuit)
        #         synthesized_circuit, _, _ = qiskit_to_my_format_circuit(
        #             layer2instructions)
        #     except:
        #         traceback.print_exc()
        #         synthesized_circuit, subp_time = synthesize(
        #             unitary, backend_3q, allowed_dist, multi_process=multi_process, lagre_block_penalty=lagre_block_penalty, verbose=False)
        #         subp_times.append(subp_time)
        else:
            synthesized_circuit, subp_time = synthesize(unitary, backend_3q, allowed_dist, multi_process=multi_process,
                lagre_block_penalty=lagre_block_penalty, verbose=False,
                n_unitary_candidates=10)  # subp_times.append(subp_time)

    '''需要映射回去'''
    for _layer_gates in synthesized_circuit:
        for _gate in _layer_gates:
            _gate['qubits'] = [gate_qubits[_qubit] for _qubit in _gate['qubits']]

    return synthesized_circuit, time.time() - start_time  # + sum(subp_times)


@ray.remote
def synthesize_unitary_gate_remote(gate, backend: Backend, allowed_dist=5e-2, lagre_block_penalty=2,
                                   multi_process=True):
    return synthesize_unitary_gate(gate, backend, allowed_dist, lagre_block_penalty, multi_process)
