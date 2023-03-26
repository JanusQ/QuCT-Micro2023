'''
    给定一个酉矩阵U，将其转化为一组门的组合
'''
# from downstream.synthesis import

from sklearn.neighbors import NearestNeighbors
from circuit.formatter import get_layered_instructions, qiskit_to_my_format_circuit
from random import choice as random_choice
import inspect
from concurrent.futures._base import Future
from itertools import combinations
import itertools
import cloudpickle as pickle
from qiskit import QuantumCircuit, transpile
from circuit.formatter import layered_circuits_to_qiskit, to_unitary
from circuit.parser import qiskit_to_layered_circuits
from circuit import gen_random_circuits
from upstream import RandomwalkModel
from downstream.synthesis.pennylane_op_jax import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op_jax import layer_circuit_to_matrix
from sklearn.neural_network import MLPRegressor as DNN  # MLPClassifier as DNN
# from sklearn.decomposition import IncrementalPCA as PCA
from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from scipy.stats import unitary_group
import random
import copy
import ray
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
import time
from collections import defaultdict
from qiskit.quantum_info import Operator
import math
from utils.backend import Backend
from utils.ray_func import *
from random import randint
from bqskit import compile
import traceback

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
            gate['params'] = np.random.rand(len(gate['params'])) * 2 * np.pi
            # [
            #     random.random() * 2 * jnp.pi
            #     for param in gate['params']
            # ]
    return layer2gates


@jax.jit
def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    # optimized implementation
    return jnp.abs(1 - jnp.abs(jnp.sum(jnp.multiply(A, jnp.conj(B)))) / A.shape[0])


def create_unitary_gate(connect_qubits):
    if len(connect_qubits) == 1:
        return {
            'name': 'u',
            'qubits': list(connect_qubits),
            'params': np.random.rand(3) * 2 * np.pi,
        }
    else:
        n_connect_qubits = len(connect_qubits)
        return {
            'name': 'unitary',
            'qubits': list(connect_qubits),
            # 不知道有没有参数量更少的方法
            'params': np.random.rand((4**n_connect_qubits)*2),
        }


class PCA():
    def __init__(self, X, k=None, reduced_prop=None) -> None:
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
                if accumulated_eigen_value > target_eigen_values:
                    break
        if k is not None:
            accumulated_eigen_value = 0
            for eigen_value in sorted_eigen_values[:k]:
                accumulated_eigen_value += eigen_value
            reduced_prop = accumulated_eigen_value/sum_eigen_values

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
        reduced_matrices: jnp.array = vmap(
            pca_transform, in_axes=(0, None, None))(X, self.X_mean, self.V_k)
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



class SynthesisModelRandom():
    def __init__(self, backend) -> None:
        self.backend = backend
    
    def choose(self, U):
        backend: Backend = self.backend
        n_qubits = backend.n_qubits
        circuit_infos = gen_random_circuits(min_gate = 5 + n_qubits, max_gate=20 + n_qubits, backend = backend, gate_num_step = 10, n_circuits=1, two_qubit_gate_probs = [2,5], optimize=True, reverse=False)
        
        return [
            cirucit_info['layer2gates']
            for cirucit_info in circuit_infos
            # if len(cirucit_info['layer2gates']) < 5
        ]
    
class SynthesisModel():
    def __init__(self, upstream_model: RandomwalkModel, name=None):
        self.upstream_model = upstream_model
        self.backend = upstream_model.backend
        self.n_qubits = upstream_model.n_qubits
        self.name = name

        if name is not None:
            self.model_path = './temp_data/' + name + '/'
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

        self.pca_model = None
        self.U_to_vec_model = None

    def construct_data(self, circuits=None, multi_process=False):
        n_qubits = self.n_qubits
        backend = self.backend
        # n_devices = n_qubits

        if circuits is None:
            # max_gate = 1000  # 4**n_qubits
            # '''生成用于测试的模板电路'''
            # circuits = gen_random_circuits(min_gate=100, max_gate=max_gate, gate_num_step=max_gate//50, n_circuits=20,
            #                                two_qubit_gate_probs=[4, 8], backend=backend, reverse=False, optimize=True, multi_process=True)
            assert False

        def _gen_data(circuit_info, n_qubits, index):
            # print(index)
            self.upstream_model.vectorize(circuit_info)
            device_gate_vecs, Us = [], []
            layer2gates = circuit_info['layer2gates']

            
            for layer_index, layer_gates in enumerate(layer2gates):
                # print(index, layer_index)
                layer_gates = [
                    gate
                    for gate in layer_gates
                    if len(gate['qubits']) == 1
                ]
                if len(layer_gates) == 0:
                    continue
                for _ in range(10):
                    target_gate_index = randint(0, len(layer_gates)-1)
                    target_gate_index = layer_gates[target_gate_index]['id']
                    target_gate = circuit_info['gates'][target_gate_index]

                    '''TODO: 那些是jnp哪些是np要理一下'''
                    # TODO: 应该有些可以重复利用才对
                    U = layer_circuit_to_matrix(random_params(
                        layer2gates[layer_index:]), n_qubits)  # TODO: vmap
                    
                    # assert np.allclose(U.T.conj() @ U, I)
                    Us.append(U)

                    gate_vec = circuit_info['vecs'][target_gate_index].astype(np.int8)
                    device_vec = np.array(
                        [target_gate['qubits'][0]], dtype=np.int8)
                    device_gate_vecs.append(np.concatenate(
                        [device_vec, gate_vec], axis=0))
                    
            # if index % 100 == 0:
            #         print('finish', index)
                    
            return device_gate_vecs, Us

        @ray.remote
        def _gen_data_remote(circuit_info, n_qubits, index):
            return _gen_data(circuit_info, n_qubits, index)

        print('Start senerating Us -> Vs, totoal', len(circuits))
        futures = []
        for index, circuit_info in enumerate(circuits):
            if multi_process:
                future = _gen_data_remote.remote(circuit_info, n_qubits, index)
            else:
                future = _gen_data(circuit_info, n_qubits, index)
            futures.append(future)

        futures = wait(futures, show_progress = True)

        Vs, Us = [], []
        for index, future in enumerate(futures):
            Vs += future[0]
            Us += future[1]

        # 加[:100000]，不然现在有点太慢了
        Vs, Us = shuffle(Vs, Us)
        
        Vs = np.array(Vs, dtype=np.int8)  # [:100000]
        Us = np.array(Us, dtype=np.complex128)  # [:100000]

        # I = np.eye(2**n_qubits)
        # for U in Us:
        #     assert np.allclose(U.T.conj() @ U, I)

        print('len(Us) = ', len(Us), 'len(gate_vecs) = ', len(Vs))

        return Us, Vs

    def construct_model(self, data):
        print('Start construct model')
        start_time = time.time()

        Us, Vs = data

        self.pca_model = PCA(Us, reduced_prop=.7)
        output = self.pca_model.transform(Us)

        U_to_vec_model = DecisionTreeClassifier(splitter='random')
        U_to_vec_model.fit(output, Vs)

        self.U_to_vec_model = U_to_vec_model

        print(f'Finish construct model, costing {time.time()-start_time}s')

    def choose(self, U):
        pca_model: PCA = self.pca_model
        U_to_vec_model: DecisionTreeClassifier = self.U_to_vec_model
        upstream_model = self.upstream_model
        backend = self.backend
        n_qubits = self.n_qubits
        
        device_gate_vec = pca_model.transform(jnp.array([U]))
        device_gate_vec = U_to_vec_model.predict(device_gate_vec)[0]
        ''' 基于预测的gate vector来构建子电路'''
        device = int(device_gate_vec[0])
        gate_vec = device_gate_vec[1:]
        '''TODO: 万一有冲突怎么办'''
        '''TODO:有可能找到的子电路还会使dist变大，可能需要类似残差的结构'''
        layer2gates = upstream_model.reconstruct(
            device, gate_vec)  # TODO: 如果出现全是0的怎么办
        # print('reconstruct', layer_gates)
        
        return [layer2gates] #, device_gate_vec

        involved_qubits = []
        connected_sets = []  # [[q1, q2], [q3, q4]]  存了在path里面连接的比特组
        for layer_gates in layer2gates:
            for gate in layer_gates:
                involved_qubits += gate['qubits']
                set_indexs = []
                for _index, _connected_set in enumerate(connected_sets):
                    if any([_qubit in _connected_set for _qubit in gate['qubits']]):
                        set_indexs.append(_index)

                if len(set_indexs) == 0:
                    connected_sets.append(list(gate['qubits']))
                elif len(set_indexs) == 1:
                    _index = set_indexs[0]
                    connected_sets[_index] += gate['qubits']
                    connected_sets[_index] = list(
                        set(connected_sets[_index]))
                elif len(set_indexs) > 2:
                    new_connected_set = []
                    for _index in set_indexs:
                        new_connected_set += connected_sets[_index]
                        connected_sets.remove(connected_sets[_index])
                    connected_sets.append(new_connected_set)

        return [[
            create_unitary_gate(connected_set)
            for connected_set in connected_sets
        ]], device_gate_vec

    def save(self):
        with open(f'./temp_data/{self.name}.pkl', 'wb') as f:
            pickle.dump(self, f)
        return

    @staticmethod
    def load(name):
        with open(f'./temp_data/{name}.pkl', 'rb') as f:
            obj = pickle.load(f)
        return obj



def flatten(U):
    U = U.reshape(U.size)
    return np.concatenate([U.real, U.imag])

# @jax.jit
def unflatten(x):
    n_qubits = int(math.log2(x.size//2)//2)
    x_real = x[:x.size//2]
    x_imag = x[x.size//2:]
    return x_real.reshape((2**n_qubits, 2**n_qubits)) + 1j * x_imag.reshape((2**n_qubits, 2**n_qubits))

class SynthesisModelNN(SynthesisModel):
    
    @staticmethod
    def flatten_matrix_distance_squared(x1, x2):
        x1 = unflatten(x1)
        x2 = unflatten(x2)
        return matrix_distance_squared(x1, x2)
    
    def construct_model(self, data):
        print('Start construct model')
        start_time = time.time()

        Us, Vs = data

        Us = [flatten(U) for U in Us]
        
        self.nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree', metric= self.flatten_matrix_distance_squared, n_jobs = -1).fit(Us)
        self.Vs = Vs
        
        print(f'Finish construct model, costing {time.time()-start_time}s')

    
    def choose(self, U, verbose = False):
        nbrs: NearestNeighbors = self.nbrs
        upstream_model = self.upstream_model
        backend = self.backend
        n_qubits = self.n_qubits
        Vs = self.Vs

        U = flatten(U)
        
        distances, indices = nbrs.kneighbors([U])

        ''' 基于预测的gate vector来构建子电路'''

        if verbose:
            print('heuristic distance = ', distances)
        
        candidates = []
        # indices = indices[distances < .8]
        for index in indices[0]:
            device_gate_vec = Vs[index]
            device = int(device_gate_vec[0])
            gate_vec = device_gate_vec[1:]

            candidate = upstream_model.reconstruct(
                device, gate_vec)  # TODO: 如果出现全是0的怎么办
            candidates.append(candidate)
        
        return candidates

def find_parmas(n_qubits, layer2gates, U, lr=1e-1, max_epoch=1000, allowed_dist=1e-2, n_iter_no_change=10, no_change_tolerance=1e-2, random_params=True, verbose=False):
    np.random.RandomState()
    param_size = 0
    params = []
    for layer in layer2gates:
        for gate in layer:
            param_size += len(gate['params'])
            params += list(gate['params'])

    if random_params:
        params = jax.random.normal(jax.random.PRNGKey(
            np.random.randint(0, 100)), (param_size,), dtype=jnp.float64)
    else:
        params = jnp.array(params, dtype=jnp.float64)

    dev = qml.device("default.qubit", wires=n_qubits*2)
    '''lightning.qubit没法对unitary做优化'''
    # dev = qml.device("lightning.qubit", wires=n_qubits*2)
    '''TODO：hilbert_test相比于distance似乎会偏高一些，要计算一些换算公式'''
    '''TODO: 对比下hilbert_test和local hilbert_test，probs和expval的速度差异, 目前看来五比特以下差异不大 '''

    @qml.qnode(dev, interface="jax")  # interface如果没有jax就不会有梯度
    def hilbert_test(params, U):
        for q in range(n_qubits):
            qml.Hadamard(wires=q)

        for q in range(n_qubits):
            qml.CNOT(wires=[q, q+n_qubits])

        qml.QubitUnitary(U.conj(), wires=list(range(n_qubits)))
        '''TODO: jax.jit'''
        layer_circuit_to_pennylane_circuit(
            layer2gates, params=params, offest=n_qubits)

        for q in range(n_qubits):
            qml.CNOT(wires=[q, q+n_qubits])

        for q in range(n_qubits):
            qml.Hadamard(wires=q)

        '''hilbert_test get probs'''
        return qml.probs(list(range(n_qubits*2)))

        '''hilbert_test get expval'''
        # base = qml.PauliZ(0)
        # for q in range(1, n_qubits*2):
        #     base @= qml.PauliZ(q)
        # return  qml.expval(base)

    def cost_hst(params, U):
        probs = hilbert_test(params, U)
        return (1 - probs[0])

    opt = optax.adamw(learning_rate=lr)
    opt_state = opt.init(params)

    min_loss = 1e2
    best_params = None

    loss_decrease_history = []
    epoch = 0
    # start_time = time.time()
    while True:
        loss_value, gradient = jax.value_and_grad(
            cost_hst)(params, U)  # 需要约1s
        updates, opt_state = opt.update(gradient, opt_state, params)
        params = optax.apply_updates(params, updates)

        # if epoch % 10 == 0:
        if verbose:
            circuit_U: jnp.array = layer_circuit_to_matrix(
                layer2gates, n_qubits, params)
            matrix_dist = matrix_distance_squared(circuit_U, U)
            print('Epoch: {:5d} | Loss: {:.5f}  | Dist: {:.5f}'.format(
                epoch, loss_value, matrix_dist))

        loss_decrease_history.append(min_loss - loss_value)

        if min_loss > loss_value:
            min_loss = loss_value
            best_params = params

        if epoch < n_iter_no_change:
            loss_no_change = False
        else:
            loss_no_change = True
            for loss_decrement in loss_decrease_history[-n_iter_no_change:]:
                if loss_decrement > no_change_tolerance:
                    loss_no_change = False
        if loss_no_change or epoch > max_epoch or loss_value < allowed_dist:
            break

        epoch += 1
    # print('Epoch: {:5d} | Loss: {:.5f} '.format(epoch, loss_value))
    # print('time per itr = ', (time.time() - start_time)/epoch, 's')
    return best_params, min_loss, epoch


@ray.remote
def find_parmas_remote(n_qubits, layer2gates, U, max_epoch, allowed_dist, n_iter_no_change, no_change_tolerance, random_params):
    return find_parmas(n_qubits, layer2gates, U, max_epoch=max_epoch, allowed_dist=allowed_dist,
                       n_iter_no_change=n_iter_no_change, no_change_tolerance=no_change_tolerance, random_params=random_params)[:2]


def _optimize(total_layers, new_layers, n_optimized_layers, U, lr, n_iter_no_change, n_qubits, allowed_dist, now_dist):
    start_time = time.time()
    
    '''找到当前的最优参数'''
    total_layers = total_layers + new_layers
    unchanged_total_layer2gates, total_layers = total_layers[:-
                                                             n_optimized_layers], total_layers[-n_optimized_layers:]
    unchanged_part_matrix = layer_circuit_to_matrix(
        unchanged_total_layer2gates, n_qubits)

    '''TODO: 将连续的两个操作同一组qubits的unitary合并'''
    params, qml_dist, epoch_spent = find_parmas(n_qubits, total_layers, U @ unchanged_part_matrix.T.conj(), max_epoch=1000, allowed_dist=allowed_dist,
                                                n_iter_no_change=n_iter_no_change, no_change_tolerance=now_dist/10, random_params=False, lr=lr, verbose=False)
    # allowed_dist/100

    total_layers = assign_params(params, total_layers)
    circuit_U: jnp.array = layer_circuit_to_matrix(
        total_layers, n_qubits)

    circuit_U = circuit_U @ unchanged_part_matrix  # TODO: 放在电路里面可能算快一些

    total_layers = unchanged_total_layer2gates + total_layers

    now_dist = matrix_distance_squared(circuit_U, U)

    remained_U = U @ circuit_U.T.conj()  # 加上这个remained_U之后circuit_U就等于U了
    return remained_U, total_layers, qml_dist, now_dist, epoch_spent, time.time() - start_time


@ray.remote
def _optimize_remote(total_layers, new_layers, n_optimized_layers, invU, lr, n_iter_no_change, n_qubits, allowed_dist, now_dist):
    return _optimize(total_layers, new_layers, n_optimized_layers, invU, lr, n_iter_no_change, n_qubits, allowed_dist, now_dist)


def synthesize(U, backend: Backend, allowed_dist=1e-5, heuristic_model: SynthesisModel = None, multi_process=True, 
               verbose=False, lagre_block_penalty=2, heuristic_ratio = .5, timeout = 5*3600):
    start_time = time.time()
    subp_times = []
    
    if verbose:
        print('start synthesis')

    # multi_process = True
    n_qubits = int(math.log2(len(U)))
    if n_qubits < 3: # TODO: 等于 3
        gate = {
            'qubits': list(range(n_qubits)),
            'unitary': U,
        }
        return synthesize_unitary_gate(gate, backend, allowed_dist, lagre_block_penalty, multi_process)

    I = jnp.eye(2**n_qubits)

    if matrix_distance_squared(U, I) <= allowed_dist:
        return QuantumCircuit(n_qubits)

    use_heuristic = heuristic_model is not None

    if use_heuristic:
        assert heuristic_model.backend == backend

    now_dist = 1
    remained_U = U
    total_layers = []
    iter_count = 0

    min_dist = 1e2
    # former_device_gate_vec = None  

    heuristic_takeeffact_count = 0 # 记录下起heuristic作用的数量
    while now_dist > allowed_dist:
        if time.time() - start_time > timeout: break
        
        '''TODO: 有些电路要是没有优化动就不插了'''
        canditate_layers = []

        device_gate_vec = None
        if use_heuristic: # and now_dist > .5:
        # if use_heuristic and heuristic_ratio > random() and iter_count != 0 and now_dist > 0.3:
            '''相似的酉矩阵矩阵可能会导致重复插入相似的电路'''
            # canditates, device_gate_vec = heuristic_model.choose(remained_U,)
            # canditate_layers.append(new_layers)
            canditates = heuristic_model.choose(remained_U, verbose = True)
            canditate_layers += canditates
            # print('use heuristic')

        if True:
        # if not use_heuristic or device_gate_vec is None or (former_device_gate_vec is not None and jnp.allclose(device_gate_vec, former_device_gate_vec)):
            '''explore最优的子电路'''
            # 单比特直接搞一层算了
            new_layer = []
            for _q in range(n_qubits):
                new_layer.append(create_unitary_gate([_q]))
            canditate_layers.append([new_layer])

            # max_n_subset_qubits = max([3, int(n_qubits//2)]) + 1
            # max_n_subset_qubits = max([3, int(n_qubits//3*2)]) + 1
            '''TODO: 这个函数是要试几次的'''
            if iter_count != 0:
                max_n_subset_qubits = max([3, n_qubits - 1])
                max_n_subset_qubits_step = max([1, (max_n_subset_qubits-3)//2])
                # TODO: 可能后面要设置成关于n_qubits的函数
                for n_subset_qubits in [2, 3] + list(range(max_n_subset_qubits, 3, -max_n_subset_qubits_step)):
                    if n_qubits <= n_subset_qubits:  # 不会生成大于自己的比特数的子电路
                        continue

                    for subset_qubits in backend.get_connected_qubit_sets(n_subset_qubits):
                        # for _ in range(3): # 多来两次会不会效果好一些, 不会
                        canditate_layers.append(
                            [[create_unitary_gate(subset_qubits)]])

        # canditate_layers = list(itertools.product(canditate_layers, canditate_layers))
        futures = []
        former_device_gate_vec = device_gate_vec

        candiate_selection_start_time = time.time()
        for candidate_layer in canditate_layers:
            # TODO: 每隔一段时间进行一次finetune
            if (iter_count+1) % 10 == 0 or now_dist < 1e-2:  # TODO: 需要尝试的超参数
                n_optimized_layers = len(total_layers) + len(candidate_layer)
            else:
                n_optimized_layers = 10 + len(candidate_layer)  # TODO: 需要尝试的超参数

            if now_dist < 1e-2:
                n_iter_no_change = 10
            else:
                n_iter_no_change = 5

            lr = max([now_dist/5, 1e-2])  # TODO: 需要尝试的超参数
            if multi_process:
                futures.append(_optimize_remote.remote(
                    total_layers, candidate_layer, n_optimized_layers, U, lr, n_iter_no_change, n_qubits, allowed_dist, now_dist))
            else:
                futures.append(
                    _optimize(total_layers, candidate_layer, n_optimized_layers, U, lr, n_iter_no_change, n_qubits, allowed_dist, now_dist))

        futures = wait(futures)
        max_dist_decrement = 0
        _min_dist = None
        
        heuristic_takeeffact = False
        for future in futures:
            c_remained_U, c_total_layers, c_qml_dist, c_now_dist, c_epoch, subp_time = future
            subp_times.append(subp_time)
            
            '''TODO: 相似的距离优先比特数小的'''
            dist_decrement = now_dist - c_now_dist
            subset_qubits = c_total_layers[-1][0]['qubits']
            subset_penalty = lagre_block_penalty**(len(subset_qubits)-1)
            if c_total_layers[-1][0]['name'] != 'unitary':
                subset_penalty = sum([len([gate for gate in layer if len(gate['qubits']) > 2]) for layer in c_total_layers]) + 1#lagre_block_penalty
            
            if dist_decrement > 0:
                # 比特数多虽然能够减少的快，但是代价大
                dist_decrement /= subset_penalty
            else:
                dist_decrement *= subset_penalty

            candidate_layer = c_total_layers[-1][0]

            if verbose:
                print(candidate_layer['name'], candidate_layer['qubits'],
                      dist_decrement, now_dist, c_now_dist, subset_penalty, c_epoch)

            if dist_decrement > max_dist_decrement:
                max_dist_decrement = dist_decrement
                total_layers = c_total_layers
                remained_U = c_remained_U
                _min_dist = c_now_dist
                
                if c_total_layers[-1][0]['name'] != 'unitary' and len(c_total_layers) != 1:
                    heuristic_takeeffact = True
                else:
                    heuristic_takeeffact = False
        
        if heuristic_takeeffact:
            if verbose:
                print('heuristic takes effact')
            heuristic_takeeffact_count += 1
                
        if verbose:
            print('candiate selection cost', time.time() -
                  candiate_selection_start_time)

        now_dist = _min_dist
        # print('choose', total_layers[-1][0]['name'], total_layers[-1][0]['qubits'], now_dist)

        if max_dist_decrement == 0:
            print('Warning: no improvement in the exploration')
            future = random_choice(futures)
            remained_U, total_layers, c_qml_dist, now_dist, epoch_spent = future

        if verbose:
            # 只显示一部分就好了，不然太乱了
            qiskit_circuit = layered_circuits_to_qiskit(
                n_qubits, total_layers[-6:], barrier=False)   # 0.00287s
            print(qiskit_circuit)
            print('iter_count=', iter_count, 'now_dist=', now_dist, '\n')

        if now_dist < min_dist:
            min_dist = now_dist

        iter_count += 1

    # multi_process = False

    if verbose:
        print('synthesize decomposed unitaries')

    # 大于3比特的先放到别的进程去跑
    for layer_gates in total_layers:
        for gate in layer_gates:
            gate_name = gate['name']
            if gate_name == 'unitary':
                if len(gate['qubits']) >= 3:
                    if multi_process:
                        gate['synthesis_result'] = synthesize_unitary_gate_remote.remote(
                            gate, backend, allowed_dist=allowed_dist/10, lagre_block_penalty=4 if len(gate['qubits']) == 3 else 2, multi_process=multi_process)
                    else:
                        gate['synthesis_result'] = synthesize_unitary_gate(
                            gate, backend, allowed_dist=allowed_dist/10, lagre_block_penalty=4 if len(gate['qubits']) == 3 else 2, multi_process=multi_process)

    # 两比特的自己跑还快一些
    for layer_gates in total_layers:
        for gate in layer_gates:
            gate_name = gate['name']
            if gate_name == 'unitary':
                if len(gate['qubits']) == 2:
                    gate['synthesis_result'] = synthesize_unitary_gate(
                        gate, backend, allowed_dist=allowed_dist/10, multi_process=multi_process)

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
                unitary_circuit = wait(gate['synthesis_result'])
                new_total_layers += unitary_circuit
            else:
                raise Exception('unknown', gate)

    '''最后finetune一下，反而变差了感觉像是hl_test对global phase也敏感的原因'''

    if verbose:
        print('compose')

    qiskit_circuit: QuantumCircuit = layered_circuits_to_qiskit(
        n_qubits, new_total_layers, barrier=False)

    qiskit_circuit = transpile(qiskit_circuit, coupling_map=backend.coupling_map, optimization_level=3,
                               basis_gates=backend.basis_gates, initial_layout=[qubit for qubit in range(n_qubits)])
    layer2instructions, _, _, _, _ = get_layered_instructions(qiskit_circuit)
    synthesized_circuit, _, _ = qiskit_to_my_format_circuit(layer2instructions)

    if verbose: 
        print('heuristic takes effect in', heuristic_takeeffact_count / iter_count * 100, '%')
    return synthesized_circuit, time.time() - start_time + sum(subp_times)

# from qiskit.quantum_info import TwoQubitBasisDecomposer
# kak_decomposer = TwoQubitBasisDecomposer(CXGate())


def closest_unitary(matrix):
    svd_u, _, svd_v = np.linalg.svd(matrix)
    return svd_u.dot(svd_v)


def synthesize_unitary_gate(gate: dict, backend: Backend, allowed_dist=1e-10, lagre_block_penalty=2, multi_process=True):
    start_time = time.time()
    subp_times = []
    
    gate_qubits: list = gate['qubits']
    
    if 'params' in gate:
        unitary_params = gate['params']
        unitary_params = (unitary_params[0: 4**len(gate_qubits)] + 1j * unitary_params[4**len(
            gate_qubits):]).reshape((2**len(gate_qubits), 2**len(gate_qubits)))
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

        layer2instructions, _, _, _, _ = get_layered_instructions(
            qiskit_circuit)
        synthesized_circuit, _, _ = qiskit_to_my_format_circuit(
            layer2instructions)

        '''特别注意qiskit和pennylane的大小端是不一样的'''
        for _layer_gates in synthesized_circuit:
            for _gate in _layer_gates:
                _gate['qubits'] = [
                    len(gate_qubits) - _qubit - 1
                    for _qubit in _gate['qubits']
                ]
        # print('2q', time.time() -  start_time)
    else:
        # 重新映射backend
        '''TODO: 三比特的可以换成qfast'''
        remap_topology = {
            gate_qubits.index(q1): [
                gate_qubits.index(q2)
                for q2 in backend.topology[q1]
                if q2 in gate_qubits
            ]
            for q1 in gate_qubits
        }

        backend_3q = Backend(n_qubits=len(gate_qubits), topology=remap_topology, neighbor_info=None, basis_single_gates=['u'],
                             basis_two_gates=['cz'], divide=False, decoupling=False)

        # print('hah')

        if len(gate_qubits) == 4:
            '''QFAST'''
            from qfast.synthesis import synthesize as qfast_synthesis

            qasm = qfast_synthesis(
                unitary, coupling_graph=backend_3q.coupling_map)
            qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
            qiskit_circuit = transpile(qiskit_circuit, optimization_level=3, basis_gates=backend.basis_gates,
                                       coupling_map=backend.coupling_map, initial_layout=[qubit for qubit in range(len(gate_qubits))])
            layer2instructions, _, _, _, _ = get_layered_instructions(
                qiskit_circuit)
            synthesized_circuit, _, _ = qiskit_to_my_format_circuit(
                layer2instructions)
        elif len(gate_qubits) == 3:
            try:
                synthesized_circuit = compile(
                    unitary, max_synthesis_size=len(gate_qubits))
                qasm = synthesized_circuit.to('qasm')
                qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
                qiskit_circuit = transpile(qiskit_circuit, optimization_level=3, basis_gates=backend.basis_gates,
                                           coupling_map=backend.coupling_map, initial_layout=[qubit for qubit in range(len(gate_qubits))])
                layer2instructions, _, _, _, _ = get_layered_instructions(
                    qiskit_circuit)
                synthesized_circuit, _, _ = qiskit_to_my_format_circuit(
                    layer2instructions)
            except:
                traceback.print_exc()
                synthesized_circuit, subp_time = synthesize(
                    unitary, backend_3q, allowed_dist, multi_process=multi_process, lagre_block_penalty=lagre_block_penalty, verbose=False)
                subp_times.append(subp_time)
        else:
            synthesized_circuit, subp_time = synthesize(
                unitary, backend_3q, allowed_dist, multi_process=multi_process, lagre_block_penalty=lagre_block_penalty, verbose=False)
            subp_times.append(subp_time)

    '''需要映射回去'''
    for _layer_gates in synthesized_circuit:
        for _gate in _layer_gates:
            _gate['qubits'] = [
                gate_qubits[_qubit]
                for _qubit in _gate['qubits']
            ]
            
    return synthesized_circuit, time.time() - start_time + sum(subp_times)


@ray.remote
def synthesize_unitary_gate_remote(gate, backend: Backend, allowed_dist=5e-2, lagre_block_penalty=2, multi_process=True):
    return synthesize_unitary_gate(gate, backend, allowed_dist, lagre_block_penalty, multi_process)
