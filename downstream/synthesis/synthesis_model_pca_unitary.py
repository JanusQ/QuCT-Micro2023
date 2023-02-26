'''
    给定一个酉矩阵U，将其转化为一组门的组合
'''
# from downstream.synthesis import

from itertools import combinations
import cloudpickle as pickle
from qiskit import QuantumCircuit, transpile
from circuit.formatter import layered_circuits_to_qiskit, to_unitary
from circuit.parser import qiskit_to_layered_circuits
from circuit import gen_random_circuits
from upstream import RandomwalkModel
from downstream.synthesis.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op import layer_circuit_to_matrix
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

from utils.backend import Backend


config.update("jax_enable_x64", True)

''' 换了find_parmas的方法变得更快了'''


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
    return


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
            gate['params'] = [
                random.random() * 2 * jnp.pi
                for param in gate['params']
            ]
    return layer2gates


@jax.jit
def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    # optimized implementation
    return jnp.abs(1 - jnp.abs(jnp.sum(jnp.multiply(A, jnp.conj(B)))) / A.shape[0])


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
            _transform, in_axes=(0, None, None))(X, self.X_mean, self.V_k)
        return reduced_matrices.astype(jnp.float64)


@jax.jit
def _transform(m, X_mean, V_k):
    # 对每个输入矩阵都做降维，并且保持距离相似性
    m_centered = m - X_mean[jnp.newaxis, :]
    m_reduced = jnp.dot(m_centered, V_k)
    # 对降维后的矩阵做幺正正交化，保证在降维前后距离有相似性
    q, r = jnp.linalg.qr(m_reduced)
    q = q.reshape(q.size)
    q = jnp.concatenate([q.imag, q.real], dtype=jnp.float64)
    return q  # m_reduced #q


def find_parmas(n_qubits, layer2gates, U, lr=1e-1, max_epoch=1000, allowed_dist=1e-2, n_iter_no_change=10, no_change_tolerance=1e-2, random_params=True):
    param_size = 0
    params = []
    for layer in layer2gates:
        for gate in layer:
            param_size += len(gate['params'])
            params += list(gate['params'])

    if random_params:
        params = jax.random.normal(jax.random.PRNGKey(
            random.randint(0, 100)), (param_size,), dtype=jnp.float64)
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
        # qml.QubitUnitary(U, wires=list(range(n_qubits, 2*n_qubits)))
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
    former_loss = 0
    epoch = 0
    # start_time = time.time()
    while True:
        loss_value, gradient = jax.value_and_grad(
            cost_hst)(params, U)  # 需要约1s
        updates, opt_state = opt.update(gradient, opt_state, params)
        params = optax.apply_updates(params, updates)

        # if epoch % 10 == 0:
        # print('Epoch: {:5d} | Loss: {:.5f} '.format(epoch, loss_value))

        if min_loss > loss_value:
            min_loss = loss_value
            best_params = params

        loss_decrease_history.append(former_loss-min_loss)
        former_loss = loss_value
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
    return best_params, min_loss


@ray.remote
def find_parmas_remote(n_qubits, layer2gates, U, max_epoch, allowed_dist, n_iter_no_change, no_change_tolerance, random_params):
    return find_parmas(n_qubits, layer2gates, U, max_epoch=max_epoch, allowed_dist=allowed_dist,
                       n_iter_no_change=n_iter_no_change, no_change_tolerance=no_change_tolerance, random_params=random_params)


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
        n_devices = n_qubits

        if circuits is None:
            max_gate = 1000  # 4**n_qubits
            '''生成用于测试的模板电路'''
            circuits = gen_random_circuits(min_gate=100, max_gate=max_gate, gate_num_step=max_gate//50, n_circuits=20,
                                           two_qubit_gate_probs=[4, 8], backend=backend, reverse=False, optimize=True, multi_process=True)

        def _gen_data(circuit_info, n_qubits):
            device_gate_vecs, Us = [], []
            layer2gates = circuit_info['layer2gates']
            # print(layer2gates)
            for layer_index, layer_gates in enumerate(layer2gates):
                layer_gates = [
                    gate
                    for gate in layer_gates
                    if len(gate['qubits']) == 1
                ]
                if len(layer_gates) == 0:
                    continue
                for _ in range(30):
                    target_gate_index = random.randint(0, len(layer_gates)-1)
                    target_gate_index = layer_gates[target_gate_index]['id']
                    target_gate = circuit_info['gates'][target_gate_index]

                    gate_vec = circuit_info['vecs'][target_gate_index].astype(
                        jnp.int8)

                    _layer2gates = random_params(layer2gates[layer_index:])
                    # TODO: 应该有些可以重复利用才对
                    U = layer_circuit_to_matrix(
                        _layer2gates, n_qubits)  # TODO: vmap

                    '''TODO: 那些是jnp哪些是np要理一下'''
                    Us.append(U)

                    device_vec = np.array(
                        [target_gate['qubits'][0]], dtype=jnp.int8)
                    device_gate_vecs.append(np.concatenate(
                        [device_vec, gate_vec], axis=0))

            return device_gate_vecs, Us

        @ray.remote
        def _gen_data_remote(circuit_info, n_qubits):
            return _gen_data(circuit_info, n_qubits)

        print('start generate Us -> gate vectos')
        futures = []
        for index, circuit_info in enumerate(circuits):
            self.upstream_model.vectorize(circuit_info)
            if multi_process:
                future = _gen_data_remote.remote(circuit_info, n_qubits)
            else:
                future = _gen_data(circuit_info, n_qubits)
            # future = _gen_data(circuit_info, n_qubits)
            futures.append(future)

        device_gate_vecs, Us = [], []
        for index, future in enumerate(futures):
            if index % 1000 == 0:
                print('finished rate =', index, '/', len(futures))
            if multi_process:
                future = ray.get(future)
            device_gate_vecs += future[0]
            Us += future[1]

        # 加[:100000]，不然现在有点太慢了
        device_gate_vecs, Us = shuffle(device_gate_vecs, Us)
        device_gate_vecs = np.array(
            device_gate_vecs, dtype=np.int8)  # [:100000]
        Us = np.array(Us, dtype=np.float64)  # [:100000]

        print('len(Us) = ', len(Us), 'len(gate_vecs) = ', len(device_gate_vecs))

        self.data = Us, device_gate_vecs
        return self.data

    def construct_model(self):
        print('start construct model')
        start_time = time.time()
        # n_qubits = self.n_qubits
        # backend = self.backend
        # upstream_model = self.upstream_model

        Us, gate_vecs = self.data

        self.pca_model = PCA(Us, reduced_prop=.7)
        output = self.pca_model.transform(Us)

        U_to_vec_model = DecisionTreeClassifier()
        U_to_vec_model.fit(output, gate_vecs)

        self.U_to_vec_model = U_to_vec_model
        # self.data = None
        print(f'finish construct model, cost {time.time()-start_time}s')
        return

    def synthesize(self, U, allowed_dist=5e-2, use_heuristic=True, multi_process=True):
        print('start synthesis')
        n_qubits = self.n_qubits
        pca_model = self.pca_model
        U_to_vec_model: DecisionTreeClassifier = self.U_to_vec_model

        upstream_model = self.upstream_model
        backend = self.backend

        inv_U = U.T.conj()
        now_U = U
        I = jnp.eye(2**n_qubits)
        circuit_U = I
        # if nowU is not identity
        now_dist = 1

        total_layer2gates = []
        iter_count = 0

        min_dist = 1e2
        qml_dist = 1
        '''TODO: 设置finetune和不finetune'''
        # no_change_tolerance = 1e-3
        finetune_tolerance = 1e-3
        coarsetune_tolerance = 1e-2
        # endless_detector = 0
        former_device_gate_vec = None

        def create_unitary_gate(connect_qubits):
            if len(connect_qubits) == 1:
                return {
                    'name': 'u',
                    'qubits': connect_qubits,
                    'params': np.random.rand(3) * 2 * np.pi,
                }
            else:
                n_connect_qubits = len(connect_qubits)
                return {
                    'name': 'unitary',
                    'qubits': connect_qubits,
                    # 不知道有没有参数量更少的方法
                    'params': np.random.rand((4**n_connect_qubits)*2),
                }

        while now_dist > allowed_dist:
            inv_now_U = now_U.T.conj()

            if use_heuristic:
                device_gate_vec = pca_model.transform(jnp.array([inv_now_U]))
                device_gate_vec = U_to_vec_model.predict(device_gate_vec)[0]

            '''相似的酉矩阵矩阵可能会导致重复插入相似的电路'''

            '''有些电路要是没有优化动就不插了'''

            if use_heuristic and (former_device_gate_vec is None or jnp.allclose(device_gate_vec, former_device_gate_vec)):
                former_device_gate_vec = coarsetune_tolerance  # TODO: 什么时候更新要确认下

                ''' 基于预测的gate vector来构建子电路'''
                device = int(device_gate_vec[0])
                gate_vec = device_gate_vec[1:]

                gate_vec = device_gate_vec[n_qubits:]
                '''TODO: 万一有冲突怎么办'''
                '''TODO:有可能找到的子电路还会使dist变大，可能需要类似残差的结构'''
                layer2gates = upstream_model.reconstruct(
                    device, gate_vec)  # TODO: 如果出现全是0的怎么办
                # print('reconstruct', layer_gates)

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

                new_layer2gates = [[
                    create_unitary_gate(connected_set)
                    for connected_set in connected_sets
                ]]

            else:
                '''explore最优的子电路'''
                futures = []
                for n_subset_qubits in (1, 2, 3):
                    if n_qubits == 3 and n_subset_qubits == 3:  # 三比特的时候就不试
                        continue

                    for subset_qubits in combinations(list(range(n_qubits)), n_subset_qubits):
                        subset_qubits = list(subset_qubits)
                        # total_layer2gates
                        layer_gates = [create_unitary_gate(subset_qubits)]
                        if multi_process:
                            futures.append((
                                subset_qubits, layer_gates,
                                find_parmas_remote.remote(n_qubits, [layer_gates], inv_now_U, max_epoch=1000, allowed_dist=allowed_dist,
                                                          n_iter_no_change=5, no_change_tolerance=coarsetune_tolerance, random_params=True)
                            ))
                        else:
                            params, dist = find_parmas(n_qubits, [layer_gates], inv_now_U, max_epoch=1000, allowed_dist=allowed_dist,
                                                       n_iter_no_change=5, no_change_tolerance=coarsetune_tolerance, random_params=True)  # 有时候甚至会长，因为优化的问题
                            futures.append(
                                (subset_qubits, layer_gates, (params, dist)))

                max_dist_decrement = -1
                new_layer2gates = None
                for future in futures:
                    subset_qubits, layer_gates, future = future
                    if multi_process:
                        params, dist = ray.get(future)
                    else:
                        params, dist = future
                    '''TODO: 相似的距离优先比特数小的'''
                    dist_decrement = qml_dist - dist  # 这里用的是hilbert_test的dis和matrix distance是不一样大，似乎要大一些
                    if dist_decrement > 0:
                        # 比特数多虽然能够减少的快，但是代价大
                        dist_decrement /= 1.2**len(subset_qubits)
                    else:
                        dist_decrement *= 1.2**len(subset_qubits)
                    print(subset_qubits, dist_decrement)
                    if dist_decrement > max_dist_decrement:
                        max_dist_decrement = dist_decrement
                        new_layer2gates = assign_params(params, [layer_gates])

                if new_layer2gates is None:
                    print('Warning: no improvement in the exploration')
                    future = random.choice(futures)
                    subset_qubits, layer_gates, future = future
                    if multi_process:
                        params, dist = ray.get(future)
                    else:
                        params, dist = future
                    new_layer2gates = assign_params(params, [layer_gates])

            total_layer2gates = total_layer2gates + new_layer2gates
            unchanged_total_layer2gates, total_layer2gates = total_layer2gates[
                :-10], total_layer2gates[-10:]
            unchanged_part_matrix = layer_circuit_to_matrix(
                unchanged_total_layer2gates, n_qubits)

            '''TODO: 将连续的两个操作同一组qubits的unitary合并'''

            params, qml_dist = find_parmas(n_qubits, total_layer2gates, inv_U @ unchanged_part_matrix.T.conj(), max_epoch=1000, allowed_dist=allowed_dist,
                                           n_iter_no_change=5, no_change_tolerance=coarsetune_tolerance, random_params=False)  # 有时候甚至会长，因为优化的问题

            total_layer2gates = assign_params(params, total_layer2gates)
            circuit_U: jnp.array = layer_circuit_to_matrix(
                total_layer2gates, n_qubits)

            circuit_U = circuit_U @ unchanged_part_matrix  # TODO: 放在电路里面可能算快一些

            total_layer2gates = unchanged_total_layer2gates + total_layer2gates

            qiskit_circuit = layered_circuits_to_qiskit(
                n_qubits, total_layer2gates, barrier=False)   # 0.00287s
            print(qiskit_circuit)

            now_U = circuit_U @ U

            now_dist = matrix_distance_squared(now_U, I)

            if now_dist < min_dist:
                min_dist = now_dist
            print('iter_count=', iter_count, 'now_dist=', now_dist, '\n')

            iter_count += 1
        '''TODO: 最后加一个finetune的步骤'''

        

        for layer_gates in total_layer2gates:
            for gate in layer_gates:
                gate_name = gate['name']
                if gate_name == 'unitary':
                    # if multi_process:
                    #     gate['synthesis_result'] = synthesize_1q_2q_remote.remote(gate, backend, allowed_dist=allowed_dist/10, multi_process = multi_process)
                    # else:
                    gate['synthesis_result'] = synthesize_1q_2q(gate, backend, allowed_dist=allowed_dist/10, multi_process = multi_process)

        new_total_layer2gates = []
        for layer_gates in total_layer2gates:
            for gate in layer_gates:
                gate_qubits: list = gate['qubits']
                gate_name = gate['name']
                if gate_name in ('u', 'cz', 'cx'):
                    new_total_layer2gates.append([gate])
                elif gate_name == 'unitary':
                    # if multi_process:
                    #     unitary_circuit = ray.get(gate['synthesis_result'])
                    # else:
                    unitary_circuit = gate['synthesis_result']
                    new_total_layer2gates += unitary_circuit
                else:
                    raise Exception('unknown', gate)

        synthesized_circuit: QuantumCircuit = layered_circuits_to_qiskit(
            n_qubits, new_total_layer2gates, barrier=False)

        synthesized_circuit = transpile(synthesized_circuit, coupling_map=backend.coupling_map, optimization_level=3, basis_gates=[
                                            'u', 'cz'], initial_layout=[qubit for qubit in range(n_qubits)])
        '''TODO: 当前的电路要逆一下才是U的电路'''
        return synthesized_circuit.inverse()

    def save(self):
        with open(f'./temp_data/{self.name}.pkl', 'wb') as f:
            pickle.dump(self, f)
        return

    @staticmethod
    def load(name):
        with open(f'./temp_data/{name}.pkl', 'rb') as f:
            obj = pickle.load(f)
        return obj

# check下是不是本身就有了
# from qiskit.quantum_info import TwoQubitBasisDecomposer
# kak_decomposer = TwoQubitBasisDecomposer(CXGate())


def synthesize_1q_2q(gate, backend: Backend, allowed_dist=5e-2, multi_process = True):
    gate_qubits: list = gate['qubits']
    gate_name = gate['name']
    unitary_params = gate['params']
    unitary_params = (unitary_params[0: 4**len(gate_qubits)] + 1j * unitary_params[4**len(
        gate_qubits):]).reshape((2**len(gate_qubits), 2**len(gate_qubits)))
    unitary = to_unitary(unitary_params)
    if len(gate_qubits) == 2:
        circuit = QuantumCircuit(len(gate_qubits))
        gate = Operator(unitary)
        circuit.append(gate, list(range(len(gate_qubits))))
        circuit = transpile(circuit, optimization_level=3, basis_gates=[
            'u', 'cz'])
        unitary_circuit = qiskit_to_layered_circuits(
            circuit, divide=False, decoupling=False)['layer2gates']

    elif len(gate_qubits) == 3:
        # 重新映射backend
        remap_topology = {
            gate_qubits.index(q1): [
                gate_qubits.index(q2)
                for q2 in backend.topology[q1]
                if q2 in gate_qubits
            ]
            for q1 in gate_qubits
        }
        # TODO: 怎么不连在一起的也有unitary

        backend_3q = Backend(n_qubits=3, topology=remap_topology, neighbor_info=None, basis_single_gates=['u'],
                                basis_two_gates=['cz'], divide=False, decoupling=False)
        upstream_model_3q = RandomwalkModel(1, 20, backend = backend_3q)
        unitary_circuit = SynthesisModel(upstream_model_3q).synthesize(
            unitary, allowed_dist, use_heuristic=False, multi_process=multi_process)
        unitary_circuit = qiskit_to_layered_circuits(
            unitary_circuit, divide=False, decoupling=False)['layer2gates']
        
    # 需要重新映射
    for _layer_gates in unitary_circuit:
        for _gate in _layer_gates:
            _gate['qubits'] = [
                gate_qubits[_qubit]
                for _qubit in _gate['qubits']
            ]
    return unitary_circuit

@ray.remote
def synthesize_1q_2q_remote(gate, backend: Backend, allowed_dist=5e-2, multi_process = True):
    return synthesize_1q_2q(gate, backend, allowed_dist, multi_process)