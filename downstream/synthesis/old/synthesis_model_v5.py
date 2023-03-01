'''
    给定一个酉矩阵U，将其转化为一组门的组合
'''
# from downstream.synthesis import

import cloudpickle as pickle
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend
from circuit import gen_random_circuits
from upstream import RandomwalkModel
from downstream.synthesis.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op import layer_circuit_to_matrix
from downstream.synthesis.neural_network import NeuralNetworkModel
from sklearn.neural_network import MLPRegressor as DNN  # MLPClassifier as DNN

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

config.update("jax_enable_x64", True)

''' 换了find_parmas的方法变得更快了'''
''' 搞清楚了新加的层在计算now_U之间的关系'''

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
    # return


def random_params(layer2gates):
    layer2gates = copy.deepcopy(layer2gates)
    for layer in layer2gates:
        for gate in layer:
            gate['params'] = [
                random.random() * 2 * jnp.pi
                for param in gate['params']
            ]
    return layer2gates


'''linear mapping'''


def step_function(x):
    return jnp.where(x > 0, 1.0, 0.0)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def final_mapping(mapping, x):
    # print(mapping)
    return sigmoid(mapping['W'] * x + mapping['B'])


def final_mapping_cost(mapping, x, y):
    predictions = final_mapping(mapping, x)
    return optax.l2_loss(predictions, y).sum()


def final_mapping_cost_batch(mapping, X, Y):
    return vmap(final_mapping_cost, in_axes=(None, 0, 0))(mapping, X, Y).sum()


@jax.jit
def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    # optimized implementation
    return jnp.abs(1 - jnp.abs(jnp.sum(jnp.multiply(A, jnp.conj(B)))) / A.shape[0])


def transformU(U):
    '''转成神经网络里面的格式'''
    U = U.reshape(U.size)
    U = jnp.concatenate([U.imag, U.real])
    U = jnp.array(U, dtype=jnp.float64)
    return U


class SynthesisModel():
    def __init__(self, upstream_model: RandomwalkModel, name=None):
        self.upstream_model = upstream_model
        self.backend = upstream_model.backend
        self.n_qubits = upstream_model.n_qubits
        self.name = name

        # self.action_Q = None  # 每个device一个
        # U.size -> len(path_table)
        # , reward = mean(output)

        return

    # 准备用强化学习来查找
    # def get_action(self, state) -> jnp.array:
    #     return

    # def learn(self, state, action, reward, learning_rate):
    #     '''
    #         state: target unitary
    #         action: gate vector
    #     '''
    #     return

    def construct_data(self, circuits=None, multi_process=False):
        # pkl_name = f'./temp_data/sys_data_{self.name}.pkl'
        # data = pkl_load(pkl_name)

        # if data is not None:
        #     self.data = data
        #     return data

        n_qubits = self.n_qubits
        backend = self.backend

        if circuits is None:
            max_gate = 1000  # 4**n_qubits
            '''生成用于测试的模板电路'''
            circuits = gen_random_circuits(min_gate=100, max_gate=max_gate, gate_num_step=max_gate//50, n_circuits=20,
                                           two_qubit_gate_probs=[4, 8], backend=backend, reverse=False, optimize=True, multi_process=True)

        def _gen_data(circuit_info, n_qubits):
            gate_vecs, Us = [], []
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
                for _ in range(10):
                    target_gate_index = random.randint(0, len(layer_gates)-1)
                    target_gate_index = layer_gates[target_gate_index]['id']
                    target_gate = circuit_info['gates'][target_gate_index]

                    gate_vec = circuit_info['vecs'][target_gate_index]

                    _layer2gates = random_params(layer2gates[layer_index:])
                    # TODO: 应该有些可以重复利用才对
                    U = layer_circuit_to_matrix(
                        _layer2gates, n_qubits)  # TODO: vmap
                    U = transformU(U)

                    Us.append(U)
                    gate_vecs.append(np.concatenate(
                        [jnp.array(target_gate['qubits']), gate_vec], axis=0))

            return gate_vecs, Us

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

        gate_vecs, Us = [], []
        for index, future in enumerate(futures):
            if index % 1000 == 0:
                print('finished rate =', index, '/', len(futures))
            if multi_process:
                future = ray.get(future)
            gate_vecs += future[0]
            Us += future[1]

        gate_vecs = jnp.array(gate_vecs, dtype=jnp.float64)
        Us = jnp.array(Us, dtype=jnp.float64)

        print('len(Us) = ', len(Us), 'len(gate_vecs) = ', len(gate_vecs))

        self.data = Us, gate_vecs
        # pkl_dump(self.data, pkl_name)
        return self.data

    def construct_model(self):
        # pkl_name = f'./temp_data/sys_U_to_vec_model_{self.name}.pkl'
        # pkl_result = pkl_load(pkl_name)

        # if pkl_result is not None:
        #     self.U_to_vec_model, self.final_mapping = pkl_result
        #     return self.U_to_vec_model, self.final_mapping

        n_qubits = self.n_qubits
        # backend = self.backend
        upstream_model = self.upstream_model

        Us, gate_vecs = self.data
        U_to_vec_model = DNN((4**n_qubits, upstream_model.max_table_size**2, 2**n_qubits, 2**n_qubits,
                             upstream_model.max_table_size*2), verbose=True, n_iter_no_change=10)  # , activation='sigmoid'
        U_to_vec_model.fit(Us, gate_vecs)

        # 训练最后一层
        X = U_to_vec_model.predict(Us[:20000])[
            :, 1:upstream_model.max_table_size+1]
        Y = gate_vecs[:, 1:upstream_model.max_table_size+1][:20000]
        # print(X.shape, Y.shape)

        upstream_model = self.upstream_model
        vec_elm_classifiers = []
        for elm_index in range(upstream_model.max_table_size):
            model = DecisionTreeClassifier(
                max_depth=3, random_state=random.randint(0, 1000))  # 初始化模型
            model.fit(X, Y[:, elm_index])  # 训练模型
            vec_elm_classifiers.append(model)

        self.U_to_vec_model = U_to_vec_model
        self.vec_elm_classifiers = vec_elm_classifiers
        # pkl_dump((U_to_vec_model, vec_elm_classifiers), pkl_name)
        return

    @staticmethod
    def find_parmas(n_qubits, layer2gates, U, lr=1e-1, max_epoch=1000, allowed_dist=1e-2, n_iter_no_change=10, no_change_tolerance=1e-2, random_params = True):
        print('start find_parmas')

        param_size = 0
        params = []
        for layer in layer2gates:
            for gate in layer:
                param_size += len(gate['params'])
                params += list(gate['params'])

        if random_params:
            params = jax.random.normal(jax.random.PRNGKey(
                random.randint(0, 100)), (param_size,))
        else:
            params = jnp.array(params)

        dev = qml.device("default.qubit", wires=n_qubits*2)
        '''TODO：hilbert_test相比于distance似乎会偏高一些，要计算一些换算公式'''
        '''
        TODO: 对比下hilbert_test和local hilbert_test，probs和expval的速度差异
        '''
        @qml.qnode(dev, interface="jax")
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

            '''local hilbert_test'''
            # qml.CNOT(wires= [0, n_qubits])
            # qml.Hadamard(0)
            # return qml.expval(qml.PauliZ(0) @ qml.PauliZ(n_qubits))

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
            # return (1 - probs)**2
            return (1 - probs[0])

        opt = optax.adamw(learning_rate=lr)
        opt_state = opt.init(params)

        min_dist = 1e2
        best_params = None

        loss_decrease_history = []
        former_loss = 0
        epoch = 0
        while True:
            # _time = time.time()
            loss_value, gradient = jax.value_and_grad(
                cost_hst)(params, U)  # 需要约1s
            # print(1, time.time() - _time)

            # _time = time.time()
            updates, opt_state = opt.update(gradient, opt_state, params)
            params = optax.apply_updates(params, updates)
            # print(2, time.time() - _time)

            # if epoch % 10 == 0:
            print('Epoch: {:5d} | Loss: {:.5f} '.format(epoch, loss_value))

            if min_dist > loss_value:
                min_dist = loss_value
                best_params = params

            loss_decrease_history.append(former_loss-loss_value)
            former_loss = loss_value
            if epoch < n_iter_no_change:
                loss_no_change = False
            else:
                loss_no_change = True
                for loss_decrement in loss_decrease_history[-n_iter_no_change:]:
                    if loss_decrement > no_change_tolerance:
                        loss_no_change = False
            if loss_no_change or epoch > max_epoch or loss_value < allowed_dist:
                if epoch < 10:
                    print()
                break

            epoch += 1

        print('finish find_parmas')
        return best_params, min_dist

    def synthesize(self, U, allowed_dist=5e-2):
        n_qubits = self.n_qubits
        U_to_vec_model: DNN = self.U_to_vec_model
        # final_mapping_parms = self.final_mapping_parms
        vec_elm_classifiers = self.vec_elm_classifiers
        upstream_model = self.upstream_model

        def assign_params(params, layer2gates):
            layer2gates = copy.deepcopy(layer2gates)
            count = 0
            for gates in layer2gates:
                for gate in gates:
                    for index, _ in enumerate(gate['params']):
                        gate['params'][index] = params[count]
                        count += 1
            return layer2gates
        
        '''TODO: 还没加上， 万一现在里面有一个layer有操作同一个比特的门怎么办还要想下'''
        def inverse(layer2gates: list):
            layer2gates = copy.deepcopy(layer2gates)
            layer2gates.reverse()

        
        inv_U = U.T.conj()
        now_U = inv_U
        I = jnp.eye(2**n_qubits)
        circuit_U = I
        # if nowU is not identity
        now_dist = 1
        # print(matrix_distance_squared(inv_U @ U, I))

        # @ray.remote
        # def find_parmas_remote(n_qubits, total_layer2gates, U):
        #     return SynthesisModel.find_parmas(n_qubits, total_layer2gates, U, max_epoch=50)

        def merge(total_layer2gates, allowed_dist, no_change_tolerance):
            params, dist = self.find_parmas(n_qubits, total_layer2gates, inv_U, max_epoch=50, allowed_dist=allowed_dist,
                                            n_iter_no_change=5, no_change_tolerance=no_change_tolerance, random_params = False)  # 有时候甚至会长，因为优化的问题
            total_layer2gates = assign_params(params, total_layer2gates)
            circuit_U: jnp.array = layer_circuit_to_matrix(total_layer2gates, n_qubits)
            now_U = circuit_U @ U
            return total_layer2gates, circuit_U, now_U
            
        total_layer2gates = []
        iter_count = 0
        
        min_dist  = 1e2
        endless_detector = 0
        while now_dist > allowed_dist:
            inv_now_U = now_U.T.conj()
            vec = U_to_vec_model.predict([transformU(inv_now_U)])[0]
            device = round(float(vec[0]))
            device = min([device, self.n_qubits-1])
            device = max([device, 0])
            # gate_vec = final_mapping(final_mapping_parms, vec[1:])

            gate_vec = jnp.array([
                vec_elm_classifiers[elm_index].predict([vec[1:]])[0]
                for elm_index in range(upstream_model.max_table_size)
            ])
            gate_vec = jnp.where(gate_vec > (1-5e-1), 1, 0)  # TODO: 万一有冲突怎么办
            # TODO: 怎么映射到0和1还要研究下
            '''TODO:有可能找到的子电路还会使dist变大，可能需要类似残差的结构'''
            layer2gates = upstream_model.reconstruct(
                device, gate_vec)  # TODO: 如果出现全是0的怎么办
            
            # print(layer2gates)

            '''TODO：试下set多个初始参数并行的找'''
            # futures = [
            #     find_parmas_remote.remote(n_qubits, total_layer2gates, U)
            #     for _ in range(50)
            # ]
            # futures = ray.get(futures)
            # min_dist = 1e10
            # for params, dist in futures:
            #     if dist < min_dist:  #TODO: 现在每个值好像都是一样的
            #         min_dist = dist
            #         best_params = params
            # params = best_params

            '''TODO: allowed_dist, n_iter_no_change 可以未来动态调整'''
            # 之前学到的参数没有放到本轮的迭代中
            if now_dist/allowed_dist < 3:
                no_change_tolerance = allowed_dist/2
            else:
                no_change_tolerance = 5e-3
                
            params, dist = self.find_parmas(n_qubits, layer2gates, inv_now_U, max_epoch=50, allowed_dist=allowed_dist,
                                                                    n_iter_no_change=5, no_change_tolerance=no_change_tolerance)
            layer2gates = assign_params(params, layer2gates)

            layer_U: jnp.array = layer_circuit_to_matrix(layer2gates, n_qubits)
            circuit_U = layer_U @ circuit_U
            # circuit_U = layer_circuit_to_matrix(total_layer2gates, n_qubits)
            now_U = circuit_U @ U
            '''now_dist 可不可以用find_params的dist代替'''
            now_dist = matrix_distance_squared(now_U, I)
            print('now_dist=', now_dist)
            if now_dist > min_dist:
                should_merge = True
                '''TODO: 不加入会不会死循环'''
                endless_detector += 1
                if endless_detector > 10:
                    print('EROOR: endless loop')
                if endless_detector > 2:
                    should_merge = False
                    total_layer2gates += layer2gates
                    endless_detector = 0
            else:
                total_layer2gates += layer2gates
                should_merge = False
                endless_detector = 0
                
            if (iter_count+1) % 3 == 0 or should_merge:
                print('merge')
                total_layer2gates, circuit_U, now_U = merge(total_layer2gates, allowed_dist, no_change_tolerance)

            iter_count += 1
            
            now_dist = matrix_distance_squared(now_U, I)
            if now_dist < min_dist:
                min_dist = now_dist
            print('iter=', iter_count, 'now_dist=', now_dist)
        
        '''TODO: 当前的电路要逆一下才是U的电路'''
        return total_layer2gates

    def save(self):
        with open(f'./temp_data/{self.name}.pkl', 'wb') as f:
            pickle.dump(self, f)
        return

    @staticmethod
    def load(name):
        with open(f'./temp_data/{name}.pkl', 'rb') as f:
            obj = pickle.load(f)
        return obj