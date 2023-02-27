'''
    给定一个酉矩阵U，将其转化为一组门的组合
'''
# from downstream.synthesis import

import cloudpickle as pickle
from qiskit import transpile
from circuit.formatter import layered_circuits_to_qiskit
from circuit.parser import qiskit_to_layered_circuits
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
from pennylane.optimize import NesterovMomentumOptimizer

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

        if name is not None:
            self.model_path = './temp_data/' + name + '/'
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
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
        pkl_name = self.model_path + f'data.pkl'
        # data = pkl_load(pkl_name)
        # if data is not None:
        #     self.data = data
        #     return data

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

                    gate_vec = circuit_info['vecs'][target_gate_index].astype(jnp.int8)

                    _layer2gates = random_params(layer2gates[layer_index:])
                    # TODO: 应该有些可以重复利用才对
                    U = layer_circuit_to_matrix(
                        _layer2gates, n_qubits)  # TODO: vmap
                    U = transformU(U)

                    '''TODO: 那些是jnp哪些是np要理一下'''
                    Us.append(U)
                    
                    device_vec = np.array([target_gate['qubits'][0]], dtype=jnp.int8)
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
            device_gate_vecs, dtype=np.int8)#[:100000]
        Us = np.array(Us, dtype=np.float64)#[:100000]

        print('len(Us) = ', len(Us), 'len(gate_vecs) = ', len(device_gate_vecs))

        self.data = Us, device_gate_vecs
        # pkl_dump(self.data, pkl_name)
        return self.data

    def construct_model(self):
        print('start construct model')
        start_time = time.time()
        # pkl_name = self.model_path + f'U_to_vec_model.pkl'
        # pkl_result = pkl_load(pkl_name)

        # if pkl_result is not None:
        #     self.U_to_vec_model = pkl_result
        #     return self.U_to_vec_model

        n_qubits = self.n_qubits
        # backend = self.backend
        # upstream_model = self.upstream_model

        Us, gate_vecs = self.data
        # U_to_vec_model = DecisionTreeClassifier()
        # U_to_vec_model.fit(Us, gate_vecs)

        self.neural_model = DNN((4**n_qubits, 2**n_qubits, 2**n_qubits, n_qubits), verbose = True, n_iter_no_change=10)
        self.neural_model.fit(Us, gate_vecs)

        output = self.neural_model.predict(Us[:10000])
        gate_vecs = gate_vecs[:10000]
        U_to_vec_model = DecisionTreeClassifier()
        U_to_vec_model.fit(output, gate_vecs)

        self.U_to_vec_model = U_to_vec_model
        # self.data = None
        # U_to_vec_model.predict(Us[:1])
        # pkl_dump(U_to_vec_model, pkl_name)
        print(f'finish construct model, cost {time.time()-start_time}s')
        return


    @staticmethod
    def find_parmas(n_qubits, layer2gates, U, lr=1e-1, max_epoch=1000, allowed_dist=1e-2, n_iter_no_change=10, no_change_tolerance=1e-2, random_params=True):
        # print('start find_parmas')

        # lr = 1e-3
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
        '''
        TODO: 对比下hilbert_test和local hilbert_test，probs和expval的速度差异
        '''
        @qml.qnode(dev, interface="jax") # interface如果没有jax就不会有梯度
        # @qml.qnode(dev)
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
            # return qml.probs([0, n_qubits])  # 这个可能有问题，优化出来的对不上矩阵的参数
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
        start_time = time.time()
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
            # if epoch < n_iter_no_change:
            #     loss_no_change = False
            # else:
            #     loss_no_change = True
            #     for loss_decrement in loss_decrease_history[-n_iter_no_change:]:
            #         if loss_decrement > no_change_tolerance:
            #             loss_no_change = False
            loss_no_change = False
            if loss_no_change or epoch > max_epoch or loss_value < allowed_dist:
                # if epoch < 10:
                #     print()
                break

            epoch += 1
        print('Epoch: {:5d} | Loss: {:.5f} '.format(epoch, loss_value))
        # print('time per itr = ', (time.time() - start_time)/epoch, 's')
        # print('finish find_parmas')
        return best_params, min_dist

    def synthesize(self, U, allowed_dist=5e-2):
        print('start synthesis')
        n_qubits = self.n_qubits
        U_to_vec_model: DecisionTreeClassifier = self.U_to_vec_model
        neural_model: DNN = self.neural_model
        upstream_model = self.upstream_model

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

        '''TODO: 还没加上， 万一现在里面有一个layer有操作同一个比特的门怎么办还要想下'''
        def inverse(layer2gates: list):
            layer2gates = copy.deepcopy(layer2gates)
            layer2gates.reverse()

        inv_U = U.T.conj()
        now_U = U
        I = jnp.eye(2**n_qubits)
        circuit_U = I
        # if nowU is not identity
        now_dist = 1

        # @ray.remote
        # def find_parmas_remote(n_qubits, total_layer2gates, U):
        #     return SynthesisModel.find_parmas(n_qubits, total_layer2gates, U, max_epoch=50)

        total_layer2gates = []
        iter_count = 0

        min_dist = 1e2
        endless_detector = 0
        while now_dist > allowed_dist:
            inv_now_U = now_U.T.conj()
            device_gate_vec = neural_model.predict([transformU(inv_now_U)])
            device_gate_vec = U_to_vec_model.predict(device_gate_vec)[0]
            
            device = int(device_gate_vec[0])
            gate_vec = device_gate_vec[1:]

            gate_vec = device_gate_vec[n_qubits:]
            # gate_vec = jnp.where(gate_vec > (1-5e-1), 1, 0)  # TODO: 万一有冲突怎么办
            # TODO: 怎么映射到0和1还要研究下
            '''TODO:有可能找到的子电路还会使dist变大，可能需要类似残差的结构'''
            layer2gates = upstream_model.reconstruct(
                device, gate_vec)  # TODO: 如果出现全是0的怎么办
            
            involved_qubits = []
            for layer_gates in layer2gates:
                for gate in layer_gates:
                    involved_qubits += gate['qubits']
            involved_qubits = set(involved_qubits)
            
            # dim = 2**len(involved_qubits)
            n_involved_qubits = len(involved_qubits)
            layer2gates = [
                [{
                    'name' : 'unitary',
                    'qubits': involved_qubits,
                    # 'params': (np.random.rand(dim, dim) + 1j* np.random.rand(dim, dim)).reshape(dim**2)  #unitary_group.rvs(2**len(involved_qubits)).reshape((4**len(involved_qubits),))  #jnp.array(, dtype=jnp.complex128)
                    'params': np.random.rand((4**n_involved_qubits)*2),
                }]
            ]
            # jnp.eye(2**len(involved_qubits))
            
            no_change_tolerance = 1e-3

            total_layer2gates = total_layer2gates + layer2gates
            # print('merge')
            '''TODO: 每次值调整一部分'''
            unchanged_total_layer2gates, total_layer2gates = total_layer2gates[:-10], total_layer2gates[-10:]
            unchanged_part_matrix = layer_circuit_to_matrix(
                unchanged_total_layer2gates, n_qubits)


            # print(qiskit_circuit)
            # start_time = time.time()
            params, dist = self.find_parmas(n_qubits, total_layer2gates, inv_U @ unchanged_part_matrix.T.conj(), max_epoch=50, allowed_dist=allowed_dist,
                                            n_iter_no_change=5, no_change_tolerance=no_change_tolerance, random_params=False)  # 有时候甚至会长，因为优化的问题
            # print(4, time.time() - start_time)  # 35.502s
            
            # start_time = time.time()
            total_layer2gates = assign_params(params, total_layer2gates)
            circuit_U: jnp.array = layer_circuit_to_matrix(
                total_layer2gates, n_qubits) 
            # print(5, time.time() - start_time)
            
            # start_time = time.time()
            circuit_U = circuit_U @ unchanged_part_matrix  #TODO: 放在电路里面可能算快一些
            # print(6, time.time() - start_time)
            
            total_layer2gates = unchanged_total_layer2gates + total_layer2gates
            # start_time = time.time()
            now_U = circuit_U @ U
            # print(7, time.time() - start_time)
            # print(matrix_distance_squared(circuit_U.T.conj(), U))  # 和now_dist是相等的
            
            # start_time = time.time()
            now_dist = matrix_distance_squared(now_U, I)
            # print(8, time.time() - start_time)
            
            if now_dist < min_dist:
                min_dist = now_dist
            print('iter_count=',iter_count, 'now_dist=', now_dist)

            iter_count += 1

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
