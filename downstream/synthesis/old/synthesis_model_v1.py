'''
    给定一个酉矩阵U，将其转化为一组门的组合
'''
# from downstream.synthesis import 

import cloudpickle as pickle
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend
from circuit import gen_random_circuits
from upstream import RandomwalkModel
from downstream.synthesis.wrong.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.wrong.tensor_network_op_ import layer_circuit_to_matrix
from downstream.synthesis.neural_network import NeuralNetworkModel
from sklearn.neural_network import MLPRegressor as DNN #MLPClassifier as DNN

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

config.update("jax_enable_x64", True)

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
    U = jnp.array(U, dtype = jnp.float64)
    return U

class SynthesisModel():
    def __init__(self, backend: Backend):
        self.backend = backend
        self.n_qubits = backend.n_qubits
        
        self.action_Q = None  # 每个device一个
        # U.size -> len(path_table)
        
        # , reward = mean(output)
        
        return
    
    # 用强化学习来查找
    def get_action(self, state) -> jnp.array:
        return

    def learn(self, state, action, reward, learning_rate):
        ''' 
            state: target unitary
            action: gate vector 
        '''
        return
    
    def reconstruct():
        sub_layer_circuit = []
        return sub_layer_circuit
    
    
    '''没跑通，距离算出来是错的，但是这样的方法会不会更快一些'''
    # def find_parmas(self, circuit_info, U, lr = 1e-2, max_epoch = 100):
    #     n_qubits = self.n_qubits
    #     layer2gates = circuit_info['layer2gates']
        
    #     param_size = 0
    #     params = []
    #     for layer in layer2gates:
    #         for gate in layer:
    #     # for gate in circuit_info['gates']:
    #             param_size += len(gate['params'])
    #             params += gate['params']
        
    #     params = jnp.array(params)
    #     # params = jax.random.normal(jax.random.PRNGKey(random.randint(0, 100)), (param_size,))
        

    #     with qml.tape.QuantumTape(do_queue=False) as u_tape:
    #         qml.QubitUnitary(U, wires=list(range(n_qubits)))

    #     def circuit(params):
    #         layer_circuit_to_pennylane_circuit(layer2gates, params= params, offest=n_qubits)
            
        
    #     dev = qml.device("default.qubit", wires=n_qubits*2)
        
    #     @qml.qnode(dev, interface="jax")
    #     def hilbert_test(params):
    #         qml.HilbertSchmidt(params, v_function=circuit, v_wires=list(range(n_qubits, 2*n_qubits)), u_tape=u_tape)
    #         return qml.probs(list(range(2*n_qubits)))  # qml.expval
        
    #     def cost_hst(params):
    #         return (1 - hilbert_test(params)[0])
        
    #     opt = optax.adamw(learning_rate=lr)
    #     opt_state = opt.init(params)
        
    #     for epoch in range(max_epoch):
    #         loss_value, gradient = jax.value_and_grad(cost_hst)(params, )
    #         updates, opt_state = opt.update(gradient, opt_state, params)
    #         params = optax.apply_updates(params, updates)
            
    #         print(epoch, loss_value)   
        
        
    #     print('finish')
    #     return
    
    @staticmethod
    def find_parmas(n_qubits, layer2gates, U, lr = 1e-1, max_epoch = 100, max_dist = 1e-2):
        # n_qubits = self.n_qubits

        param_size = 0
        params = []
        for layer in layer2gates:
            for gate in layer:
        # for gate in circuit_info['gates']:
                param_size += len(gate['params'])
                params += gate['params']
        
        # params = jnp.array(params)
        params = jax.random.normal(jax.random.PRNGKey(random.randint(0, 100)), (param_size,))
        
        def cost_hst(params):
            return matrix_distance_squared(layer_circuit_to_matrix(layer2gates, n_qubits, params), U)
        
        opt = optax.adamw(learning_rate=lr)
        opt_state = opt.init(params)
        
        for epoch in range(max_epoch):
            loss_value, gradient = jax.value_and_grad(cost_hst)(params, )
            updates, opt_state = opt.update(gradient, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            if loss_value < max_dist:
                break
            
            print(epoch, loss_value)   
    
        
        print('finish')
        return

    
    def construct_model(self): 
        n_qubits = self.n_qubits
        backend = self.backend
        
        max_gate = 1000 #4**n_qubits
        
        '''生成用于测试的模板电路'''
        circuits = gen_random_circuits(min_gate = 100, max_gate = max_gate, gate_num_step = max_gate//50, n_circuits = 1, two_qubit_gate_probs=[4, 8], backend = backend, reverse = False, optimize = True)

        upstream_model = RandomwalkModel(1, 20, backend = backend, travel_directions=('parallel', 'next'))
        upstream_model.train(circuits, multi_process = True)
        self.upstream_model = upstream_model
        
        # for epoch in range(1000):
        #     U = layer_circuit_to_matrix(layer2gates, n_qubits)
            
        #     init_unitary_mat = unitary_group.rvs(2**n_qubits)
        
        
        def random_params(layer2gates):
            layer2gates = copy.deepcopy(layer2gates)
            for layer in layer2gates:
                for gate in layer:
                    gate['params'] = [
                        random.random() * 2 * jnp.pi
                        for param in gate['params']
                    ]
            return layer2gates
        
        # @staticmethod
        # def transformUs(Us):
        #     new_Us = []
        #     for U in Us:
        #         U = jnp.concatenate([U.imag, U.real])
        #         new_Us.append(U)
        #     new_Us = jnp.array(new_Us, dtype = jnp.float64)
        #     return new_Us
        
        # @staticmethod

        
        @ray.remote
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
                for _ in range(20):
                    target_gate_index = random.randint(0, len(layer_gates)-1)
                    target_gate_index = layer_gates[target_gate_index]['id']
                    target_gate = circuit_info['gates'][target_gate_index]

                    gate_vec = circuit_info['vecs'][target_gate_index]
                
                    _layer2gates = random_params(layer2gates[layer_index:])
                    U = layer_circuit_to_matrix(_layer2gates, n_qubits)
                    U = transformU(U)
                    # U = U.reshape(U.size)
                    
                    # 似乎根本没有出现过
                    # _dists = vmap(matrix_distance_squared, in_axes=(None, 0))(U, jnp.array(U, dtype=jnp.complex128))
                    # if jnp.any(_dists < 0.1):
                    #     continue
                    
                    Us.append(U)
                    gate_vecs.append(np.concatenate([jnp.array(target_gate['qubits']), gate_vec], axis=0)  )
                    
            return gate_vecs, Us
        
        futures = []
        for circuit_info in circuits:
            # upstream_model.vectorize(circuit_info)
            future = _gen_data.remote(circuit_info, n_qubits)
            # future = _gen_data(circuit_info, n_qubits)
            futures.append(future)
            
        gate_vecs, Us = [], []
        for future in futures:
            if not isinstance(future, tuple):
                future = ray.get(future)
            gate_vecs += future[0]
            Us += future[1]

        gate_vecs = jnp.array(gate_vecs, dtype=jnp.float64)
        Us = jnp.array(Us, dtype=jnp.float64)
        
        neural_model = DNN((4**n_qubits, 2**n_qubits, 2**n_qubits, n_qubits), verbose = True, n_iter_no_change=20)
        neural_model.fit(Us, gate_vecs)
        
        self.U_to_gatevec_model = neural_model
        # neural_model.score(transform(Us_test), gate_vecs_test)
        
        # neural_model = DNN((2**n_qubits, 2**n_qubits, n_qubits), verbose = True)
        # neural_model.fit(Us, gate_vecs)
        
        # Us, Us_test, gate_vecs, gate_vecs_test = train_test_split(Us, gate_vecs, test_size=.1)
        
        # neural_model = NeuralNetworkModel([4**n_qubits, 2**n_qubits, 2**n_qubits, n_qubits, upstream_model.max_table_size + 1])
        # neural_model.fit(Us, gate_vecs)
        # self.neural_model = neural_model
        
        # for circuit_info in circuits:
        #     print(circuit_info['qiskit_circuit'])
        #     layer2gates = circuit_info['layer2gates']
            
        #     '''TODO: 用jax加速'''
        #     U = layer_circuit_to_matrix(layer2gates, n_qubits)
            
        #     # U = layer_circuit_to_pennylane_tape(layer2gates)
        #     # U = qml.matrix(U)
            
        #     self.find_parmas(circuit_info, U, 1e-1)
            
        #     # 1. 选几个比特(sQ)构建model
        #     # 2. 对于其第一个门做randomwalk
        #     # q.fit(U -> vec) for q in sQ 
            
        #     '''TODO: vmap'''
        #     # tc_m = layer_circuit_to_matrix(layer2gates, n_qubits)

        #     # U = layer_circuit_to_pennylane_tape(layer2gates)
        #     # assert np.allclose(tc_m, qml.matrix(U))
            

        #     # print(matrix_distance_squared(tc_m, qml_m))
        #     # 强化学习
        #     print()
            
        return
    
    def synthesize(self, U):
        layer2gates = []
        return layer2gates
    
    def save(self, path):
        with open(f'./model/{path}', 'wb') as f:
            pickle.dump(self, f)
        return

    @staticmethod
    def load(path):
        return _load(path)


def _load(path) -> SynthesisModel:
    with open(f'./model/{path}', 'rb') as f:
        obj  = pickle.load(f)
    return obj
