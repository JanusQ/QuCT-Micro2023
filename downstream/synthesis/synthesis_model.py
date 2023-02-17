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

config.update("jax_enable_x64", True)

@jax.jit
def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    # optimized implementation
    return jnp.abs(1 - jnp.abs(jnp.sum(jnp.multiply(A, jnp.conj(B)))) / A.shape[0])

def get_params(layer2gates, U: jnp.array):
    return

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
    
    
    def find_parmas(self, circuit_info, U, lr = 1e-1, max_epoch = 100, max_dist = 1e-2):
        n_qubits = self.n_qubits
        layer2gates = circuit_info['layer2gates']
        
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
        
        max_gate = 150 #4**n_qubits
        
        '''生成用于测试的模板电路'''
        circuits = gen_random_circuits(min_gate = 100, max_gate = max_gate, gate_num_step = max_gate//50, n_circuits = 1, two_qubit_gate_probs=[4, 8], backend = backend, reverse = False, optimize = True)

        # upstream_model = RandomwalkModel(1, 20, backend = backend)
        # upstream_model.train(circuits, multi_process = True)
        
        # for epoch in range(1000):
        #     U = layer_circuit_to_matrix(layer2gates, n_qubits)
            
        #     init_unitary_mat = unitary_group.rvs(2**n_qubits)
            
        
        
        for circuit_info in circuits:
            print(circuit_info['qiskit_circuit'])
            layer2gates = circuit_info['layer2gates']
            
            '''TODO: 用jax加速'''
            U = layer_circuit_to_matrix(layer2gates, n_qubits)
            
            # U = layer_circuit_to_pennylane_tape(layer2gates)
            # U = qml.matrix(U)
            
            self.find_parmas(circuit_info, U, 1e-1)
            
            # 1. 选几个比特(sQ)构建model
            # 2. 对于其第一个门做randomwalk
            # q.fit(U -> vec) for q in sQ 
            
            '''TODO: vmap'''
            # tc_m = layer_circuit_to_matrix(layer2gates, n_qubits)

            # U = layer_circuit_to_pennylane_tape(layer2gates)
            # assert np.allclose(tc_m, qml.matrix(U))
            

            # print(matrix_distance_squared(tc_m, qml_m))
            # 强化学习
            print()
            
        return
    
    def synthesize(self, U):
        layer2gates = []
        return layer2gates
    
    def save(self, path):
        return
    
    @staticmethod
    def load(path):
        return
