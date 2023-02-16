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

config.update("jax_enable_x64", True)

def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    # optimized implementation
    return np.abs(1 - np.abs(np.sum(np.multiply(A, np.conj(B)))) / A.shape[0])
        
class SynthesisModel():
    def __init__(self, backend: Backend):
        self.backend = backend
        self.n_qubits = backend.n_qubits
        
        self.action_Q = None
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
    
    def construct_model(self): 
        n_qubits = self.n_qubits
        backend = self.backend
        
        max_gate = 100 #4**n_qubits
        
        '''生成用于测试的模板电路'''
        circuits = gen_random_circuits(min_gate = 20, max_gate = max_gate, gate_num_step = max_gate//50, n_circuits = 1, two_qubit_gate_probs=[4, 8], backend = backend, reverse = False, optimize = True)

        # upstream_model = RandomwalkModel(1, 20, backend = backend)
        # upstream_model.train(circuits, multi_process = True)
        

        for circuit_info in circuits:
            print(circuit_info['qiskit_circuit'])
            layer2gates = circuit_info['layer2gates']
            '''TODO: 用jax加速'''
            
            # qml_m = 
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
