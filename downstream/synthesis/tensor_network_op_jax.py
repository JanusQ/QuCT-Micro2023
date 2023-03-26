import tensorcircuit as tc
import numpy as np
from numpy import e, pi, cos
import pennylane as qml
import jax
from downstream.synthesis.pennylane_op_jax import layer_circuit_to_pennylane_tape
from jax import numpy as jnp
K = tc.set_backend("jax")

# @jax.jit
def layer_circuit_to_matrix(layer2gates, n_qubits, params = None) -> jax.numpy.array:
    if len(layer2gates) == 0:
        return jnp.eye(2**n_qubits)
    return qml.matrix(layer_circuit_to_pennylane_tape(layer2gates, params), wire_order = list(range(n_qubits)))
    
    ''''值没有搞清楚先不用, 现在算出来的似乎不是unitary的'''
    # point = 0
    # circuit = tc.Circuit(n_qubits)
    # for layer in layer2gates:
    #     for gate in layer:
    #         # gate_qubits = [n_qubits - qubit - 1 for qubit in gate['qubits']]
    #         qubits = gate['qubits']
    #         if gate['name'] == 'u':
    #             if params is None:
    #                 phi, theta, omega = gate['params']
    #             else:
    #                 phi, theta, omega = params[point:point+3]
    #                 point+=3
    #             u_gate = qml.Rot.compute_matrix(phi = phi, theta = theta, omega = omega)
    #             circuit.any(*qubits, unitary=u_gate)
    #             pass
    #         elif gate['name'] == 'cx':
    #             circuit.cnot(*qubits)
    #         elif gate['name'] == 'cz':
    #             circuit.cz(*qubits)
    #         elif gate['name'] == 'unitary':
    #             n_qubits = len(qubits)
    #             # if params is not None:
    #             #     unitary_params = params[point: point+(4**qubits)].reshape((2**n_qubits, 2**n_qubits))
    #             #     point += 4**qubits
    #             # else:
    #             #     unitary_params =gate['params'].reshape((2**n_qubits, 2**n_qubits))
    #             if params is not None:
    #                 unitary_params = params[point: point+(4**n_qubits)*2]
    #                 point += (4**n_qubits) * 2
    #             else:
    #                 unitary_params = gate['params']
    #             unitary_params = (unitary_params[0: 4**n_qubits] + 1j * unitary_params[4**n_qubits:]).reshape((2**n_qubits, 2**n_qubits))
    #             unitary = to_unitary(unitary_params)
    #             circuit.any(*qubits, unitary=unitary)
    #         else:
    #             raise Exception('Unkown gate type', gate)
    
    # return circuit.matrix()

