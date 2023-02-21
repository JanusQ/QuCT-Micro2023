import tensorcircuit as tc
import numpy as np
from numpy import e, pi, cos
import pennylane as qml
import jax

K = tc.set_backend("jax")

# @jax.jit
def layer_circuit_to_matrix(layer2gates, n_qubits, params = None) -> jax.numpy.array:
    point = 0
    
    circuit = tc.Circuit(n_qubits)
    for layer in layer2gates:
        for gate in layer:
            # gate_qubits = [n_qubits - qubit - 1 for qubit in gate['qubits']]
            gate_qubits = gate['qubits']
            if gate['name'] == 'u':
                if params is None:
                    phi, theta, omega = gate['params']
                else:
                    phi, theta, omega = params[point:point+3]
                    point+=3
                u_gate = qml.Rot.compute_matrix(phi = phi, theta = theta, omega = omega)
                circuit.any(*gate_qubits, unitary=u_gate)
                pass
            elif gate['name'] == 'cx':
                circuit.cnot(*gate_qubits)
            elif gate['name'] == 'cz':
                circuit.cz(*gate_qubits)
            else:
                raise Exception('Unkown gate type', gate)
    
    return circuit.matrix()






# def layer_circuit_to_matrix(layer2gates, n_qubits):
    
#     circuit = tc.Circuit(n_qubits)
#     for layer in layer2gates:
#         for gate in layer:
#             # gate_qubits = [n_qubits - qubit - 1 for qubit in gate['qubits']]
#             gate_qubits = gate['qubits']
#             if gate['name'] == 'u':
#                 # theta: float = 0, phi: float = 0, lbd:
#                 phi, theta, omega = gate['params']
#                 u_gate = qml.Rot.compute_matrix(phi = phi, theta = theta, omega = omega)
#                 # u_gate = tc.gates.u_gate(theta = theta, phi = phi, lbd = omega).get_tensor()
                
#                 # u_gate = np.array([
#                 #     [e**(-1j*(phi+omega)/2)*cos()  ],
#                 #     [],
#                 # ], dtype=np.complex128)
#                 # c = qml.math.cos(theta / 2)
#                 # s = qml.math.sin(theta / 2)
#                 # one = qml.math.ones_like(phi) * qml.math.ones_like(omega)
#                 # c = c * one
#                 # s = s * one
#                 # u_gate = [
#                 #     [
#                 #         qml.math.exp(-0.5j * (phi + omega)) * c,
#                 #         -qml.math.exp(0.5j * (phi - omega)) * s,
#                 #     ],
#                 #     [
#                 #         qml.math.exp(-0.5j * (phi - omega)) * s,
#                 #         qml.math.exp(0.5j * (phi + omega)) * c,
#                 #     ],
#                 # ]
#                 circuit.any(*gate_qubits, unitary=u_gate)
#                 # circuit.apply_general_gate_delayed(u_gate, )
                
#                 # circuit.Rot(*gate['params'], wires=gate['qubits'][0] + offest)
#                 # circuit.u_gate(*gate['params'])
#                 # https://tensorcircuit.readthedocs.io/en/latest/api/gates.html#tensorcircuit.gates.u_gate
#                 # assert False, 'not implemeted'
#                 pass
#             elif gate['name'] == 'cx':
#                 circuit.cnot(*gate_qubits)
#             elif gate['name'] == 'cz':
#                 circuit.cz(*gate_qubits)
#             else:
#                 raise Exception('Unkown gate type', gate)
    
#     return circuit.matrix()