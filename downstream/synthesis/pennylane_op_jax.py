import pennylane as qml
from pennylane import numpy as pnp
from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config
import numpy as np



# def to_unitary(parmas):
#     z = 1/pnp.sqrt(2)*parmas
#     q, r = pnp.linalg.qr(z)
#     d = r.diagonal()
#     q *= d/pnp.abs(d)
#     return q

def to_unitary(parmas):
    z = 1/jnp.sqrt(2)*parmas
    q, r = jnp.linalg.qr(z)
    d = r.diagonal()
    q *= d/jnp.abs(d)
    return q

def layer_circuit_to_pennylane_circuit(layer2gates, params = None, offest = 0):
    point = 0
    for layer in layer2gates:
        for gate in layer:
            qubits = [q+offest for q in gate['qubits']]
            if gate['name'] == 'u':
                # if name == 'u':  
                # pennylane和qiskit的结构是不一样的，我们这一qiskit的顺序为准更方便些
                if params is None:
                    theta, phi, lam = gate['params']
                    # qml.Rot(*parms, wires=qubits)
                else:
                    theta, phi, lam = params[point: point+3]
                    point += 3
                qml.U3(theta, phi, lam, wires=qubits)
                # qml.Rot(phi, theta, lam, wires=qubits)
                # qml.Rot(theta, phi, lam, wires=qubits)
                # qml.Rot(theta, lam, phi, wires=qubits)
                # qml.Rot(phi, lam, theta, wires=qubits)
            elif gate['name'] == 'cx':
                qml.CNOT(wires=qubits)
            elif gate['name'] == 'cz':
                qml.CZ(wires=qubits)
            elif gate['name'] == 'unitary':
                n_qubits = len(qubits)
                n_params = (4**n_qubits)*2
                if params is not None:
                    unitary_params = params[point: point+n_params]
                    point += n_params
                else:
                    unitary_params = gate['params']
                unitary_params = (unitary_params[0: n_params//2] + 1j * unitary_params[n_params//2:]).reshape((2**n_qubits, 2**n_qubits))
                unitary = to_unitary(unitary_params)
                # assert jnp.allclose(unitary.T.conj() @ unitary, jnp.eye(2**n_qubits))
                qml.QubitUnitary(unitary, wires=qubits)
            else:
                raise Exception('Unkown gate type', gate)

'''TODO: 会出现比特没有被用到然后矩阵算错的情况'''
def layer_circuit_to_pennylane_tape(layer2gates, params = None, offest = 0):
    with qml.tape.QuantumTape(do_queue=False) as U:
        layer_circuit_to_pennylane_circuit(layer2gates, params = params, offest = offest)
    return U