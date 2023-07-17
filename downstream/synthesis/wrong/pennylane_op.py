import pennylane as qml
from pennylane import numpy as pnp
# from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config
import numpy as np

assert False, '有bug'

# def to_unitary(parmas):
#     z = 1/pnp.sqrt(2)*parmas
#     q, r = pnp.linalg.qr(z)
#     d = r.diagonal()
#     q *= d/pnp.abs(d)
#     return q

# def to_unitary(parmas):
#     z = 1/pnp.sqrt(2)*parmas
#     q, r = pnp.linalg.qr(z)
#     d = r.diagonal()
#     q *= d/pnp.abs(d)
#     return q

# def to_unitary(matrix):
#     svd_u, _, svd_v = pnp.linalg.svd(matrix, full_matrices = False)
#     return pnp.dot(svd_u, svd_v)

def layer_circuit_to_pennylane_circuit(layer2gates, params = None, offest = 0):
    point = 0
    for layer in layer2gates:
        for gate in layer:
            qubits = [q+offest for q in gate['qubits']]
            if gate['name'] == 'u':
                if params is None:
                    theta, phi, lam = gate['params']
                else:
                    theta, phi, lam = params[point: point+3]
                    point += 3
                qml.U3(theta, phi, lam, wires=qubits)
            elif gate['name'] == 'cx':
                qml.CNOT(wires=qubits)
            elif gate['name'] == 'cz':
                qml.CZ(wires=qubits)
            elif gate['name'] == 'unitary':
                n_qubits = len(qubits)
                # n_params = (4**n_qubits)*2
                n_params = 4**n_qubits
                if params is not None:
                    unitary_params = params[point: point+n_params]
                    point += n_params
                else:
                    unitary_params = gate['params']
                # unitary = (unitary_params[0: n_params//2] + 1j * unitary_params[n_params//2:]).reshape((2**n_qubits, 2**n_qubits))
                unitary = unitary_params[0: n_params].reshape((2**n_qubits, 2**n_qubits))
                # unitary = to_unitary(unitary)
                # assert np.allclose(pnp.conj(unitary.T) @ unitary, pnp.eye(2**n_qubits))
                qml.QubitUnitary(unitary, wires=qubits)
            else:
                raise Exception('Unkown gate type', gate)

'''TODO: 会出现比特没有被用到然后矩阵算错的情况'''
def layer_circuit_to_pennylane_tape(layer2gates, params = None, offest = 0):
    with qml.tape.QuantumTape(do_queue=False) as U:
        layer_circuit_to_pennylane_circuit(layer2gates, params = params, offest = offest)
    return U