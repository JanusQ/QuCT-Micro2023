import pennylane as qml
from pennylane import numpy as pnp
from jax import numpy as jnp


from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config

def to_unitary(parmas):
    z = 1/jnp.sqrt(2)*parmas
    q, r = jnp.linalg.qr(z)
    d = r.diagonal()
    q *= d/jnp.abs(d)
    return q

def layer_circuit_to_pennylane_circuit(layer2gates, params = None, offest = 0):
    # with qml.tape.QuantumTape(do_queue=False) as U:
    point = 0
    for layer in layer2gates:
        for gate in layer:
            qubits = [q+offest for q in gate['qubits']]
            if gate['name'] == 'u':
                if params is None:
                    qml.Rot(*gate['params'], wires=qubits)
                else:
                    qml.Rot(*params[point: point+3], wires=qubits)
                    point += 3
                # qml.RX(0, wires=gate['qubits'][0])
                pass
            elif gate['name'] == 'cx':
                qml.CNOT(wires=qubits)
            elif gate['name'] == 'cz':
                qml.CZ(wires=qubits)
            elif gate['name'] == 'unitary':
                n_qubits = len(qubits)
                # if params is not None:
                #     unitary_params = params[point: point+(4**n_qubits)].reshape((2**n_qubits, 2**n_qubits))
                #     point += 4**n_qubits
                # else:
                #     unitary_params = gate['params'].reshape((2**n_qubits, 2**n_qubits))
                if params is not None:
                    unitary_params = params[point: point+(4**n_qubits)*2]
                    point += (4**n_qubits) * 2
                else:
                    unitary_params = gate['params']
                unitary_params = (unitary_params[0: 4**n_qubits] + 1j * unitary_params[4**n_qubits:]).reshape((2**n_qubits, 2**n_qubits))
                unitary = to_unitary(unitary_params)
                # assert jnp.allclose(unitary.T.conj() @ unitary, jnp.eye(2**n_qubits))
                qml.QubitUnitary(unitary, wires=qubits)
            else:
                raise Exception('Unkown gate type', gate)
    # return U

'''TODO: 会出现比特没有被用到然后矩阵算错的情况'''
def layer_circuit_to_pennylane_tape(layer2gates, params = None, offest = 0):
    with qml.tape.QuantumTape(do_queue=False) as U:
        layer_circuit_to_pennylane_circuit(layer2gates, params = params, offest = offest)
    return U

# def dist(U: jnp.array, layer2gates: list, n_qubits: int):
#     '''
#         Returns the distance between target U and the unitary of layer2gates
#         output: [0, 1], 0 is means that U and layer2gates are same
#     '''
#     dev = qml.device("default.qubit", wires=n_qubits)
    
#     def circuit():
#         layer_circuit_to_pennylane_circuit(layer2gates)

                    
#     @qml.qnode(dev)
#     def hilbert_test(v_params, v_function, v_wires, u_tape):
#         qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
#         return qml.probs(u_tape.wires + v_wires)  #qml.probs([0, 1, 2, 3]) #qml.expval(qml.PauliX(0))  #qml.probs(u_tape.wires + v_wires)

#     dist = 1 - hilbert_test(v_params=None, v_function=circuit, v_wires=list(range(n_qubits, 2*n_qubits)), u_tape=u_tape)[0]

#     return