import pennylane as qml
from pennylane import numpy as pnp
from jax import numpy as jnp


from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config


def layer_circuit_to_pennylane_circuit(layer2gates, offest = 0):
    # with qml.tape.QuantumTape(do_queue=False) as U:
    for layer in layer2gates:
        for gate in layer:
            if gate['name'] == 'u':
                qml.Rot(*gate['params'], wires=gate['qubits'][0] + offest)
                # qml.RX(0, wires=gate['qubits'][0])
                pass
            elif gate['name'] == 'cx':
                qml.CNOT(wires=[q+offest for q in gate['qubits']])
            elif gate['name'] == 'cz':
                qml.CZ(wires=[q+offest for q in gate['qubits']])
            else:
                raise Exception('Unkown gate type', gate)
    # return U

'''TODO: 会出现比特没有被用到然后矩阵算错的情况'''
def layer_circuit_to_pennylane_tape(layer2gates, offest = 0):
    with qml.tape.QuantumTape(do_queue=False) as U:
        layer_circuit_to_pennylane_circuit(layer2gates, offest)
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