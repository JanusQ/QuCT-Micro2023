import pennylane as qml
from pennylane import numpy as np

from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config
from downstream.synthesis.wrong.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op_jax import layer_circuit_to_matrix
from scipy.stats import unitary_group
import random
from downstream.synthesis import matrix_distance_squared
config.update("jax_enable_x64", True)
import time

n_qubits = 2
layer2gates = [
    [
        {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
        {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
    ],
    [{'name': 'cz', 'qubits': [1, 0], 'params': []}],
    [
        {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
        {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
    ],
    [{'name': 'cz', 'qubits': [0, 1], 'params': []}],
    [
        {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
        {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
    ],
    [{'name': 'cz', 'qubits': [0, 1], 'params': []}],
    [
        {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
        {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
    ],
    [{'name': 'cz', 'qubits': [1, 0], 'params': []}],
    [
        {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
        {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
    ],
]

# n_qubits = 1
# layer2gates = [
#     [
#         {'name': 'u', 'qubits': [0], 'params': [1.1, 1.2, 1.3]},
#     ],
#     [
#         {'name': 'u', 'qubits': [0], 'params': [1.3, 1.4, 1.5]},
#     ],
# ]

print(layer2gates)

param_size = 0
params = []
for layer in layer2gates:
    for gate in layer:
        param_size += len(gate['params'])
        params += gate['params']

# original_params = params
# params = jnp.array(params)

U = jnp.array(unitary_group.rvs(2**n_qubits), dtype=jnp.complex128)

# U = layer_circuit_to_matrix(layer2gates, n_qubits)

# U = layer_circuit_to_pennylane_tape(layer2gates)
# U = qml.matrix(U)


params = jax.random.normal(jax.random.PRNGKey(random.randint(0, 100)), (param_size,))


dev = qml.device("default.qubit", wires=n_qubits*2)

'''
TODO: 对比下hilbert_test和local hilbert_test，probs和expval的速度差异

'''
@qml.qnode(dev, interface="jax")
def hilbert_test(params, U):
    for q in range(n_qubits):
        qml.Hadamard(wires=q)

    for q in range(n_qubits):
        qml.CNOT(wires=[q, q+n_qubits])

    qml.QubitUnitary(U.conj(), wires=list(range(n_qubits)))
    # qml.QubitUnitary(U, wires=list(range(n_qubits, 2*n_qubits)))
    layer_circuit_to_pennylane_circuit(
        layer2gates, params=params, offest=n_qubits)

    '''local hilbert_test'''
    # qml.CNOT(wires= [0, n_qubits])
    # qml.Hadamard(0)
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


print('dist = ', cost_hst(params, U))

opt = optax.adamw(learning_rate=1e-1)
opt_state = opt.init(params)

# fig, ax = qml.draw_mpl(hilbert_test, show_all_wires=True)(params, U)
# fig.show()

# 对于两比特expval和probs差不多
start_time = time.time()
for epoch in range(100):
    loss_value, gradient = jax.value_and_grad(cost_hst)(params, U)
    updates, opt_state = opt.update(gradient, opt_state, params)
    params = optax.apply_updates(params, updates)

    print(epoch, loss_value)

    # circuit_U = layer_circuit_to_pennylane_tape(layer2gates, params)
    # circuit_U = qml.matrix(circuit_U)
    # print(matrix_distance_squared(circuit_U, U))

print(time.time() - start_time)
# print(original_params)
# print(params)
# print(jnp.allclose(U, circuit_U))
# print(U)
# print(circuit_U)


# print(U @ (U.T.conj()))
print('finish')
