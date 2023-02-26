import pennylane as qml
from pennylane import numpy as np

from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config
from downstream.synthesis.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from scipy.stats import unitary_group
import time
import random
import scipy
config.update("jax_enable_x64", True)


n_qubits = 4
layer2gates = [[{'name': 'u', 'params': [1.1, 1.1, 1.3], 'qubits': [2]}]]


print(qml.matrix(layer_circuit_to_pennylane_tape(layer2gates), wire_order = list(range(n_qubits))))
def to_unitary(parms):
    z = 1/jnp.sqrt(2)*parms
    q, r = jnp.linalg.qr(z)
    d = r.diagonal()
    q *= d/jnp.abs(d)
    return q

# U = jnp.array(unitary_group.rvs(2**2), dtype=jnp.complex128)
dev = qml.device("default.qubit", wires=3)
@qml.qnode(dev, interface="jax")
def op2(parms):
    unitary = to_unitary(parms)
    for q in range(3):
        qml.Hadamard(wires=q)
    # qml.QubitUnitary(U, wires=[0,1])
    qml.QubitUnitary(unitary, wires=[0,1])
    return qml.probs(list(range(2))) #qml.expval(-qml.PauliX(0))

def cost(parms):
    return op2(parms)[0]

# params = jnp.array(unitary_group.rvs(2**2), dtype=jnp.complex128).reshape((16,)) #jnp.eye(4, dtype=jnp.complex128)
dim = 4
params =  jnp.array(np.random.rand(dim, dim) + 1j* np.random.rand(dim, dim))

unitary = to_unitary(params)
assert jnp.allclose(unitary.T.conj() @ unitary, jnp.eye(4))


opt = optax.adamw(learning_rate=.01)
opt_state = opt.init(params)
for i in range(10):
    loss_value, gradient = jax.value_and_grad(
        cost)(params)  # 需要约1s
    updates, opt_state = opt.update(gradient, opt_state, params)
    params = optax.apply_updates(params, updates)


dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev, interface="jax")
def op2(parm):
    for q in range(4):
        qml.Hadamard(wires=q)
    for q in range(3):
        qml.CNOT(wires=[q, q+1])
    for q in range(1,4):
        qml.CNOT(wires=[q-1, q])
    for q in range(1,4):
        qml.CRZ(parm['name']['name'][0], wires=[q-1, q]) 
    return qml.probs([0, 1, 2,3])

# jax是会有加速的
for i in range(10000):
    start_time = time.time()
    op2({'name': {'name': [random.random()]}})
    if i %100 == 0:
        print(time.time() - start_time)



# n_qubits = 2
# layer2gates = [
#     [
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#     ],
#     [{'name': 'cz', 'qubits': [1,0], 'params': []}],
#     [
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#     ],
#     [{'name': 'cz', 'qubits': [0,1], 'params': []}],
#     [
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#     ],
#     [{'name': 'cz', 'qubits': [0,1], 'params': []}],
#     [
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#     ],
#     [{'name': 'cz', 'qubits': [1,0], 'params': []}],
#     [
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#         {'name': 'u', 'qubits': [0], 'params': [np.pi]*3},
#     ],
# ]
# U = jnp.array(unitary_group.rvs(2**n_qubits), dtype=jnp.complex128)
    
# param_size = 0
# params = []
# for layer in layer2gates:
#     for gate in layer:
# # for gate in circuit_info['gates']:
#         param_size += len(gate['params'])
#         params += gate['params']

# params = jnp.array(params)
# # params = jax.random.normal(jax.random.PRNGKey(random.randint(0, 100)), (param_size,))


# with qml.tape.QuantumTape(do_queue=False) as u_tape:
#     qml.QubitUnitary(U, wires=list(range(n_qubits)))

# def circuit(params):
#     layer_circuit_to_pennylane_circuit(layer2gates, params= params, offest=n_qubits)
    
# dev = qml.device("default.qubit", wires=n_qubits*2)

# @qml.qnode(dev, interface="jax")
# def hilbert_test(params):
#     qml.HilbertSchmidt(params, v_function=circuit, v_wires=list(range(n_qubits, 2*n_qubits)), u_tape=u_tape)
#     return qml.probs(list(range(2*n_qubits)))  # qml.expval

# def cost_hst(params):
#     return (1 - hilbert_test(params)[0])

# opt = optax.adamw(learning_rate=1e-2)
# opt_state = opt.init(params)

# for epoch in range(10):
#     loss_value, gradient = jax.value_and_grad(cost_hst)(params, )
#     updates, opt_state = opt.update(gradient, opt_state, params)
#     params = optax.apply_updates(params, updates)
    
#     print(epoch, loss_value)   


# print('finish')





# def test_hs_decomposition_2_qubits():
#     """Test if the HS operation is correctly decomposed for 2 qubits."""
#     with qml.tape.QuantumTape(do_queue=False) as U:
#         qml.SWAP(wires=[0, 1])

#     def v_circuit(params):
#         qml.RZ(params[0], wires=2)
#         qml.CNOT(wires=[2, 3])

#     op = qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2, 3], u_tape=U)

#     with qml.tape.QuantumTape() as tape_dec:
#         op.decomposition()

#     expected_operations = [
#         qml.Hadamard(wires=[0]),
#         qml.Hadamard(wires=[1]),
#         qml.CNOT(wires=[0, 2]),
#         qml.CNOT(wires=[1, 3]),
#         qml.SWAP(wires=[0, 1]),
#         qml.RZ(-0.1, wires=[2]),
#         qml.CNOT(wires=[2, 3]),
#         qml.CNOT(wires=[1, 3]),
#         qml.CNOT(wires=[0, 2]),
#         qml.Hadamard(wires=[0]),
#         qml.Hadamard(wires=[1]),
#     ]

#     for i, j in zip(tape_dec.operations, expected_operations):
#         assert i.name == j.name
#         assert i.data == j.data
#         assert i.wires == j.wires


# test_hs_decomposition_2_qubits()

# define the quantum circuit
dev = qml.device("default.qubit", wires=4)

# U = np.array([[0,1],
#               [1,0]],
#             dtype='complex128')




with qml.tape.QuantumTape(do_queue=False) as U:
    _U = np.array([[0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]])
    qml.QubitUnitary(_U, wires=[0,1])
    # qml.RZ(np.pi, wires=0)
    # qml.SWAP(wires=[0, 1])
    
print(qml.matrix(U))

@qml.qnode(dev, interface="jax")
def op2():
    # qml.QubitUnitary(U, wires=[0,1])
    qml.SWAP(wires=[0, 1])
    return qml.probs([0, 1, 2,3])
print(op2())

def v_circuit(params):
    qml.RZ(params[0], wires=2)
    qml.CNOT(wires=[2, 3])

@qml.qnode(dev, interface="jax")
def op():
    qml.HilbertSchmidt([0.1], v_function=v_circuit, v_wires=[2, 3], u_tape=U)
    return qml.probs([0, 1, 2,3])

print(op())

# circuit = qml.Node(dev)(op)
# circuit([0.1])


U = np.array([[0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, -1]])


# Represents unitary U
with qml.tape.QuantumTape(do_queue=False) as u_tape1:
    # qml.SWAP(wires=[0, 1])
    # qml.Hadamard(wires=0)
    # qml.Hadamard(wires=1)
    # qml.QubitUnitary(U, wires=[0,1])
    qml.RZ(1.1, wires=0)
    qml.Hadamard(wires=0)
    qml.RZ(1.1, wires=1)
    # qml.Identity(wires=1)

# Represents unitary U
with qml.tape.QuantumTape(do_queue=False) as u_tape:
    qml.QubitUnitary(qml.matrix(u_tape1), wires=[0,1])


# print(qml.matrix(u_tape))

# with qml.tape.QuantumTape(do_queue=False) as U:
#     qml.SWAP(wires=[0, 1])

# Represents unitary V
# @qml.qnode(dev)
def v_function(params):
    # qml.RZ(params[0], wires=1)
    qml.RZ(1.1, wires=2)
    qml.Hadamard(wires=2)
    qml.RZ(1.1, wires=3)
    # qml.Identity(wires=3)
    # qml.Hadamard(wires=3)

@qml.qnode(dev, interface="jax")
def hilbert_test(v_params, v_function, v_wires, u_tape):
    qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
    return qml.probs(u_tape.wires + v_wires)  #qml.probs([0, 1, 2, 3]) #qml.expval(qml.PauliX(0))  #qml.probs(u_tape.wires + v_wires)

def cost_hst(parameters, v_function, v_wires, u_tape):
    return (1 - hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape)[0])

# hilbert_test(v_params=[0.1], v_function=v_function, v_wires=[2,3], u_tape=u_tape)
print(qml.draw(hilbert_test)(v_params=[0.1], v_function=v_function, v_wires=[2,3], u_tape=u_tape))

'''[0, 1] 越大越远'''
dist = cost_hst(parameters=[0.1], v_function=v_function, v_wires=[2,3], u_tape=u_tape)  

print(dist)

# @qml.qnode(dev)
# def circuit():
#     qml.Hadamard(wires=0)
#     qml.CNOT(wires=[0, 1])
#     return qml.probs(wires=[0, 1])

# # print the unitary matrix
# unitary = qml.unitary(circuit)
# print(unitary)