
import pennylane as qml
from pennylane import numpy as np
from scipy.stats import unitary_group

import scipy as sp

# The Pauli Matrices
X = np.array( [ [ 0, 1 ],
                [ 1, 0 ] ], dtype = np.complex128 )

Y = np.array( [ [ 0, -1j ],
                [ 1j,  0 ] ], dtype = np.complex128 )

Z = np.array( [ [ 1,  0 ],
                [ 0, -1 ] ], dtype = np.complex128 )

I = np.array( [ [ 1, 0 ],
                [ 0, 1 ] ], dtype = np.complex128 )


_norder_paulis_map = [ np.array( [ I ] ), np.array( [ I, X, Y, Z ] ) ]
import itertools  as it

def get_norder_paulis( n ):
    if n < 0:
        raise ValueError( "n must be nonnegative" )

    if len( _norder_paulis_map ) > n:
        return _norder_paulis_map[n]

    norder_paulis = []
    for pauli_n_1, pauli_1 in it.product( get_norder_paulis( n - 1 ),
                                          get_norder_paulis(1) ):
        norder_paulis.append( np.kron( pauli_n_1, pauli_1 ) )

    _norder_paulis_map.append( np.array( norder_paulis ) )

    return _norder_paulis_map[n]
def dot_product ( alpha, sigma ):
    if len( alpha ) != len( sigma ):
        raise ValueError( "Length of alpha and sigma must be the same." )
    return np.sum(np.array([ a*s for a, s in zip( alpha, sigma ) ], dtype=np.complex128), 0 )

from itertools import product
from scipy.linalg import LinAlgError, bandwidth
from scipy.linalg._matfuncs_expm import pick_pade_structure, pade_UV_calc

def _exp_sinch(x):
    # Higham's formula (10.42), might overflow, see GH-11839
    lexp_diff = np.diff(np.exp(x))
    l_diff = np.diff(x)
    mask_z = l_diff == 0.
    lexp_diff[~mask_z] /= l_diff[~mask_z]
    lexp_diff[mask_z] = np.exp(x[:-1][mask_z])
    return lexp_diff

def expm(A):
    a = A
    # larger problem with unspecified stacked dimensions.
    n = a.shape[-1]
    eA = np.empty(a.shape, dtype=a.dtype)
    # working memory to hold intermediate arrays
    Am = np.empty((5, n, n), dtype=a.dtype)

    for ind in product(*[range(x) for x in a.shape[:-2]]):
        aw = a[ind]

        lu = bandwidth(aw)
        if not any(lu):  # a is diagonal?
            eA[ind] = np.diag(np.exp(np.diag(aw)))
            continue

        # Generic/triangular case; copy the slice into scratch and send.
        # Am will be mutated by pick_pade_structure
        Am[0, :, :] = aw
        m, s = pick_pade_structure(Am)

        if s != 0:  # scaling needed
            Am[:4] *= [[[2**(-s)]], [[4**(-s)]], [[16**(-s)]], [[64**(-s)]]]

        pade_UV_calc(Am, n, m)
        eAw = Am[0]

        if s != 0:  # squaring needed

            if (lu[1] == 0) or (lu[0] == 0):  # lower/upper triangular
                # This branch implements Code Fragment 2.1 of [1]

                diag_aw = np.diag(aw)
                # einsum returns a writable view
                np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2**(-s))
                # super/sub diagonal
                sd = np.diag(aw, k=-1 if lu[1] == 0 else 1)

                for i in range(s-1, -1, -1):
                    eAw = eAw @ eAw

                    # diagonal
                    np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2.**(-i))
                    exp_sd = _exp_sinch(diag_aw * (2.**(-i))) * (sd * 2**(-i))
                    if lu[1] == 0:  # lower
                        np.einsum('ii->i', eAw[1:, :-1])[:] = exp_sd
                    else:  # upper
                        np.einsum('ii->i', eAw[:-1, 1:])[:] = exp_sd

            else:  # generic
                for _ in range(s):
                    eAw = eAw @ eAw

        # Zero out the entries from np.empty in case of triangular input
        if (lu[0] == 0) or (lu[1] == 0):
            eA[ind] = np.triu(eAw) if lu[0] == 0 else np.tril(eAw)
        else:
            eA[ind] = eAw

    return eA



def get_gate_matrix( n_qubits_gate, x):
    """Produces the matrix for this gate on its own."""
    sigma = get_norder_paulis( n_qubits_gate )
    sigma = (-1j / ( 2 ** n_qubits_gate )) * sigma
    H = dot_product( x, sigma )
    return expm( H )  # np.exp(H) #


def get_param_count (n_qubits_gate ):
    """Returns the number of the gate's input parameters."""
    return 4 ** n_qubits_gate

# Define the quantum circuit
dev = qml.device("default.qubit", wires=2)

# def to_unitary(parmas):
#     z = 1/np.sqrt(2)*parmas
#     q, r = np.linalg.qr(z)
#     d = r.diagonal()
#     q *= d/np.abs(d)
#     return q

@qml.qnode(dev)
def circuit(params):
    qml.RX(2, 0)
    qml.QubitUnitary(get_gate_matrix(n_qubits, params), wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# Define the objective function
def cost(params):
    return np.abs(circuit(params) - 1) ** 2

# Choose an optimizer
optimizer = qml.NesterovMomentumOptimizer()



# def to_unitary(matrix):
#     svd_u, _, svd_v = pnp.linalg.svd(matrix, full_matrices = False)
#     return pnp.dot(svd_u, svd_v)

# Initialize the parameters
n_qubits = 2
unitary = np.array(unitary_group.rvs(2**2), requires_grad = True, dtype = np.complex128)
params = np.random.rand(get_param_count(n_qubits)) * np.pi #* (1+0j)

# {
#     '1': [
#         np.array(unitary_group.rvs(2**2), requires_grad = True, dtype = np.complex128),
#         np.array(unitary_group.rvs(2**2), requires_grad = True, dtype = np.complex128),
#     ],
#     '2': [
#         np.array(unitary_group.rvs(2**2), requires_grad = True, dtype = np.complex128),
#         np.array(unitary_group.rvs(2**2), requires_grad = True, dtype = np.complex128),
#     ]
# }

# Optimize the circuit
num_iterations = 100
for i in range(num_iterations):
    params = optimizer.step(cost, params)
    
    # for key, unitaries in params.items():
        # for uni
    assert np.allclose(params @ np.conj(params.T), np.eye(4))
    print(params)
    
# Print the optimized parameters
print("Optimized parameters:", params)

# Run the optimized circuit
result = circuit(params)
print("Expectation value:", result)
print(params)


import pennylane as qml
from pennylane import numpy as np

from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config
from downstream.synthesis.wrong.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.wrong.tensor_network_op_ import layer_circuit_to_matrix
from scipy.stats import unitary_group
import random
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




# import numpy as np

# with qml.tape.QuantumTape(do_queue=False) as u_tape:
#     qml.CZ(wires=[0,1])

# def v_function(params):
#     qml.RZ(params[0], wires=2)
#     qml.RZ(params[1], wires=3)
#     qml.CNOT(wires=[2, 3])
#     qml.RZ(params[2], wires=3)
#     qml.CNOT(wires=[2, 3])

# dev = qml.device("default.qubit", wires=4)

# @qml.qnode(dev)
# def local_hilbert_test(v_params, v_function, v_wires, u_tape):
#     qml.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
#     return qml.probs(u_tape.wires + v_wires)

# def cost_lhst(parameters, v_function, v_wires, u_tape):
#     return (1 - local_hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape)[0])

# print(cost_lhst([3*np.pi/2, 3*np.pi/2, np.pi/2], v_function = v_function, v_wires = [1], u_tape = u_tape))




with qml.tape.QuantumTape(do_queue=False) as u_tape:
    # qml.Hadamard(wires=0)
    qml.Rot(1.1, 1.2, 1.3, wires=0)

def v_function(params):
    qml.Rot(1.1, 1.2, 1.3, wires=1)
    # qml.Hadamard(wires=1)
    # qml.RZ(params[0], wires=1)

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def hilbert_test(v_params, v_function, v_wires, u_tape):
    qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
    # qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
    return qml.probs(u_tape.wires + v_wires)

def cost_hst(parameters, v_function, v_wires, u_tape):
    return (1 - hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape)[0])

print(cost_hst([0], v_function = v_function, v_wires = [1], u_tape = u_tape))


# n_qubits = 2
# layer2gates = [
#     [
#         {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
#         {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
#     ],
#     [{'name': 'cz', 'qubits': [1,0], 'params': []}],
#     [
#         {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
#         {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
#     ],
#     # [{'name': 'cz', 'qubits': [0,1], 'params': []}],
#     # [
#     #     {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
#     #     {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
#     # ],
#     # [{'name': 'cz', 'qubits': [0,1], 'params': []}],
#     # [
#     #     {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
#     #     {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
#     # ],
#     # [{'name': 'cz', 'qubits': [1,0], 'params': []}],
#     # [
#     #     {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
#     #     {'name': 'u', 'qubits': [0], 'params': [random.random()*np.pi]*3},
#     # ],
# ]

n_qubits = 1
layer2gates = [
    [
        {'name': 'u', 'qubits': [0], 'params': [1.1, 1.2, 1.3]},
    ],
    [
        {'name': 'u', 'qubits': [0], 'params': [1.3, 1.4, 1.5]},
    ],
]

print(layer2gates)

param_size = 0
params = []
for layer in layer2gates:
    for gate in layer:
# for gate in circuit_info['gates']:
        param_size += len(gate['params'])
        params += gate['params']

original_params = params
params = jnp.array(params)

# U = jnp.array(unitary_group.rvs(2**n_qubits), dtype=jnp.complex128)
# U = layer_circuit_to_matrix(layer2gates, n_qubits)

U = layer_circuit_to_pennylane_tape(layer2gates)
U = qml.matrix(U)


# params = jax.random.normal(jax.random.PRNGKey(random.randint(0, 100)), (param_size,))

with qml.tape.QuantumTape(do_queue=False) as u_tape:
    qml.QubitUnitary(U, wires=list(range(n_qubits)))

def circuit(params):
    layer_circuit_to_pennylane_circuit(layer2gates, params = params, offest=n_qubits)

with qml.tape.QuantumTape(do_queue=False) as u_tape:
    # qml.QubitUnitary(U, wires=list(range(n_qubits)))
    layer_circuit_to_pennylane_circuit(layer2gates, params = params, offest=0)



dev = qml.device("default.qubit", wires=n_qubits*2)

@qml.qnode(dev, interface="jax")
def hilbert_test(params):
    # qml.HilbertSchmidt(params, v_function=circuit, v_wires=list(range(n_qubits, 2*n_qubits)), u_tape=u_tape)
    qml.LocalHilbertSchmidt(params, v_function=circuit, v_wires=[n_qubits-1], u_tape=u_tape)
    return qml.probs([0, n_qubits-1])
# qml.probs(list(range(2*n_qubits)))  # qml.expval

def cost_hst(params):
    return (1 - hilbert_test(params)[0])

opt = optax.adamw(learning_rate=1e-2)
opt_state = opt.init(params)

fig, ax = qml.draw_mpl(hilbert_test, show_all_wires = True)(params)
fig.show()


print('dist = ', cost_hst(params))
# for epoch in range(50):
#     # 
#     loss_value, gradient = jax.value_and_grad(cost_hst)(params, )
#     updates, opt_state = opt.update(gradient, opt_state, params)
#     params = optax.apply_updates(params, updates)
    
#     print(epoch, loss_value)   

U2 = layer_circuit_to_pennylane_tape(layer2gates, params)
U2 = qml.matrix(U2)


print(original_params)
print(params)
print(jnp.allclose(U, U2))
print(U)
print(U2)


# print()

print('finish')





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