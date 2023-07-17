import jax
from jax import numpy as jnp
import numpy as np
import optax
from jax import vmap

import tensorcircuit as tc
import numpy as np
from numpy import e, pi, cos
import pennylane as qml
import jax

# @jax.jit
# def _layer_circuit_to_matrix(gates, n_qubits, params) -> jax.numpy.array:
#     point = 0
    
#     circuit = tc.Circuit(n_qubits)
#     for gate in gates:
#         phi, theta, omega = params[point:point+3]
#         point+=3
#         u_gate = qml.Rot.compute_matrix(phi = phi, theta = theta, omega = omega)
        
#         for layer in layer2gates:
#             for gate in layer:
#                 # gate_qubits = [n_qubits - qubit - 1 for qubit in gate['qubits']]
#                 gate_qubits = gate['qubits']
#                 if gate['name'] == 'u':
#                     if params is None:
#                         phi, theta, omega = gate['params']
#                     else:
#                         phi, theta, omega = params[point:point+3]
#                         point+=3
#                     u_gate = qml.Rot.compute_matrix(phi = phi, theta = theta, omega = omega)
#                     circuit.any(*gate_qubits, unitary=u_gate)
#                     pass
#                 elif gate['name'] == 'cx':
#                     circuit.cnot(*gate_qubits)
#                 elif gate['name'] == 'cz':
#                     circuit.cz(*gate_qubits)
#                 else:
#                     raise Exception('Unkown gate type', gate)
    
#     return circuit.matrix()






A = jnp.array([0,1,2])
B = jnp.array([2,3,4])

def test(a, b):
    return b[a]

test(A[0], B)
result = vmap(test, in_axes=(0, None))(A, B)

A = jnp.array([0,1,2])
B = {
    0: 1,
    1: 2,
    2: 3,
}

def test(a, b):
    return b[a]

test(A[0], B)
result = vmap(test, in_axes=(0, None))(A, B)




def gen_data(x): 
    return 1/(12*x) + 1

def preidct(weights, x):
    return 1/(weights[0] * x) + weights[1]

X = jnp.arange(0, 100)
Y = gen_data(X)

weights = jax.random.normal(jax.random.PRNGKey(0), (2,))
opt = optax.adamw(learning_rate=1e-2)
opt_state = opt.init(weights)

for x, y in zip(X, Y):
    def loss(weights, x, y):
        return optax.l2_loss(preidct(weights, x) - y)
    
    grad_loss = jax.grad(loss)
    gradient = grad_loss(weights, x, y)
    updates, opt_state = opt.update(gradient, opt_state, weights)
    weights = optax.apply_updates(weights, updates)        

    loss_value = loss(weights, x, y)

grad_tanh = jax.grad(jax.numpy.tanh)

print(grad_tanh(0.2))


def func1(x): 
    return 1/x

grad_func1 = jax.grad(func1)
print(grad_func1(2.0))  # -1/x^2

def func2(x, y): 
    return 1/(x+y)

grad_func2 = jax.grad(func2, argnums=[0,1])
print(grad_func2(2.0, 1.0))



def func3(x, y): 
    return 1/(x[0] + x[1] + y)

grad_func3 = jax.grad(func3, argnums=[0,1])
print(grad_func3([2.0, 1.0], 1.0))