import cloudpickle as pickle
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend
from circuit import gen_random_circuits
from upstream import RandomwalkModel
from downstream.synthesis.pennylane_op import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op import layer_circuit_to_matrix

from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from scipy.stats import unitary_group
import random

config.update("jax_enable_x64", True)

lr = 1e-2
max_epoch = 1000
n_qubits = 1
layer2gates = [
    [{
        'name': 'u',
        'qubits': [0],
        'params': [np.pi/3, np.pi/4, np.pi/5,]
    }]
]

param_size = 0
params = []
for layer in layer2gates:
    for gate in layer:
        param_size += len(gate['params'])
        params += gate['params']

params = jnp.array(params)
print('params', params)
# params = jax.random.normal(jax.random.PRNGKey(random.randint(0, 100)), (param_size,))

U = layer_circuit_to_pennylane_tape(layer2gates, params = params)
U = qml.matrix(U)
            
with qml.tape.QuantumTape(do_queue=False) as u_tape:
    qml.QubitUnitary(U, wires=list(range(n_qubits)))

def circuit(_params):
    layer_circuit_to_pennylane_circuit(layer2gates, params = params, offest=n_qubits)
    
dev = qml.device("default.qubit", wires=n_qubits*2)

@qml.qnode(dev, interface="jax")
def hilbert_test(params):
    qml.HilbertSchmidt(params, v_function=circuit, v_wires=list(range(n_qubits, 2*n_qubits)), u_tape=u_tape)
    return qml.probs(list(range(2*n_qubits)))  # qml.expval

def cost_hst(params):
    return (1 - hilbert_test(params)[0])

opt = optax.adamw(learning_rate=lr)
opt_state = opt.init(params)

print(cost_hst(params))

for epoch in range(max_epoch):
    loss_value, gradient = jax.value_and_grad(cost_hst)(params, )
    updates, opt_state = opt.update(gradient, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    print(epoch, loss_value, params)   

print('params', params)

print('finish')
