import pennylane as qml
from pennylane import numpy as np

from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config

config.update("jax_enable_x64", True)




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
with qml.tape.QuantumTape(do_queue=False) as u_tape:
    # qml.SWAP(wires=[0, 1])
    # qml.Hadamard(wires=0)
    # qml.Hadamard(wires=1)
    # qml.QubitUnitary(U, wires=[0,1])
    qml.RZ(1.1, wires=0)
    qml.Hadamard(wires=0)
    qml.Identity(wires=1)

# with qml.tape.QuantumTape(do_queue=False) as U:
#     qml.SWAP(wires=[0, 1])

# Represents unitary V
# @qml.qnode(dev)
def v_function(params):
    # qml.RZ(params[0], wires=1)
    qml.RZ(1.1, wires=2)
    qml.Hadamard(wires=2)
    qml.Identity(wires=3)
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