import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

def qft_U(n: int):
    n = 2**n
    root = np.e ** (2j * np.pi / n)
    Q = np.array(np.fromfunction(lambda x,y: root**(x*y), (n,n))) / np.sqrt(n)
    return Q

def optimal_grover(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
    for i in range(num_qubits):
        qc.x(i)

    controls = [i for i in range(1, num_qubits)]
    target = 0
    qc.mcp(np.pi, controls, target)

    for i in range(num_qubits):
        qc.x(i)
    for i in range(num_qubits):
        qc.h(i)
    return qc

def grover_U(n: int):
    grover = optimal_grover(4)
    return Operator(grover).data