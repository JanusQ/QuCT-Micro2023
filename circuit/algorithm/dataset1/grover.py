from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate


def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in range(qubits):
        qc.h(q)
    return qc

def oracle(qc, nqubits):
    if nqubits < 5:
        qc.append(ZGate().control(nqubits - 1), range(nqubits))
    else:
        qc.append(ZGate().control(4), range(5))
    return qc

def diffuser(qc, nqubits):
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    return qc

def get_cir(n_qubits):
    qc = QuantumCircuit(n_qubits)
    initialize_s(qc, n_qubits)
    oracle(qc, n_qubits)
    diffuser(qc, n_qubits)
    return qc


# if __name__ == '__main__':
#     from circuit.InternetComputing.run_circuit import transpile_circuit

#     x = [i for i in range(3, 10)]
#     y = [len(transpile_circuit(get_cir(x_i))) for x_i in x]
#     import matplotlib.pyplot as plt

#     plt.plot(x, y)
#     plt.show()