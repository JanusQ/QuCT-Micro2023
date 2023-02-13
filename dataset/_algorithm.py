from qiskit import QuantumCircuit

# QFT
def get_qft_cir(n_qubits):
    from qiskit.circuit.library import QFT
    return QFT(n_qubits)

# GHZ
def get_ghz_cir(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc

# H初始化
def initialize_H(qc, n_qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in range(n_qubits):
        qc.h(q)
    return qc

# 扩散器
def grover(n_qubits, qc):
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(n_qubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(n_qubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(n_qubits - 1)
    qc.mct(list(range(n_qubits - 1)), n_qubits - 1)  # multi-controlled-toffoli
    qc.h(n_qubits - 1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(n_qubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(n_qubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    return qc

# 黑盒，随便设计的
def grover_oracle(n_qubits, qc):
    from qiskit.circuit.library import ZGate
    qc.append(ZGate().control(n_qubits - 1), range(n_qubits))
    return qc

# Grover算法
def get_grover_cir(n_qubits, iteration):
    # grover
    qc = QuantumCircuit(n_qubits)
    qc = initialize_H(qc, n_qubits)
    for _ in range(iteration):
        qc = grover_oracle(n_qubits, qc)
        qc = grover(n_qubits, qc)
    return qc

dataset = []
for n_qubits in range(2, 10):
    dataset.append({
        'id': f'grover_{n_qubits}_itr1',
        'qiskit_circuit': get_grover_cir(n_qubits, 1)
    })
    dataset.append({
        'id': f'qft_{n_qubits}',
        'qiskit_circuit': get_qft_cir(n_qubits)
    })
    dataset.append({
        'id': f'qft_{n_qubits}',
        'qiskit_circuit': get_ghz_cir(n_qubits)
    })