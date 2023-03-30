import qiskit


def state_preparation(circuit: qiskit.QuantumCircuit, qubits, cbits):
    # Prepare 3-bit code accross 3 quits ( I.e |111> + |000> )
    assert len(qubits) == 9
    circuit.h(qubits[0])
    circuit.cnot(qubits[0], qubits[3])
    circuit.cnot(qubits[0], qubits[6])
    circuit.h(qubits[0])
    circuit.h(qubits[3])
    circuit.h(qubits[6])
    # Now we do the error test
    circuit.cnot(qubits[0], qubits[1])
    circuit.cnot(qubits[0], qubits[2])
    circuit.cnot(qubits[3], qubits[4])
    circuit.cnot(qubits[3], qubits[5])
    circuit.cnot(qubits[6], qubits[7])
    circuit.cnot(qubits[6], qubits[8])


def z_error_detection(circuit: qiskit.QuantumCircuit, qubits,
                      ancilla_qubits, cbits: qiskit.ClassicalRegister):
    for anc_state, state_l in zip(range(0, 8, 2), range(0, 9, 3)):
        circuit.cnot(qubits[state_l], ancilla_qubits[anc_state])
        circuit.cnot(qubits[state_l + 1], ancilla_qubits[anc_state])
        circuit.cnot(qubits[state_l + 1], ancilla_qubits[anc_state + 1])
        circuit.cnot(qubits[state_l + 2], ancilla_qubits[anc_state + 1])
    # for indx in range(6):
    # circuit.measure(ancilla_qubits[indx], cbits[indx])


def x_error_detection(circuit: qiskit.QuantumCircuit, qbts,
                      ancilla_qubits, cbts: qiskit.ClassicalRegister):
    for qb in qbts:
        circuit.h(qb)
    for i in range(6):
        circuit.cnot(qbts[i], ancilla_qubits[6])
        circuit.cnot(qbts[i + 3], ancilla_qubits[7])
    # circuit.measure(ancilla_qubits[6],cbts[6])
    # circuit.measure(ancilla_qubits[7],cbts[7])
    for i in range(8):
        circuit.h(qbts[i])


def get_cir(k):
    qubits = list(range(9 * k))
    ancilla_qubits = list(range(9 * k, 9 * k + 8 * k))
    cbits = qiskit.ClassicalRegister(8 * k)
    circuit = qiskit.QuantumCircuit(17 * k)
    # circuit.add_register(ancilla_qubits)
    for i, j, k in zip(range(0, len(qubits), 9), range(0, len(ancilla_qubits), 8), range(0, len(cbits), 8)):
        state_preparation(circuit, qubits[i:i + 9], cbits[k:k + 8])
        z_error_detection(circuit, qubits[i:i + 9], ancilla_qubits[j:j + 8], cbits[k:k + 8])
        x_error_detection(circuit, qubits[i:i + 9], ancilla_qubits[j:j + 8], cbits[k:k + 8])
    return circuit

if __name__ == '__main__':
    print(get_cir(5).num_qubits)
