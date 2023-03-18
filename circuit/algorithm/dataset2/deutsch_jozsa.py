from qiskit import QuantumCircuit


def get_balanced_oracle(n, b_str):
    if len(b_str) != n:
        raise Exception("bitstring长度应和n保持一致")
    balanced_oracle = QuantumCircuit(n + 1)
    # Place X-gates
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            balanced_oracle.x(qubit)

    # Use barrier as divider
    balanced_oracle.barrier()

    # Controlled-NOT gates
    for qubit in range(n):
        balanced_oracle.cx(qubit, n)

    balanced_oracle.barrier()

    # Place X-gates
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            balanced_oracle.x(qubit)

    return balanced_oracle


def get_cir(n, b_str):
    dj_circuit = QuantumCircuit(n+1)

    # Apply H-gates
    for qubit in range(n):
        dj_circuit.h(qubit)

    # Put qubit in state |->
    dj_circuit.x(n)
    dj_circuit.h(n)

    # Add oracle
    
    dj_circuit = dj_circuit.compose(get_balanced_oracle(n, b_str))
    # dj_circuit += get_balanced_oracle(n, b_str)

    # Repeat H-gates
    for qubit in range(n):
        dj_circuit.h(qubit)
    dj_circuit.barrier()

    return dj_circuit

if __name__ == '__main__':
    import random
    def get_bitstr(n_qubits):
        b = ""
        for i in range(n_qubits):
            if random.random() > 0.5:
                b += '0'
            else:
                b += '1'
        return b
    print(get_cir(100, get_bitstr(100)).num_qubits)