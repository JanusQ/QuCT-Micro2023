from qiskit import QuantumCircuit

def simon_oracle(b):
    """returns a Simon oracle for bitstring b"""
    b = b[::-1] # reverse b for easy iteration
    n = len(b)
    qc = QuantumCircuit(n*2)
    # Do copy; |x>|0> -> |x>|x>
    for q in range(n):
        qc.cx(q, q+n)
    if '1' not in b:
        return qc  # 1:1 mapping, so just exit
    i = b.find('1') # index of first non-zero bit in b
    # Do |x> -> |s.x> on condition that q_i is 1
    for q in range(n):
        if b[q] == '1':
            qc.cx(i, (q)+n)
    return qc


def get_cir(bitstring: str) -> QuantumCircuit:

    n = len(bitstring)
    simon_circuit = QuantumCircuit(n * 2)

    # Apply Hadamard gates before querying the oracle
    simon_circuit.h(range(n))

    # Apply barrier for visual separation
    simon_circuit.barrier()

    simon_circuit = simon_circuit.compose(simon_oracle(bitstring))

    # Apply barrier for visual separation
    simon_circuit.barrier()

    # Apply Hadamard gates to the input register
    simon_circuit.h(range(n))

    return simon_circuit


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
    print(get_cir(get_bitstr(50)).num_qubits)
