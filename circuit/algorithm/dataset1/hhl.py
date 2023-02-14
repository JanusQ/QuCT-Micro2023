import numpy as np
from qiskit.algorithms.linear_solvers.hhl import HHL


def get_cir(n_qubits):
    matrix = np.random.rand(2 ** n_qubits, 2 ** n_qubits)
    matrix = matrix + matrix.conj().T
    vector = np.random.randint(low=0, high=2, size=2 ** n_qubits)
    while np.sum(vector) == 0:
        vector = np.random.randint(low=0, high=2, size=2 ** n_qubits)
    circuit = HHL().construct_circuit(matrix, vector)
    return circuit


if __name__ == '__main__':
    print(get_cir(3).num_qubits)