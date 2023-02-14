import random

import numpy as np
import qiskit


class VQE:
    def __init__(self, k):
        self.PAULI_MATRICES = {'Z': np.array([[1, 0], [0, -1]]),
                               'X': np.array([[0, 1], [1, 0]]),
                               'Y': np.array([[0, 0 - 1j], [0 + 1j, 0]]),
                               'I': np.array([[1, 0], [0, 1]])}
        self.k = k
        self.circuit = None
        self.hamiltonian = np.zeros((2 ** k, 2 ** k))
        self.generate_random_hamiltonian_matrix()
        self.generate_trainable_circuit()

    def generate_random_hamiltonian_matrix(self):
        weights = np.random.randint(10, size=10)
        for weight in weights:
            new_matrix = 1
            for i in range(self.k):
                new_matrix = np.kron(new_matrix, self.PAULI_MATRICES[self.z_or_i()])
            self.hamiltonian += new_matrix * weight * 0.5

    def z_or_i(self):
        p = 0.5
        if random.random() > p:
            return "Z"
        else:
            return "I"

    def generate_trainable_circuit(self):
        self.circuit = qiskit.circuit.library.EfficientSU2(num_qubits=self.k, entanglement='linear')
        self.circuit = qiskit.compiler.transpile(self.circuit, basis_gates=['h', 'rx', 'ry', 'rz', 'cz'])
        n_param = self.circuit.num_parameters
        self.circuit = self.circuit.assign_parameters(np.random.rand(n_param) * np.pi)


def get_cir(n_qubits):
    return VQE(n_qubits).circuit

if __name__ == '__main__':
    print(get_cir(10))
