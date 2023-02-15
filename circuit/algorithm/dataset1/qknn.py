import numpy as np
import qiskit


class QKNN():
    def __init__(self,qubit_count):
        self.qubit_count = qubit_count
        self.quantum_register = qiskit.circuit.QuantumRegister(self.qubit_count)
        self.classical_register = qiskit.circuit.ClassicalRegister(1)
        self.circuit = qiskit.circuit.QuantumCircuit(self.quantum_register)
        self.data_load()
        self.swap_test()
        # self.circuit.measure(self.quantum_register[0],self.classical_register[0])


    def data_load(self):
        for qubit in list(self.quantum_register)[1:]:
            self.circuit.ry(np.random.rand()*np.pi,qubit)

    def swap_test(self):
        self.circuit.h(0)
        qubit_list = list(self.quantum_register)[1:]
        q1 = qubit_list[:len(qubit_list)//2]
        q2 = qubit_list[len(qubit_list)//2:]
        for q_1,q_2 in zip(q1,q2):
            self.circuit.cswap(self.quantum_register[0],q_1,q_2)
        self.circuit.h(0)


def get_cir(n_qubits):
    return QKNN(n_qubits).circuit


if __name__ == '__main__':
    print(get_cir(100))
