from numpy import pi
from qiskit import QuantumCircuit


class Algorithm:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        
    def qft_rotations(self, circuit, n):
        """Performs qft on the first n qubits in circuit (without swaps)"""
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cp(pi / 2 ** (n - qubit), qubit, n)
        # At the end of our function, we call the same function again on
        # the next qubits (we reduced n by one earlier in the function)
        self.qft_rotations(circuit, n)

    def swap_registers(self, circuit, n):
        for qubit in range(n // 2):
            circuit.swap(qubit, n - qubit - 1)
        return circuit

    # QFT
    def qft(self):
        qc = QuantumCircuit(self.n_qubits)
        self.qft_rotations(qc, self.n_qubits)
        self.swap_registers(qc, self.n_qubits)
        return qc
    # # QFT
    # def qft(self):
    #     from qiskit.circuit.library import QFT
    #     return QFT(self.n_qubits)

    # GHZ
    def ghz(self):
        qc = QuantumCircuit(self.n_qubits)
        qc.h(0)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    # H初始化
    def _initialize_s(self, qc):
        """Apply a H-gate to 'qubits' in qc"""
        for q in range(self.n_qubits):
            qc.h(q)
        return qc

    # 黑盒，随便设计的
    def grover_oracle(self, target: str):
        from qiskit.quantum_info import Statevector
        from qiskit.algorithms import AmplificationProblem
        qc = AmplificationProblem(Statevector.from_label(target)).grover_operator.oracle.decompose()
        return qc

    def amplitude_amplification(self, target: str):
        from qiskit.quantum_info import Statevector
        from qiskit.algorithms import AmplificationProblem
        qc = AmplificationProblem(Statevector.from_label(target)).grover_operator.decompose()
        return qc

    # Grover算法
    def grover(self, target: str, iterations=1):
        # grover
        qc = QuantumCircuit(self.n_qubits)
        qc = self._initialize_s(qc)
        for _ in range(iterations):
            qc.append(self.amplitude_amplification(target=target), range(self.n_qubits))
        return qc

    @staticmethod
    def shor(N):
        from qiskit.algorithms.factorizers.shor import Shor
        shor = Shor()
        return shor.construct_circuit(N)

    def phase_estimation(self, U: QuantumCircuit):
        from qiskit.algorithms.phase_estimators.phase_estimation import PhaseEstimation
        pe = PhaseEstimation(self.n_qubits)
        return pe.construct_circuit(U)

    def bernstein_vazirani(self, s: str):
        bv_circuit = QuantumCircuit(self.n_qubits +1)

        # put auxiliary in state |->
        bv_circuit.h(self.n_qubits)
        bv_circuit.z(self.n_qubits)

        # Apply Hadamard gates before querying the oracle
        for i in range(self.n_qubits):
            bv_circuit.h(i)

        # Apply barrier
        bv_circuit.barrier()

        # Apply the inner-product oracle
        s = s[::-1]  # reverse s to fit qiskit's qubit ordering
        for q in range(self.n_qubits):
            if s[q] == '0':
                bv_circuit.i(q)
            else:
                bv_circuit.cx(q, self.n_qubits)

        # Apply barrier
        bv_circuit.barrier()

        # Apply Hadamard gates after querying the oracle
        for i in range(self.n_qubits):
            bv_circuit.h(i)

        return bv_circuit

    def qft_inverse(self, circuit: QuantumCircuit, n: int) -> QuantumCircuit:
        """ Applies the inverse of the Quantum Fourier Transform on the first n qubits in the given circuit. """
        for qubit in range(n // 2):
            circuit.swap(qubit, n - qubit - 1)
        for j in range(n):
            for m in range(j):
                circuit.cp(-np.pi / float(2 ** (j - m)), m, j)
            circuit.h(j)

        return circuit



from qiskit import QuantumCircuit
import numpy as np
from qiskit.circuit import ParameterVector


def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

from qiskit.circuit.library import ZFeatureMap

feature_map = ZFeatureMap(8)

def QCNN():
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)

    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combining the feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    return circuit

# print(Algorithm(7).phase_estimation(random_circuit(7,7)))
