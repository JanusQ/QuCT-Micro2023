from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit

def get_cir(circuit: QuantumCircuit, n_qubits) -> QuantumCircuit:
    from qiskit.algorithms.amplitude_estimators.ae import AmplitudeEstimation
    from qiskit.algorithms.amplitude_estimators.estimation_problem import EstimationProblem
    return AmplitudeEstimation(n_qubits).construct_circuit(EstimationProblem(circuit, n_qubits))

print(get_cir(random_circuit(10, 10), 10))
