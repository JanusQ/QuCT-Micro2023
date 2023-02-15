from unicodedata import name
from simulator.noise_free_simulator import simulate_noise_free
from simulator.noise_simulator import *
from qiskit import QuantumCircuit, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from utils.backend_info import coupling_map, initial_layout, max_qubit_num, default_basis_gates, single_qubit_fidelity, two_qubit_fidelity, readout_error
from qiskit.quantum_info.analysis import hellinger_fidelity

def naive_predict(circuit):
    fidelity = 1
    for instruction in circuit.data:
        # print(instruction)
        gate_type = instruction.operation.name
        # print(gate_type)
        operated_qubits = [_.index for _ in instruction.qubits]
        if gate_type == 'barrier':
            continue
        elif gate_type == 'measure':
            _redout_error = readout_error[operated_qubits[0]]
            fidelity *= (_redout_error[0][0] + _redout_error[1][1]) / 2
        elif gate_type in default_basis_single_gates:
            fidelity *= single_qubit_fidelity[operated_qubits[0]]
        elif gate_type in default_basis_two_gates:
            fidelity *= two_qubit_fidelity[tuple(operated_qubits)]
        else:
            raise Exception('unkown gate', instruction)
    return fidelity


circ = QuantumCircuit(max_qubit_num)
circ.h(0)
for q in range(1, max_qubit_num):
    circ.cx(0, q)
print(circ)

circ.measure_all()
circ = transpile(circ, basis_gates = default_basis_gates, coupling_map = coupling_map, initial_layout = initial_layout, optimization_level=0)

noisy_reuslt = simulate_noise(circ, 10000)
noise_free_result = simulate_noise_free(circ, 10000)

ground_truth = hellinger_fidelity(noisy_reuslt.get_counts(), noise_free_result.get_counts())
predict_fidelity = naive_predict(circ)
print(ground_truth, predict_fidelity)

plot_histogram(noisy_reuslt.get_counts())
plot_histogram(noise_free_result.get_counts())
plt.show()

