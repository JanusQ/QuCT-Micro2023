from unicodedata import name
from upstream.randomwalk_model import RandomwalkModel, add_pattern_error
from simulator.noise_free_simulator import simulate_noise_free
from simulator.noise_simulator import *
from qiskit import QuantumCircuit, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from simulator.hardware_info import coupling_map, initial_layout, max_qubit_num, basis_gates, single_qubit_fidelity, two_qubit_fidelity, readout_error
from qiskit.quantum_info.analysis import hellinger_fidelity
from dataset.random_circuit import random_circuit
from downstream.fidelity_predict.other import naive_predict

path = 'rwm.pkl'
model = RandomwalkModel.load(path)
for _ in range(100):
    circ = random_circuit(10, 10)  
    error_circ, succeed_add_error = add_pattern_error(circ, model)

    if not succeed_add_error:
        continue

    error_circ.measure_all()
    error_circ = transpile(error_circ, basis_gates = basis_gates, coupling_map = coupling_map, initial_layout = initial_layout, optimization_level=0)

    circ = transpile(circ, basis_gates = basis_gates, coupling_map = coupling_map, initial_layout = initial_layout, optimization_level=0)
    circ.measure_all()

    # print(circ)
    # print(error_circ)
    # print('\n\n')

    # print(circ)
    noisy_reuslt = simulate_noise(error_circ, 1000)
    noise_free_result = simulate_noise_free(circ, 10000)

    ground_truth = hellinger_fidelity(noisy_reuslt.get_counts(), noise_free_result.get_counts())
    predict_fidelity = naive_predict(circ) # 对于电路比较浅的预测是准的，大概是因为有可能翻转回去吧，所以电路深了之后会比理论的保真度要高一些
    print(ground_truth, predict_fidelity)

    # plot_histogram(noisy_reuslt.get_counts())
    # plot_histogram(noise_free_result.get_counts())
    # plt.show()
