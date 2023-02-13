from collections import defaultdict
from operator import index
import random
from pattern_extractor.randomwalk_model import RandomwalkModel, add_pattern_error_path, Step, Path
from simulator.noise_free_simulator import simulate_noise_free
from simulator.noise_simulator import *
from qiskit import QuantumCircuit, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from simulator.hardware_info import coupling_map, initial_layout, max_qubit_num, basis_gates, single_qubit_fidelity, two_qubit_fidelity, readout_error
from qiskit.quantum_info.analysis import hellinger_fidelity
from dataset.random_circuit import one_layer_random_circuit, random_circuit
from analysis.predict_fidelity import naive_predict
from analysis.dimensionality_reduction import batch
import numpy as np
from dataset.dataset_loader import load_algorithms, load_randomcircuits

from sklearn.utils import shuffle

from jax import numpy as jnp
import jax
from jax import grad, jit, vmap, pmap
import optax

model_path = 'rwm_5qubit.pkl'


# TODO:测试下多大规模的比较合理 
dataset = []
for n_gates in range(2, 20, 5):
    for prob in range(3, 8):
        prob = prob * 0.1
        # dataset += load_randomcircuits(n_qubits = max_qubit_num, n_gates = 20, two_qubit_prob = prob, n_circuits = 2000)
        dataset += load_randomcircuits(n_qubits = max_qubit_num, n_gates = n_gates, two_qubit_prob = prob, n_circuits = 100)

print(f'generate {len(dataset)} circuits')

model = RandomwalkModel(1, 20)  #max-step=2 会有14000维，max-step 也有10000维，减少生成的特征的数量``
# # 还是需要降维的比如基于一些相似性啥的
model.train(dataset)
print(f'succeed in training, and generate {len(model.hash_table)} path type.')

# model.load_reduced_vecs()
# print('succeeded in mds')
# # 怎么评判向量化的效果，KL散度？但是感觉没啥可以解释的

# for circuit_info in dataset:
#     qiskit_circuit = circuit_info['qiskit_circuit'].copy()
#     qiskit_circuit.measure_all()

#     # results = simulate_noise_free(qiskit_circuit)
#     # print(results)
#     # assert len(results) == 1 and '0'*max_qubit_num in results

#     error_circuit, n_erroneous_patterns = add_pattern_error_path(circuit_info, model)
#     error_circuit.measure_all()
#     circuit_info['error_circuit'] = error_circuit
#     circuit_info['n_erroneous_patterns'] = n_erroneous_patterns

# noisy_dataset = [circuit_info for circuit_info in dataset if circuit_info['n_erroneous_patterns'] != 0]
# print(f'len(noisy_dataset): {len(noisy_dataset)}')

# # print('start simulate')
# results = [simulate_noise_free(circuit_info['error_circuit']) for circuit_info in dataset]
# true_result = {
#     '0'*max_qubit_num: 2000
# }
# for i, circuit_info in enumerate(dataset):
#     circuit_info['error_result'] = results[i]
#     circuit_info['ground_truth_fidelity'] = hellinger_fidelity(circuit_info['error_result'], true_result)

# model.save(model_path)
# len(dataset)
# print('succeeded in saving')
