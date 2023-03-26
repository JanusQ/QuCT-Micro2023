from circuit import gen_random_circuits
from circuit.formatter import qiskit_to_my_format_circuit, layered_circuits_to_qiskit, get_layered_instructions
from upstream import RandomwalkModel
from collections import defaultdict

import math
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, gen_linear_topology, get_linear_neighbor_info, gen_fulllyconnected_topology

from qiskit import QuantumCircuit
import pennylane as qml
import numpy as np
from downstream.synthesis.pennylane_op_jax import layer_circuit_to_pennylane_circuit, layer_circuit_to_pennylane_tape
from downstream.synthesis.tensor_network_op_jax import layer_circuit_to_matrix

from scipy.stats import unitary_group

from downstream.synthesis.synthesis_model_pca_unitary_jax import SynthesisModelNN, SynthesisModelRandom, find_parmas, pkl_dump, pkl_load, matrix_distance_squared, SynthesisModel, synthesize
from itertools import combinations
import time
from qiskit import transpile
import random
import cloudpickle as pickle
from circuit.formatter import layered_circuits_to_qiskit, to_unitary
from qiskit.quantum_info import Operator

from utils.unitaries import qft_U, grover_U


def cnot_count(qc: QuantumCircuit):
    count_ops = qc.count_ops()
    if 'cx' in count_ops:
        return count_ops['cx']
    return 0


def cz_count(qc: QuantumCircuit):
    count_ops = qc.count_ops()
    if 'cz' in count_ops:
        return count_ops['cz']
    return 0


# n_qubits = 3
for n_qubits in range(3, 7):
    topology = gen_fulllyconnected_topology(n_qubits)
    neigh_info = gen_fulllyconnected_topology(n_qubits)

    # topology = gen_linear_topology(n_qubits)
    # neigh_info = get_linear_neighbor_info(n_qubits, 1)

    regen = False
    if regen:
        backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
                        basis_two_gates=['cz'], divide=False, decoupling=False)

        min_gate, max_gate = 2 * 4**n_qubits - 200, 2 * 4**n_qubits
        dataset = gen_random_circuits(min_gate=100, max_gate=max_gate, gate_num_step=max_gate//50, n_circuits=50,
                                    two_qubit_gate_probs=[2, 5], backend=backend, reverse=False, optimize=True, multi_process=True)

        upstream_model = RandomwalkModel(2, 20, backend)
        upstream_model.train(dataset, multi_process=True, remove_redundancy=False)
        synthesis_model = SynthesisModelNN(upstream_model, f'synthesis_{n_qubits}')
        data = synthesis_model.construct_data(dataset, multi_process=True)
        synthesis_model.construct_model(data)
        synthesis_model.save()
    else:
        synthesis_model: SynthesisModel = SynthesisModel.load(
            f'synthesis_{n_qubits}')
        backend: Backend = synthesis_model.backend

    # backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neigh_info, basis_single_gates=['u'],
    #                 basis_two_gates=['cz'], divide=False, decoupling=False)
    # synthesis_model: SynthesisModelRandom = SynthesisModelRandom(backend)

    # init_unitary_mat = qft_U(n_qubits)

    for index in range(5):
        init_unitary_mat = unitary_group.rvs(2**n_qubits)
        for use_heuristic in [True, False]:
            start_time = time.time()

            synthesized_circuit, cpu_time = synthesize(init_unitary_mat, backend=backend, allowed_dist=1e-2,
                                                    multi_process=False, heuristic_model=synthesis_model if use_heuristic else None,
                                                    verbose=True, lagre_block_penalty=4)
            # print(synthesized_circuit)
            synthesis_time = time.time() - start_time

            qiskit_circuit = layered_circuits_to_qiskit(
                n_qubits, synthesized_circuit, barrier=False)

            result = {
                'n_qubits': n_qubits,
                'U': init_unitary_mat,
                'qiskit circuit': qiskit_circuit,
                '#gate': len(qiskit_circuit),
                '#two-qubit gate': cnot_count(qiskit_circuit) + cz_count(qiskit_circuit),
                'depth': qiskit_circuit.depth(),
                'synthesis time': synthesis_time,
                'cpu time': cpu_time,
                'use heuristic': use_heuristic,
            }

            print(result)
            print('\n')

            with open(f'temp_data/synthesis_result/3_26/{n_qubits}_{use_heuristic}_{index}_result.pkl', 'wb') as f:
                pickle.dump(result, f)

        '''TODO: 能不能先用Clifford逼近了，再对电路结构优化'''
