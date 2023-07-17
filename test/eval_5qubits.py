import copy
import pickle
from jax import numpy as jnp
from jax import vmap
import jax
import numpy as np
from numpy import random
import time
from qiskit.quantum_info.analysis import hellinger_fidelity
import matplotlib.pyplot as plt
from copy import deepcopy
from circuit.dataset_loader import gen_random_circuits
from circuit.formatter import layered_circuits_to_qiskit, get_layered_instructions, qiskit_to_my_format_circuit
from downstream.fidelity_predict.fidelity_analysis import smart_predict, error_param_rescale
from simulator.noise_free_simulator import simulate_noise_free
from simulator.noise_simulator import NoiseSimulator
from test_opt import count_error_path_num
from upstream.randomwalk_model import RandomwalkModel, extract_device
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_circuit_layout
from collections import defaultdict
from circuit.parser import get_circuit_duration
from utils.backend import Backend, gen_linear_topology, get_linear_neighbor_info
from qiskit.transpiler.preset_passmanagers import (
    level_0_pass_manager,
    level_1_pass_manager,
    level_2_pass_manager,
    level_3_pass_manager,
)
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.compiler.transpiler import _parse_coupling_map, _parse_initial_layout
from plot.plot import plot_duration_fidelity, plot_top_ratio, find_error_path, plot_correlation, plot_real_predicted_fidelity
from simulator import NoiseSimulator, get_random_erroneous_pattern
from downstream.fidelity_predict.baseline.rb import get_errors as get_errors_rb
from circuit.parser import get_circuit_duration
from qiskit.transpiler.passes import CrosstalkAdaptiveSchedule
from qiskit.providers.models import BackendProperties
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit.providers.fake_provider import FakeHanoi, FakeAlmaden, FakeTokyo, FakeNairobi, FakeQuito  # 需要时v1版本的
# backend = FakeQuito()
# backend_property = backend.properties()
# print(backend_property.to_dict())

# with open('simulate_50_350/200/rb_error.pkl', 'rb') as file:
#     obj = pickle.load(file)
# print(obj)


def sumarize_datasize(dataset, name):
    data = []
    labels = []
    for circuit_info in dataset:
        data.append([circuit_info['n_erroneous_patterns'], len(circuit_info['gates']), len(circuit_info['layer2gates']),
                    circuit_info['n_erroneous_patterns'] /
                    len(circuit_info['gates']), circuit_info['two_qubit_prob'],
                    circuit_info['ground_truth_fidelity'], circuit_info['independent_fidelity'], circuit_info[
                    'ground_truth_fidelity'] - circuit_info['independent_fidelity'],
        ])
        labels.append(circuit_info['label'])

    random.shuffle(data)
    data = data[:3000]  # 太大了画出来的图太密了
    data = np.array(data)
    plot_correlation(data, ['n_erroneous_patterns',
                            'n_gates', 'depth', 'error_prop', 'two_qubit_prob', 'ground_truth_fidelity', 'independent_fidelity',
                            'ground_truth_fidelity - independent_fidelity'], color_features=labels, name=name)

# backend_property = FakeQuito().properties()
# qiskit_backend = BackendProperties(backend_name = '', backend_version = '1.1.1', last_update_date= '', qubits = list(range(backend.n_qubits)),
#                                    gates = backend.basis_gates, general=[])


def schedule_crosstalk(backend: Backend, rb_error: list, crosstalk_prop: list):
    class QiskitBackendProperty():
        def __init__(self, backend: Backend) -> None:
            self.backend = backend
            self.qubits = list(range(backend.n_qubits))

            class _Gate():
                def __init__(self, gate, qubits) -> None:
                    self.gate = gate
                    self.qubits = list(qubits)

            self.gates = [
                _Gate(gate, qubits)
                for gate in backend.basis_two_gates
                for qubits in backend.coupling_map
            ] + [
                _Gate(gate, [qubit])
                for gate in backend.basis_single_gates
                for qubit in self.qubits
            ]
            return

        def t1(self, qubit):
            return self.backend.qubit2T1[qubit]

        def t2(self, qubit):
            return self.backend.qubit2T2[qubit]

        def gate_length(self, gate, qubits):
            if gate in self.backend.basis_single_gates or gate in ('u1', 'u2', 'u3'):
                return self.backend.single_qubit_gate_time
            elif gate in self.backend.basis_two_gates:
                return self.backend.two_qubit_gate_time

        def gate_error(self, gate, qubits):
            if gate in self.backend.basis_single_gates or gate in ('u1', 'u2', 'u3'):
                return self.backend.rb_error[0][qubits]
            elif gate in self.backend.basis_two_gates:
                return self.backend.rb_error[1][tuple(qubits)]
            else:
                raise Exception('known', gate, qubits)

    # 门都还得转成 'u1', 'u2', 'u3'
    def to_crosstalk_structure(layer2gates):
        layer2gates = copy.deepcopy(layer2gates)
        for layer in layer2gates:
            for gate in layer:
                if gate['name'] == 'rz':
                    gate['name'] = 'u1'
                if gate['name'] == 'rx':
                    gate['name'] = 'u2'
                    gate['params'] = gate['params'] + [0]
                if gate['name'] == 'ry':
                    gate['name'] = 'u3'
                    gate['params'] = gate['params'] + [0, 0]
                if gate['name'] == 'cz':
                    gate['name'] = 'cx'
        return layer2gates

    def from_crosstalk_structure(layer2gates):
        layer2gates = copy.deepcopy(layer2gates)
        for layer in layer2gates:
            for gate in layer:
                if gate['name'] == 'u1':
                    gate['name'] = 'rz'
                    gate['params'] = gate['params'][:1]
                if gate['name'] == 'u1':
                    gate['name'] = 'rx'
                    gate['params'] = gate['params'][:1]
                if gate['name'] == 'u3':
                    gate['name'] = 'ry'
                    gate['params'] = gate['params'][:1]
                if gate['name'] == 'cx':
                    gate['name'] = 'cz'
        return layer2gates

    '''这个使用rb跑出来的'''
    two_qubit_error = rb_error[1]
    backend.rb_error = (rb_error[0],
                        {tuple(coupler): two_qubit_error[index]
                         for index, coupler in enumerate(backend.coupling_map)})

    backend_property = QiskitBackendProperty(backend)
    crosstalk_scheduler = CrosstalkAdaptiveSchedule(
        backend_property, crosstalk_prop=crosstalk_prop)

    optimized_circuits = []
    for circuit_info in dataset:
        qiskit_circuit = layered_circuits_to_qiskit(
            n_qubits, to_crosstalk_structure(circuit_info['layer2gates']))
        # transpiled_circuit = transpile()
        transpiled_circuit = dag_to_circuit(crosstalk_scheduler.run(
            circuit_to_dag(qiskit_circuit)))

        transpiled_circuit = qiskit_to_my_format_circuit(
            get_layered_instructions(transpiled_circuit)[0])
        transpiled_circuit = from_crosstalk_structure(
            circuit_info['layer2gates'])

        optimized_circuits.append(transpiled_circuit)
        '''check一下对不对'''
        # print(transpiled_circuit)
        # print(simulate_noise_free(
        #     layered_circuits_to_qiskit(n_qubits, transpiled_circuit)))

    return optimized_circuits


'''这里得用cx不然那个垃圾scheduler跑不了'''
n_qubits = 5
backend = Backend(n_qubits=5, topology=gen_linear_topology(
    n_qubits), neighbor_info=get_linear_neighbor_info(n_qubits, 1), basis_two_gates=['cx'])

dataset = gen_random_circuits(min_gate=20, max_gate=300, n_circuits=10, two_qubit_gate_probs=[
    3, 7], gate_num_step=20, backend=backend, multi_process=True, circuit_type='random')
for elm in dataset:
    elm['label'] = 'train'

upstream_model = RandomwalkModel(
    1, 20, backend=backend, travel_directions=('parallel', 'former'))

upstream_model.train(dataset, multi_process=True,
                     remove_redundancy=False)
erroneous_pattern = get_random_erroneous_pattern(
    upstream_model, error_pattern_num_per_device=3)
upstream_model.erroneous_pattern = erroneous_pattern


# {(0, 1): {(2, 3): 0.2, (2): 0.15}, (2, 3): {(0, 1): 0.05, }}
crosstalk_prop = defaultdict(lambda: {})

for paths in erroneous_pattern.values():
    for path in paths:
        path = path.split('-')
        if path[1] != 'parallel':
            continue
        path = [RandomwalkModel.parse_gate_info(elm)['qubits'] for elm in path]
        crosstalk_prop[tuple(path[0])][tuple(path[2])] = 0.1

optimized_circuits = schedule_crosstalk(backend,
                                        ([0.0006551527261505729, 0.0006501014815832324, 0.000668367029294724, 0.0006419032609345003, 
                                          0.0006580083828881741],
                                         [0.008100073360988247, 0.011374481901539833, 0.009733894319475992, 0.00473000004144141]),
                                        crosstalk_prop)

simulator = NoiseSimulator(backend)

rb_fidelity = get_errors_rb(backend, simulator, upstream_model, True)
# print(rb_fidelity)

simulator.get_error_results(dataset, upstream_model,
                            multi_process=True, erroneous_pattern=erroneous_pattern)

sumarize_datasize(dataset, 'q5')
