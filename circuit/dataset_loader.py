from circuit.algorithm.get_data import get_dataset_bug_detection
from circuit.formatter import layered_circuits_to_qiskit
from circuit.parser import get_circuit_duration, qiskit_to_layered_circuits
from circuit.random_circuit import random_circuit, random_circuit_various_input
from utils.backend import Backend


def gen_random_circuits(min_gate: int, max_gate: int, n_circuits: int, two_qubit_gate_probs: list, backend: Backend):
    dataset = []
    for n_gates in range(min_gate, max_gate, 5):
        for prob in range(*two_qubit_gate_probs):
            prob *= .1
            dataset += _gen_random_circuits(n_gates=n_gates, two_qubit_prob=prob,
                                           n_circuits=n_circuits, backend = backend)
    
    print(f'finish random circuit generation with {len(dataset)} circuits')
    return dataset


def _gen_random_circuits(n_gates=40, two_qubit_prob=0.5, n_circuits=2000, backend: Backend = None, reverse=True):
    
    divide, decoupling, coupling_map, n_qubits = backend.divide, backend.decoupling, backend.coupling_map, backend.n_qubits
    basis_single_gates, basis_two_gates = backend.basis_single_gates, backend.basis_two_gates
    
    dataset = [
        ({
            'id': f'rc_{n_qubits}_{n_gates}_{two_qubit_prob}_{_}',
            'qiskit_circuit': random_circuit(n_qubits, n_gates, two_qubit_prob, reverse=reverse, coupling_map=coupling_map,
                                             basis_single_gates=basis_single_gates, basis_two_gates=basis_two_gates)
        })
        for _ in range(n_circuits)
    ]

    new_dataset = []
    for _circuit_info in dataset:
        # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的
        circuit_info = qiskit_to_layered_circuits(_circuit_info['qiskit_circuit'], divide, decoupling)
        circuit_info['id'] = _circuit_info['id']

        circuit_info['duration'] = get_circuit_duration(circuit_info['layer2gates'], backend.single_qubit_gate_time, backend.two_qubit_gate_time)
        circuit_info['gate_num'] = len(circuit_info['gates'])
        
        new_dataset.append(circuit_info)

    return new_dataset


def gen_algorithms(n_qubits):
    return get_dataset_bug_detection(n_qubits, n_qubits+1)
