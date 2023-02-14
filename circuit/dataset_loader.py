from circuit.algorithm.get_data import get_dataset_bug_detection
from circuit.formatter import my_format_circuit_to_qiskit
from circuit.parser import qiskit_to_layered_circuits
from circuit.random_circuit import random_circuit, random_circuit_various_input
from circuit.utils import get_extra_info


def gen_random_circuits(max_qubit_num, min_gate, max_gate, n_circuits, devide, require_decoupling):
    dataset = []
    for n_gates in range(min_gate, max_gate, 5):
        for prob in range(4, 8):
            prob *= .1
            dataset += load_randomcircuits(n_qubits=max_qubit_num, n_gates=n_gates, two_qubit_prob=prob,
                                           n_circuits=n_circuits, devide=devide, require_decoupling=require_decoupling)

    dataset = get_extra_info(dataset)
    return dataset


def load_randomcircuits(n_qubits, n_gates=40, two_qubit_prob=0.5, n_circuits=2000, reverse=True, devide=True,
                        require_decoupling=True):
    dataset = [
        ({
            'id': f'rc_{n_qubits}_{n_gates}_{two_qubit_prob}_{_}',
            'qiskit_circuit': random_circuit(n_qubits, n_gates, two_qubit_prob, reverse=reverse)
        })
        for _ in range(n_circuits)
    ]

    new_dataset = []
    for elm in dataset:
        # circuit = elm['qiskit_circuit']
        # print(circuit)
        circuit_info = qiskit_to_layered_circuits(elm['qiskit_circuit'], devide,
                                                  require_decoupling)  # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的
        circuit_info['id'] = elm['id']
        circuit_info['qiskit_circuit'] = elm['qiskit_circuit']
        # elm.update(circuit_info)
        new_dataset.append(circuit_info)

    return new_dataset


def load_randomcircuits_various_input(n_qubits, n_gates, center, n_circuits, devide=True, require_decoupling=True):
    dataset = []
    for i in range(center):
        dataset += random_circuit_various_input(n_qubits, n_gates, n_circuits=n_circuits, two_qubit_prob=0.5)

    new_dataset = []
    for elm in dataset:
        # circuit = elm['qiskit_circuit']
        # print(circuit)

        circuit_info = qiskit_to_layered_circuits(elm, devide,
                                                  require_decoupling)  # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的
        circuit_info['qiskit_circuit'] = elm
        circuit_info['qiskit_circuit_devide'] = my_format_circuit_to_qiskit(n_qubits,
                                                                            circuit_info['layer2instructions'])
        # elm.update(circuit_info)
        new_dataset.append(circuit_info)

    return new_dataset

def gen_algorithms(n_qubits):
    return get_dataset_bug_detection(n_qubits, n_qubits+1)
