from qiskit import transpile

from circuit.algorithm.get_data import get_dataset_bug_detection
from qiskit.circuit.random import random_circuit as qiskit_random_circuit

from circuit.algorithm.get_data import get_dataset_bug_detection
from circuit.parser import get_circuit_duration, qiskit_to_layered_circuits
from circuit.random_circuit import random_circuit, random_circuit_cycle
from utils.backend import Backend
import ray

def gen_random_circuits(min_gate: int, max_gate: int, n_circuits: int, two_qubit_gate_probs: list, backend: Backend, gate_num_step: int = 1,
                        reverse=True, optimize=False, multi_process=False, circuit_type='random'):
    dataset = []
    futures = []
    assert two_qubit_gate_probs[0] < two_qubit_gate_probs[1]
    for n_gates in range(min_gate, max_gate, gate_num_step):
        for prob in range(*two_qubit_gate_probs):
            prob *= .1
            if multi_process:
                futures.append(_gen_random_circuits_remote.remote(n_gates=n_gates, two_qubit_prob=prob,
                                                                  n_circuits=n_circuits, backend=backend, reverse=reverse,
                                                                  optimize=optimize, circuit_type=circuit_type))
            else:
                dataset += _gen_random_circuits(n_gates=n_gates, two_qubit_prob=prob,
                                                n_circuits=n_circuits, backend=backend, reverse=reverse,
                                                optimize=optimize, circuit_type=circuit_type)

    for future in futures:
        dataset += ray.get(future)

    # print(f'finish random circuit generation with {len(dataset)} circuits')
    return dataset


@ray.remote
def _gen_random_circuits_remote(n_gates=40, two_qubit_prob=0.5, n_circuits=2000, backend: Backend = None, reverse=True, optimize=False, circuit_type='random'):
    return _gen_random_circuits(n_gates=n_gates, two_qubit_prob=two_qubit_prob,
                                n_circuits=n_circuits, backend=backend, reverse=reverse,
                                optimize=optimize, circuit_type=circuit_type)


def _gen_random_circuits(n_gates=40, two_qubit_prob=0.5, n_circuits=2000, backend: Backend = None, reverse=True, optimize=False, circuit_type='random'):

    divide, decoupling, coupling_map, n_qubits = backend.divide, backend.decoupling, backend.coupling_map, backend.n_qubits
    basis_single_gates, basis_two_gates = backend.basis_single_gates, backend.basis_two_gates

    assert circuit_type in ('random', 'cycle', 'qiskit')

    # print(circuit)
    dataset = []
    for _ in range(n_circuits):

        if circuit_type == 'random':
            circuit = random_circuit(n_qubits, n_gates, two_qubit_prob, reverse=reverse, coupling_map=coupling_map,
                                     basis_single_gates=basis_single_gates, basis_two_gates=basis_two_gates)
        elif circuit_type == 'cycle':
            circuit = random_circuit_cycle(n_qubits, n_gates, two_qubit_prob, reverse=reverse, coupling_map=coupling_map,
                                           basis_single_gates=basis_single_gates, basis_two_gates=basis_two_gates)
        else:
            circuit = qiskit_random_circuit(n_qubits, n_gates // n_qubits, measure=True)


        # print(circuit)
        # try:
        if optimize:
            circuit = transpile(circuit, coupling_map=backend._true_coupling_map, optimization_level=3, basis_gates=(
                basis_single_gates+basis_two_gates), initial_layout=[qubit for qubit in range(n_qubits)])
        # except Exception as e:
        #     traceback.print_exc()

        # print(circuit)

        circuit_info = {
            'id': f'rc_{n_qubits}_{n_gates}_{two_qubit_prob}_{_}',
            'qiskit_circuit': circuit
        }

        dataset.append(circuit_info)

    new_dataset = []
    for _circuit_info in dataset:
        # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的
        circuit_info = qiskit_to_layered_circuits(
            _circuit_info['qiskit_circuit'], divide, decoupling)
        circuit_info['id'] = _circuit_info['id']

        circuit_info['duration'] = get_circuit_duration(
            circuit_info['layer2gates'], backend.single_qubit_gate_time, backend.two_qubit_gate_time)
        circuit_info['gate_num'] = len(circuit_info['gates'])
        circuit_info['devide_qubits'] = backend.devide_qubits
        circuit_info['two_qubit_prob'] = two_qubit_prob

        # fig = circuit_info['qiskit_circuit'].draw('mpl')
        # fig.savefig("devide_figure/"+str(_circuit_info['id'])+".svg")

        # 减少模型大小
        # print(circuit_info['qiskit_circuit'])
        # print(circuit_info['gate_num'])
        circuit_info['qiskit_circuit'] = None

        new_dataset.append(circuit_info)

    return new_dataset


def gen_algorithms(n_qubits, backend, mirror,trans = True):
    return get_dataset_bug_detection(n_qubits, n_qubits+1, backend, mirror, trans)
