import random

from circuit.algorithm.algorithm2_sys import Algorithm
from circuit.algorithm.dataset1_sys import hamiltonian_simulation, ising, qknn, qsvm, swap, vqe, QAOA_maxcut, grover
from circuit.algorithm.dataset2_sys import deutsch_jozsa, multiplier, qec_5_x, qnn, qugan, simon, square_root

from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.random import random_circuit

from circuit.algorithm.dataset1_sys import vqc
from circuit.parser import qiskit_to_layered_circuits
from utils.backend import Backend
from utils.backend_info import default_basis_gates


def get_data(id, qiskit_circuit, coupling_map, mirror, backend: Backend, trans=True):
    new_qiskit_circuit = QuantumCircuit(qiskit_circuit.num_qubits)
    for instruction in qiskit_circuit:
        if instruction.operation.name in ('id',):
            continue
        new_instruction = instruction.copy()
        new_instruction.qubits = ()
        qubit_list = []
        for qubit in instruction.qubits:
            if qubit.register.name != 'q':
                qubit_list.append(
                    Qubit(register=QuantumRegister(name='q', size=qubit.register.size), index=qubit.index))
            else:
                qubit_list.append(qubit)
        new_instruction.qubits = tuple(qubit_list)
        new_qiskit_circuit.append(new_instruction)

    if trans == True:
        qiskit_circuit = transpile(new_qiskit_circuit, basis_gates=backend.basis_gates, coupling_map=coupling_map,
                                   optimization_level=3) # , inst_map=list(i for i in range(18))
    else:
        qiskit_circuit = new_qiskit_circuit

    # print(id)
    
    # print(new_qiskit_circuit)
    
    # print(qiskit_circuit)    
    
    # print('\n\n')
    if mirror:
        qiskit_circuit = qiskit_circuit.compose(qiskit_circuit.inverse())

    circuit_info = qiskit_to_layered_circuits(qiskit_circuit)

    circuit_info.update({
        "id": id + '_' + str(qiskit_circuit.num_qubits),
        "alg_id": id
    })
    return circuit_info


def get_bitstr(n_qubits):
    b = ""
    for i in range(n_qubits):
        if random.random() > 0.5:
            b += '0'
        else:
            b += '1'
    return b


# from circuit.algorithm.dataset1 import grover


# def get_dataset_bug_detection(min_qubit_num, max_qubit_num, coupling_map, mirror):
#     dataset = []

#     assert min_qubit_num > 5 and max_qubit_num > min_qubit_num

#     for n_qubits in range(min_qubit_num, max_qubit_num):
#         al = Algorithm(n_qubits)
#         al2 = Algorithm(8)
#         dataset.append(
#             get_data(f'hamiltonian_simulation', hamiltonian_simulation.get_cir(n_qubits), coupling_map, mirror))
#         dataset.append(get_data(f'ising', ising.get_cir(n_qubits), coupling_map, mirror))
#         # algorithm.append(get_data(f'QAOA_maxcut', QAOA_maxcut.get_cir(n_qubits)))
#         dataset.append(get_data(f'qknn', qknn.get_cir(n_qubits), coupling_map, mirror))
#         dataset.append(get_data(f'qsvm', qsvm.get_cir(n_qubits), coupling_map, mirror))
#         dataset.append(get_data(f'vqc', vqc.get_cir(6), coupling_map, mirror))
#         dataset.append(get_data(f'qft', al2.qft(), coupling_map, mirror))
#         dataset.append(get_data(f'ghz', al.ghz(), coupling_map, mirror))
#         dataset.append(get_data(f'grover', grover.get_cir(n_qubits), coupling_map, mirror))
#         al = Algorithm(n_qubits - 1)
#         dataset.append(get_data(f'bernstein_vazirani', al.bernstein_vazirani(get_bitstr(n_qubits - 1)), coupling_map, mirror))
#         dataset.append(get_data(f'deutsch_jozsa', deutsch_jozsa.get_cir(n_qubits - 1, get_bitstr(n_qubits - 1)), coupling_map, mirror))
#         # if n_qubits % 5 == 0:
#         #     dataset.append(get_data(f'multiplier', multiplier.get_cir(n_qubits // 5), coupling_map, mirror))
#         # if n_qubits % 2 == 1:
#         #     dataset.append(get_data(f'qnn', qnn.get_cir(n_qubits), coupling_map, mirror))
#         #     dataset.append(get_data(f'qugan', qugan.get_cir(n_qubits), coupling_map, mirror))
#         #     dataset.append(get_data(f'swap', swap.get_cir(n_qubits), coupling_map, mirror))
#         # if n_qubits % 2 == 0:
#         #     dataset.append(get_data(f'simon', simon.get_cir(get_bitstr(n_qubits // 2)), coupling_map, mirror))

#     return dataset

# [4-7]
def get_dataset_alg_component(n_qubits, backend: Backend):
    dataset = []
    coupling_map = backend.coupling_map
    mirror = False

    al = Algorithm(n_qubits)
    # al2 = Algorithm(8)
    dataset.append(
        get_data(f'hamiltonian_simulation', hamiltonian_simulation.get_cir(n_qubits), coupling_map, mirror, backend))
    dataset.append(get_data(f'ising', ising.get_cir(n_qubits), coupling_map, mirror, backend))
    # dataset.append(get_data(f'QAOA_maxcut', QAOA_maxcut.get_cir(n_qubits), coupling_map, mirror))
    dataset.append(get_data(f'qknn', qknn.get_cir(n_qubits), coupling_map, mirror, backend))
    dataset.append(get_data(f'qsvm', qsvm.get_cir(n_qubits), coupling_map, mirror, backend))
    dataset.append(get_data(f'vqc', vqc.get_cir(n_qubits), coupling_map, mirror, backend))
    dataset.append(get_data(f'qft', al.qft(), coupling_map, mirror, backend))
    dataset.append(get_data(f'ghz', al.ghz(), coupling_map, mirror, backend))
    dataset.append(get_data(f'grover', grover.get_cir(n_qubits), coupling_map, mirror, backend))
    # al = Algorithm(n_qubits - 1)
    # dataset.append(get_data(f'bernstein_vazirani', al.bernstein_vazirani(get_bitstr(n_qubits - 1)), coupling_map, mirror))
    # dataset.append(get_data(f'deutsch_jozsa', deutsch_jozsa.get_cir(n_qubits - 1, get_bitstr(n_qubits - 1)), coupling_map, mirror))
    # if n_qubits % 5 == 0:
    #     dataset.append(get_data(f'multiplier', multiplier.get_cir(n_qubits // 5), coupling_map, mirror))
    # if n_qubits % 2 == 1:
    #     dataset.append(get_data(f'qnn', qnn.get_cir(n_qubits), coupling_map, mirror))
    #     dataset.append(get_data(f'qugan', qugan.get_cir(n_qubits), coupling_map, mirror))
    #     dataset.append(get_data(f'swap', swap.get_cir(n_qubits), coupling_map, mirror))
    # if n_qubits % 2 == 0:
    #     dataset.append(get_data(f'simon', simon.get_cir(get_bitstr(n_qubits // 2)), coupling_map, mirror))

    return dataset

# print(len(get_dataset()))
