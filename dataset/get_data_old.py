import random

from qiskit.circuit import Qubit
from qiskit.circuit.random import random_circuit

from dataset.algorithm2 import Algorithm, QCNN
from dataset.dataset_loader import parse_circuit
from dataset.dataset1 import hamiltonian_simulation, ising, qknn, qsvm, swap, vqc, \
    vqe, hhl, QAOA_maxcut
from dataset.dataset2 import basic_teleportation, deutsch_jozsa, multiplier, phase_kickback, \
    qec_5_x, qnn, quantum_counting, qugan, simon, square_root
from qiskit import QuantumCircuit, transpile, QuantumRegister
from simulator.hardware_info import basis_gates
import math

def get_data(id, qiskit_circuit, trans = True):
    '''trans: 是否transpile'''
    new_qiskit_circuit = QuantumCircuit(qiskit_circuit.num_qubits)
    for instruction in qiskit_circuit:
        if instruction.operation.name in  ('id', ):
            continue
        new_instruction = instruction.copy()
        new_instruction.qubits = ()
        qubit_list = []
        for qubit in instruction.qubits:
            if qubit.register.name != 'q':
                qubit_list.append(Qubit(register=QuantumRegister(name='q', size=qubit.register.size), index=qubit.index))
            else:
                qubit_list.append(qubit)
        new_instruction.qubits = tuple(qubit_list)
        new_qiskit_circuit.append(new_instruction)

    if trans == True:
        qiskit_circuit = transpile(new_qiskit_circuit, basis_gates = basis_gates, optimization_level=0)
    else:
        qiskit_circuit = new_qiskit_circuit

    circuit_info = parse_circuit(qiskit_circuit)
    
    circuit_info.update({
        "id": id,
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


def get_dataset_bug_detection(min_qubit_num, max_qubit_num):
    dataset = []

    assert min_qubit_num > 5 and max_qubit_num > min_qubit_num, 'min_qubit_num > 5 and max_qubit_num > min_qubit_num'
    
    for n_qubits in range(min_qubit_num, max_qubit_num): # 大于3
        # al = Algorithm(n_qubits)
        dataset.append(get_data(f'hamiltonian_simulation_{n_qubits}', hamiltonian_simulation.get_cir(n_qubits), trans=False))
        dataset[len(dataset)-1]['alg_id'] = 'hamiltonian_simulation'
        # dataset.append(get_data(f'ising_{n_qubits}', ising.get_cir(n_qubits), trans=False))
        # dataset.append(get_data(f'QAOA_maxcut_{n_qubits}', QAOA_maxcut.get_cir(n_qubits), trans=False))
        dataset.append(get_data(f'qknn_{n_qubits}', qknn.get_cir(n_qubits), trans=False))
        dataset[len(dataset)-1]['alg_id'] = 'qknn'
        
        dataset.append(get_data(f'qsvm_{n_qubits}', qsvm.get_cir(n_qubits), trans=False))
        dataset[len(dataset)-1]['alg_id'] = 'qsvm'
        
        dataset.append(get_data(f'vqc_{n_qubits}', vqc.get_cir(n_qubits), trans=False))
        dataset[len(dataset)-1]['alg_id'] = 'vqc'
    
    odd = math.ceil(min_qubit_num / 2) * 2 + 1
    for n_qubits in range(odd, max_qubit_num, 2):  #[3, 5, 7, 9]:  # 大于3，奇数
        # dataset.append(get_data(f'qugan_{n_qubits}', qugan.get_cir(n_qubits)))
        dataset.append(get_data(f'swap_{n_qubits}', swap.get_cir(n_qubits), trans=False))
        dataset[len(dataset)-1]['alg_id'] = 'swap'

    for n_qubits in range(odd, max_qubit_num, 2): #[5, 7, 9]:
        dataset.append(get_data(f'qnn_{n_qubits}', qnn.get_cir(n_qubits), trans=False))
        dataset[len(dataset)-1]['alg_id'] = 'qnn'
        
        dataset.append(get_data(f'qugan_{n_qubits}', qugan.get_cir(n_qubits), trans=False))
        dataset[len(dataset)-1]['alg_id'] = 'qugan'

    even = math.ceil(min_qubit_num / 2) * 2
    for n_qubits in range(even, max_qubit_num, 2): #[2, 4, 6, 8, 10]:
        dataset.append(get_data(f'simon_{n_qubits}', simon.get_cir(get_bitstr(n_qubits // 2)), trans=False))
        dataset[len(dataset)-1]['alg_id'] = 'simon'

    for n_qubits in range(min_qubit_num, max_qubit_num): #range(2, 10): #必须大于2
        dataset.append(get_data(f'deutsch_jozsa_{n_qubits + 1}', deutsch_jozsa.get_cir(n_qubits, get_bitstr(n_qubits)), trans=False))
        dataset[len(dataset)-1]['alg_id'] = 'deutsch_jozsa'

    # triple = math.ceil(min_qubit_num / 3) * 3
    # for n_qubits in range(triple, max_qubit_num, 3): #[6, 9]: #3的倍数
    #     dataset.append(get_data(f'square_root_{n_qubits}', square_root.get_cir(n_qubits), trans=False))
    #     dataset[len(dataset)-1]['alg_id'] = 'square_root'
    
    forth = math.ceil(min_qubit_num / 5) * 5
    for n_qubits in range(forth, max_qubit_num, 5): #[1, 2]: #5的倍数
        dataset.append(get_data(f'multiplier_{forth}', multiplier.get_cir(n_qubits//5), trans=False))
        dataset[len(dataset)-1]['alg_id'] = 'multiplier'
        
        dataset.append(get_data(f'qec_5_x_{forth}', qec_5_x.get_cir(n_qubits//5)))
        dataset[len(dataset)-1]['alg_id'] = 'qec'
    
    return dataset


def get_dataset():
    dataset = []

    for n_qubits in range(3, 11):
        # print(n_qubits)
        al = Algorithm(n_qubits)
        dataset.append(get_data(f'hamiltonian_simulation_{n_qubits}', hamiltonian_simulation.get_cir(n_qubits)))
        dataset.append(get_data(f'ising_{n_qubits}', ising.get_cir(n_qubits)))
        dataset.append(get_data(f'QAOA_maxcut_{n_qubits}', QAOA_maxcut.get_cir(n_qubits)))
        dataset.append(get_data(f'qknn_{n_qubits}', qknn.get_cir(n_qubits)))
        dataset.append(get_data(f'qsvm_{n_qubits}', qsvm.get_cir(n_qubits)))
        dataset.append(get_data(f'vqc_{n_qubits}', vqc.get_cir(n_qubits)))
        dataset.append(get_data(f'vqe_{n_qubits}', vqe.get_cir(n_qubits)))
        dataset.append(get_data(f'qft_{n_qubits}', al.qft()))
        dataset.append(get_data(f'ghz_{n_qubits}', al.ghz()))
        dataset.append(get_data(f'grover_oracle_{n_qubits}', al.grover_oracle(get_bitstr(n_qubits))))
        dataset.append(get_data(f'amplitude_amplification_{n_qubits}', al.amplitude_amplification(get_bitstr(n_qubits))))
        dataset.append(get_data(f'grover_{n_qubits}', al.grover(get_bitstr(n_qubits))))
        # dataset.append(get_data(f'phase_estimation_{n_qubits}', al.phase_estimation(random_circuit(n_qubits, n_qubits))))
        dataset.append(get_data(f'bernstein_vazirani_{n_qubits}', al.bernstein_vazirani(get_bitstr(n_qubits))))
        dataset.append(get_data(f'qft_inverse_{n_qubits}', al.qft_inverse(random_circuit(n_qubits, n_qubits), n_qubits)))

    # dataset.append(get_data(f'qcnn_{8}', QCNN()))
    # dataset.append(get_data('basic_teleportation_3', basic_teleportation.get_cir()))
    # dataset.append(get_data(f'quantum_counting_{8}', quantum_counting.get_cir(4, 4)))

    for n_qubits in range(2, 10):
        dataset.append(get_data(f'deutsch_jozsa_{n_qubits + 1}', deutsch_jozsa.get_cir(n_qubits, get_bitstr(n_qubits))))

    for n_qubits in [1, 2]:
        dataset.append(get_data(f'multiplier_{5 * n_qubits}', multiplier.get_cir(n_qubits)))
        dataset.append(get_data(f'qec_5_x_{5 * n_qubits}', qec_5_x.get_cir(n_qubits)))
        # dataset.append(get_data(f'hhl_{n_qubits}', hhl.get_cir(n_qubits)))

    # dataset.append(get_data(f'phase_kickback_{3}', phase_kickback.get_cir()))

    for n_qubits in [3, 5, 7, 9]:
        dataset.append(get_data(f'qnn_{n_qubits}', qnn.get_cir(n_qubits)))
        dataset.append(get_data(f'qugan_{n_qubits}', qugan.get_cir(n_qubits)))
        dataset.append(get_data(f'swap_{n_qubits}', swap.get_cir(n_qubits)))


    for n_qubits in [2, 4, 6, 8, 10]:
        dataset.append(get_data(f'simon_{n_qubits}', simon.get_cir(get_bitstr(n_qubits // 2))))

    for n_qubits in [6, 9]:
        dataset.append(get_data(f'square_root_{n_qubits}', square_root.get_cir(n_qubits)))
        
    return dataset

# print(len(get_dataset()))