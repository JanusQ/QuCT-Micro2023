import random

from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error, thermal_relaxation_error

from circuit.formatter import my_format_circuit_to_qiskit
from .hardware_info import basis_gates, coupling_map, match_hardware_constraints, max_qubit_num, basis_single_gates, basis_two_gates, two_qubit_fidelity, single_qubit_fidelity, qubit2T1, qubit2T2 #, readout_error
import pickle
from qiskit import QuantumCircuit, execute
from qiskit import IBMQ, Aer

from qiskit.quantum_info.analysis import hellinger_fidelity
from numpy import pi
# https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html

# backend_name = 'ibm_nairobi'
# provider = IBMQ.load_account()
# print(provider.backends())
# backend = provider.get_backend(backend_name)
# noise_model = NoiseModel.from_backend(backend)

# model_path = 'simulator/ibm_nairobi_backend.pkl'
# with open(model_path, 'rb') as file:
#     noise_model, _= pickle.load(file)

# print(noise_model)
# local error：指的是比特自己做门会影响到自己的噪音
# non-local error：指的是比特门会影响到别的比特的噪音

# noise_model = NoiseModel()

# # noise_model.add_readout_error()
# # Add depolarizing error to all single qubit u1, u2, u3 gates on qubit 0 only
# error = depolarizing_error(0.05, 1)
# noise_model.add_quantum_error(error, ['u1', 'u2', 'u3'], [0])
# noise_model.add_re
# # Add depolarizing error on qubit 2 forall single qubit u1, u2, u3 gates on qubit 0
# error = depolarizing_error(0.05, 1)
# noise_model.add_nonlocal_quantum_error(error, ['u1', 'u2', 'u3'], [0], [2])

# print('noise_model', noise_model)
# print('coupling_map', coupling_map)

# TODO: 目前没有考虑 T1/T2 thermal relaxation

noise_model = NoiseModel()
for qubit in range(max_qubit_num):
    _error = 1-single_qubit_fidelity[qubit]
    # error_1q = depolarizing_error(_error, 1)
    # noise_model.add_quantum_error(error_1q, ['rx', 'ry'], [qubit])

    bit_flip = pauli_error([('X', _error), ('I', 1 - _error)])
    phase_flip = pauli_error([('Z', _error), ('I', 1 - _error)])
    _depolarizing_error = depolarizing_error(_error, 1)
    thermal_error = thermal_relaxation_error(qubit2T1[qubit] * 1e3, qubit2T1[qubit] * 1e3, 30)  # ns, ns ns
    
    total_qubit_error = bit_flip.compose(phase_flip).compose(thermal_error).compose(_depolarizing_error)
    # total_qubit_error = thermal_error.compose(_depolarizing_error)
    noise_model.add_quantum_error(total_qubit_error, basis_single_gates, [qubit])
    # # noise_model.add_readout_error(ReadoutError(readout_error[qubit]), [qubit])
    # noise_model.add_quantum_error(errors_thermal, basis_single_gates, [qubit])

for index, (qubit1, qubit2) in enumerate(coupling_map):
    _error = 1-two_qubit_fidelity[index]
    bit_flip = pauli_error([('X', _error), ('I', 1 - _error)])
    phase_flip = pauli_error([('Z', _error), ('I', 1 - _error)])
    _depolarizing_error = depolarizing_error(_error, 2)
    
    thermal_error =  thermal_relaxation_error(qubit2T1[qubit1] * 1e3, qubit2T1[qubit1] * 1e3, 60).expand(thermal_relaxation_error(qubit2T1[qubit2] * 1e3, qubit2T1[qubit2] * 1e3, 60))
    
    total_coupler_error = bit_flip.compose(phase_flip).compose(thermal_error).compose(_depolarizing_error)
    # total_coupler_error = thermal_error.compose(_depolarizing_error)
    noise_model.add_quantum_error(total_coupler_error, basis_two_gates, [qubit1, qubit2])
    
    # error_2q = depolarizing_error(_error, 2)
    # noise_model.add_quantum_error(error_2q, basis_two_gates, [qubit1, qubit2])
    # noise_model.add_quantum_error(error_2q, basis_two_gates, [qubit2, qubit1])

    # bit_flip = pauli_error([('X', _error), ('I', 1 - _error)])
    # phase_flip = pauli_error([('Z', _error), ('I', 1 - _error)])
    # identity = pauli_error([('I', 1)])
    # noise_model.add_quantum_error(identity.tensor(bit_flip.compose(phase_flip)), basis_two_gates, [qubit2, qubit1])
    # noise_model.add_quantum_error(identity.tensor(bit_flip.compose(phase_flip)), basis_two_gates, [qubit1, qubit2])
    
    # noise_model.add_quantum_error(errors_thermal, basis_two_gates, [qubit1, qubit2])

qasm_simulator = Aer.get_backend('qasm_simulator')
def simulate_noise(circuit, n_samples=2000):
    n_qubits = circuit.num_qubits
    initial_layout = list(range(n_qubits))
    # 做噪音实现前就直接必须符合拓扑结构了吧
    # match_hardware_constraints(circuit)
    result = execute(circuit, qasm_simulator,
                    # coupling_map=coupling_map,
                    basis_gates=basis_gates,
                    initial_layout = initial_layout,
                    noise_model=noise_model, shots = n_samples, optimization_level = 0).result()
    return result.get_counts()

# https://qiskit.org/documentation/apidoc/aer_noise.html

# noise_model
# basis_gates: ['cx', 'id', 'reset', 'rz', 'sx', 'x']
# qubits: [0, 1, 2, 3, 4, 5, 6]
# readout_error (_local_readout_errors): 
    # (0,):ReadoutError([[0.9868 0.0132] [0.0384 0.9616]])
    # (1,): ReadoutError([[0.9878 0.0122]
    # ...
#  _custom_noise_passes: 似乎存了一些noise pass比如relaxation_noise_pass, 应该是模拟前加pass用的
# _local_quantum_errors: # 应该指的就是Amplitude damping noise之类的单比特的噪音
    # Local quantum errors are stored as:
    # dict(str: dict(tuple: QuantumError))
    # where the outer keys are the instruction str label and the
    # inner dict keys are the gate qubits
# _nonlocal_quantum_errors: 应该就是crosstalk这些，似乎qiskit现在没有这方面的模拟
    # This applies an error to a specific set of noise qubits after any occurrence of an instruction acting on a specific of gate qubits.
    # Non-local quantum errors are stored as:
    # dict(str: dict(tuple: dict(tuple: QuantumError)))
    # where the outer keys are the instruction str label, the middle dict
    # keys are the gate qubits, and the inner most dict keys are
    # the noise qubits.

# 模拟的时候似乎是通过把Noise Transpiler Passes 插入circuit来进行的
def add_pattern_error_path(circuit, n_qubits, model,erroneous_pattern):  # 单这几个碰到的概念有点少
    '''circuit can be a QuantumCircuit or a circuit_info (a dict created with the instructions's sparse vector)'''
    if isinstance(circuit, QuantumCircuit):
        circuit_info = model.vectorize(circuit)
    else:
        circuit_info = circuit
        circuit = circuit_info['qiskit_circuit']

    index2erroneous_pattern = {
        model.path_index(path) : path
        for path in erroneous_pattern
        if model.has_path(path)
    }

    error_circuit = QuantumCircuit(n_qubits)
    n_erroneous_patterns = 0
    
    instruction2sparse_vector = circuit_info['sparse_vecs']
    instructions = circuit_info['instructions']
    for index, instruction in enumerate(instructions):
        name = instruction['name']
        qubits = instruction['qubits']
        params = instruction['params']
        if name in ('rx', 'ry', 'rz'):
            assert len(params) == 1 and len(qubits) == 1
            error_circuit.__getattribute__(name)(params[0], qubits[0])
        elif name in ('cz', 'cx'):
            assert len(params) == 0 and len(qubits) == 2
            error_circuit.__getattribute__(name)(qubits[0], qubits[1])
        elif name in ('h'):
            error_circuit.__getattribute__(name)(qubits[0])

        sparse_vector = instruction2sparse_vector[index]
        for _index in sparse_vector[0]:
            if _index in index2erroneous_pattern:
                for qubit in instruction['qubits']:
                    error_circuit.rx(pi/20 * random.random(), qubit)  #pi / 20
                    # error_circuit.rx(pi/10, qubit)  #pi / 20
                n_erroneous_patterns += 1
                # break

    return error_circuit, n_erroneous_patterns

def get_error_result(circuit_info,model,erroneous_pattern):
    if 'qiskit_circuit' not in circuit_info:
        circuit_info['qiskit_circuit'] = my_format_circuit_to_qiskit(max_qubit_num, circuit_info['layer2instructions'])

    error_circuit, n_erroneous_patterns = add_pattern_error_path(circuit_info, max_qubit_num, model,erroneous_pattern)
    error_circuit.measure_all()
    noisy_count = simulate_noise(error_circuit, 1000)
    circuit_info['error_result'] = noisy_count
    true_result = {
    '0'*circuit_info['num_qubits']: 2000
    }
    circuit_info['ground_truth_fidelity'] = hellinger_fidelity(circuit_info['error_result'], true_result)
    return circuit_info


def get_random_erroneous_pattern(model):
    error_pattern_num = 20
    paths = list(model.hash_table.keys())
    random.shuffle(paths)
    erroneous_pattern = paths[:error_pattern_num]
    return erroneous_pattern


def get_error_results(dataset,model,erroneous_pattern = None):
    if erroneous_pattern == None:
        erroneous_pattern = get_random_erroneous_pattern(model)

    for circuit_info in dataset:
        get_error_result(circuit_info, model, erroneous_pattern)
