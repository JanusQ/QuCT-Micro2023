import random
from collections import defaultdict

from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error, thermal_relaxation_error

from circuit.formatter import layered_circuits_to_qiskit

from qiskit import QuantumCircuit, execute
from qiskit import Aer

from qiskit.quantum_info.analysis import hellinger_fidelity
from numpy import pi

from upstream.randomwalk_model import extract_device, RandomwalkModel
import ray
from circuit.random_circuit import random_1q_layer
from utils.backend import Backend
import time
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

from simulator.noise_free_simulator import simulate_noise_free

class NoiseSimulator():

    def __init__(self, backend):
        self.backend: Backend = backend
        self.noise_model = NoiseModel()
        self.qasm_simulator = Aer.get_backend('qasm_simulator')
        self.n_qubits = backend.n_qubits
        
        single_qubit_time = backend.single_qubit_gate_time
        two_qubit_time = backend.two_qubit_gate_time
        
        # TODO: rz 没有噪音
        for qubit in range(self.n_qubits):
            _error = 1 - backend.single_qubit_fidelity[qubit]
            # error_1q = depolarizing_error(_error, 1)
            # noise_model.add_quantum_error(error_1q, ['rx', 'ry'], [qubit])

            # bit_flip = pauli_error([('X', _error), ('I', 1 - _error)])
            # phase_flip = pauli_error([('Z', _error), ('I', 1 - _error)])
            _depolarizing_error = depolarizing_error(_error, 1)
            thermal_error = thermal_relaxation_error(backend.qubit2T1[qubit] * 1e3, backend.qubit2T1[qubit] * 1e3,
                                                     single_qubit_time)  # ns, ns ns

            # total_qubit_error = bit_flip.compose(
            #     thermal_error).compose(_depolarizing_error)  # .compose(phase_flip)
            total_qubit_error = thermal_error.compose(_depolarizing_error)
            self.noise_model.add_quantum_error(
                total_qubit_error, backend.basis_single_gates, [qubit])
            # noise_model.add_readout_error(ReadoutError(readout_error[qubit]), [qubit])

        for index, (qubit1, qubit2) in enumerate(backend.coupling_map):
            _error = 1 - backend.two_qubit_fidelity[index]
            # bit_flip = pauli_error([('X', _error), ('I', 1 - _error)])
            # phase_flip = pauli_error([('Z', _error), ('I', 1 - _error)])
            
            # bit_flip = bit_flip.expand(bit_flip)
            # phase_flip = phase_flip.expand(phase_flip)
            
            _depolarizing_error = depolarizing_error(_error, 2)

            thermal_error_q1 = thermal_relaxation_error(backend.qubit2T1[qubit1] * 1e3, backend.qubit2T2[qubit1] * 1e3, two_qubit_time)
            thermal_error_q2 = thermal_relaxation_error(backend.qubit2T1[qubit2] * 1e3, backend.qubit2T2[qubit2] * 1e3, two_qubit_time)
            thermal_error = thermal_error_q1.expand(thermal_error_q2)     
                                
            # total_coupler_error = bit_flip.compose(thermal_error).compose(_depolarizing_error)  #.compose(phase_flip)
            
            
            total_coupler_error = thermal_error.compose(_depolarizing_error)
            self.noise_model.add_quantum_error(
                total_coupler_error, backend.basis_two_gates, [qubit1, qubit2])

            # error_2q = depolarizing_error(_error, 2)
            # noise_model.add_quantum_error(error_2q, basis_two_gates, [qubit1, qubit2])
            # noise_model.add_quantum_error(error_2q, basis_two_gates, [qubit2, qubit1])


    def get_error_result(self, sub_dataset, start, model, erroneous_pattern=None):
        fidelities = []
        n_erroneous_patterns = []
        independent_fidelities = []
        
        n_samples = 1000 
        circuit_reps = 20  # 5
        
        for circuit_info in sub_dataset:
            n_qubits = circuit_info['num_qubits']
            true_result = {
                '0' * circuit_info['num_qubits']: 2000
            }
            # true_result = simulate_noise_free(res_qc)
            
            '''TODO: 加多个单层的算fidelity'''
            _fidelities = []
            for _ in range(circuit_reps):
                layer_1q = random_1q_layer(n_qubits, self.backend.basis_single_gates)
                simulate_circuit = QuantumCircuit(n_qubits)
                simulate_circuit = simulate_circuit.compose(layer_1q)
                error_circuit, _ = add_pattern_error_path(circuit_info, circuit_info['num_qubits'], model,
                                                                                    erroneous_pattern)      
                simulate_circuit = simulate_circuit.compose(error_circuit)
                simulate_circuit = simulate_circuit.compose(layer_1q.inverse())
                simulate_circuit.measure_all()
                noisy_count = self.simulate_noise(simulate_circuit, n_samples)
                _fidelities.append(hellinger_fidelity(noisy_count, true_result))
                
            error_circuit, _n_erroneous_patterns = add_pattern_error_path(circuit_info, circuit_info['num_qubits'], model,
                                                                                    erroneous_pattern)                
            error_circuit.measure_all()
            noisy_count = self.simulate_noise(error_circuit, n_samples)
            # circuit_info['error_result'] = noisy_count
            _fidelities.append(hellinger_fidelity(noisy_count, true_result))
            
            fidelities.append(sum(_fidelities)/len(_fidelities))
            
            independent_error_circuit, _ = add_pattern_error_path(circuit_info, circuit_info['num_qubits'], model, defaultdict(list))
            _fidelities = []
            for _ in range(3):
                layer_1q = random_1q_layer(n_qubits, self.backend.basis_single_gates)
                simulate_circuit = QuantumCircuit(n_qubits)
                simulate_circuit = simulate_circuit.compose(layer_1q)
                simulate_circuit = simulate_circuit.compose(independent_error_circuit)
                simulate_circuit = simulate_circuit.compose(layer_1q.inverse())
                simulate_circuit.measure_all()
                independent_noisy_count = self.simulate_noise(simulate_circuit, n_samples)
                _fidelities.append(hellinger_fidelity(independent_noisy_count, true_result))
                
            independent_error_circuit.measure_all()
            independent_noisy_count = self.simulate_noise(independent_error_circuit, n_samples)
            _fidelities.append(hellinger_fidelity(independent_noisy_count, true_result))
            independent_fidelities.append(sum(_fidelities)/len(_fidelities))
            
            # independent_fidelities.append(hellinger_fidelity(independent_noisy_count, true_result))
            
            n_erroneous_patterns.append(_n_erroneous_patterns)

        print(start+len(sub_dataset), 'finished')
        return fidelities, n_erroneous_patterns, independent_fidelities

    def match_hardware_constraints(self, circuit: QuantumCircuit):
        
        backend : Backend = self.backend
        
        if isinstance(circuit, list):
            for _ in circuit:
                self.match_hardware_constraints(_)
        else:
            for gate in circuit.get_instructions('cx') + circuit.get_instructions('cz'):
                qubits = [qubit.index for qubit in gate.qubits]
                qubits.sort()
                
                assert qubits in backend.coupling_map
        
        # return
    
    def simulate_noise(self, circuit, n_samples=2000, get_count = True):
        if not isinstance(circuit, QuantumCircuit):  # is list
            n_qubits = circuit[0].num_qubits
        else:  
            n_qubits = circuit.num_qubits
        
        # circuit.measure()
        initial_layout = list(range(n_qubits))
        # 做噪音实现前就直接必须符合拓扑结构了吧
        # self.match_hardware_constraints(circuit)
        result = execute(circuit, self.qasm_simulator,
                         # coupling_map=coupling_map,
                         basis_gates=self.backend.basis_gates,
                         initial_layout=initial_layout,
                         noise_model=self.noise_model, shots=n_samples, optimization_level=0)
        
        if get_count:
            return result.result().get_counts()
        else:
            return result



    def get_error_results(self, dataset, model, erroneous_pattern=None, multi_process=False):
        # if erroneous_pattern is None:
        #     erroneous_pattern = get_random_erroneous_pattern(model)
            
        futures = []
        fidelities = []
        independent_fidelities = []
        n_erroneous_patterns = []
        step = 100
        
        for start in range(0, len(dataset), step):
            sub_dataset = dataset[start:start + step]
            if multi_process:
                _dataset = model.dataset
                model.dataset = None
                call_time = time.time()
                futures.append(get_error_result_remote.remote(
                    self, sub_dataset, start, model, erroneous_pattern, call_time))
                model.dataset = _dataset
            else:
                _fidelities, _n_erroneous_patterns, _independent_fidelities = self.get_error_result(
                    sub_dataset, start, model, erroneous_pattern)
                fidelities += _fidelities
                n_erroneous_patterns += _n_erroneous_patterns
                independent_fidelities += _independent_fidelities

        for future in futures:
            _fidelities, _n_erroneous_patterns, _independent_fidelities = ray.get(future)
            fidelities += _fidelities
            n_erroneous_patterns += _n_erroneous_patterns
            independent_fidelities += _independent_fidelities

        for idx, cir in enumerate(dataset):
            cir['ground_truth_fidelity'] = fidelities[idx]
            cir['n_erroneous_patterns'] = n_erroneous_patterns[idx]
            cir['independent_fidelity'] = independent_fidelities[idx]
            
        return erroneous_pattern

@ray.remote
def get_error_result_remote(noiseSimulator, sub_dataset, start, model, erroneous_pattern=None, call_time = None):
    
    # if call_time is not None:
    #     print('call latency = ', time.time() - call_time, 's')
    
    return noiseSimulator.get_error_result(sub_dataset, start, model, erroneous_pattern)

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


def add_pattern_error_path(circuit, n_qubits, model, device2erroneous_pattern):  # 单这几个碰到的概念有点少
    '''circuit can be a QuantumCircuit or a circuit_info (a dict created with the instructions's sparse vector)'''
    if isinstance(circuit, QuantumCircuit):
        # print(circuit)
        circuit_info = model.vectorize(circuit)
    else:
        circuit_info = circuit
        # circuit = circuit_info['qiskit_circuit']

    device2erroneous_pattern_index = defaultdict(list)
    for device, erroneous_pattern in device2erroneous_pattern.items():
        erroneous_pattern_index = {
            model.path_index(device, path): path
            for path in erroneous_pattern
            if model.has_path(device, path)
        }
        device2erroneous_pattern_index[device] = erroneous_pattern_index

    error_circuit = QuantumCircuit(n_qubits)
    n_erroneous_patterns = 0

    gates = circuit_info['gates']
    for index, gate in enumerate(gates):
        name = gate['name']
        qubits = gate['qubits']
        params = gate['params']
        device = extract_device(gate)
        if 'map' in circuit_info:
            if isinstance(device, tuple):
                device = (circuit_info['map'][device[0]],
                          circuit_info['map'][device[1]])
            else:
                device = circuit_info['map'][device]
        erroneous_pattern_index = device2erroneous_pattern_index[device]
        if name in ('rx', 'ry', 'rz'):
            assert len(params) == 1 and len(qubits) == 1
            error_circuit.__getattribute__(name)(params[0], qubits[0])
        elif name in ('cz', 'cx'):
            assert len(params) == 0 and len(qubits) == 2
            error_circuit.__getattribute__(name)(qubits[0], qubits[1])
        elif name in ('h'):
            error_circuit.__getattribute__(name)(qubits[0])
        else:
            raise Exception(gate, 'known')

        path_index = circuit_info['path_indexs'][index]
        for _index in path_index:
            if _index in erroneous_pattern_index:
                for qubit in gate['qubits']:
                    # error_circuit.rx(
                    #     pi / 50, qubit)  # 之前跑50-300比特的时候用的10, 感觉噪声加的太大了，所以换成20
                    error_circuit.rx(
                        pi / 20 * random.random() - pi / 10, qubit)  # 之前跑50-300比特的时候用的10, 感觉噪声加的太大了，所以换成20
                    # error_circuit.rx(
                    #     pi / 100 * random.random() - pi / 50, qubit)  # 之前跑50-300比特的时候用的10, 感觉噪声加的太大了，所以换成20
                    # error_circuit.rx(
                    #     pi / 50 * random.random() - pi / 25, qubit)  # 之前跑50-300比特的时候用的10, 感觉噪声加的太大了，所以换成20
                    # pass
                    # pi / 20 + pi / 20
                    # error_circuit.rx(pi/10, qubit)  #pi / 20
                n_erroneous_patterns += 1
                # break

    return error_circuit, n_erroneous_patterns


def get_random_erroneous_pattern(model: RandomwalkModel, error_pattern_num_per_device=6):
    model.error_pattern_num_per_device = error_pattern_num_per_device
    device2erroneous_pattern = defaultdict(list)
    
    # 不均匀的
    all_device_paths = []
    for device, path_table in model.device2path_table.items():
        device_paths = [(device, path) for path in path_table.keys()]
        all_device_paths += device_paths
        
    random.shuffle(all_device_paths)
    erroneous_patterns = all_device_paths[:error_pattern_num_per_device * len(model.device2path_table)]
    
    
    for device in model.device2path_table:
        device2erroneous_pattern[device] = []
    
    for device, erroneous_pattern in erroneous_patterns:
        if '-' not in erroneous_pattern or 'rz' in erroneous_pattern:
            continue
        device2erroneous_pattern[device].append(erroneous_pattern)
    
    # for device, path_table in model.device2path_table.items():
    #     paths = list(path_table.keys())
    #     random.shuffle(paths)
    #     erroneous_patterns = paths[:error_pattern_num_per_device]
    #     erroneous_patterns = [erroneous_pattern for erroneous_pattern in erroneous_patterns if '-' in erroneous_pattern]
    #     device2erroneous_pattern[device] = erroneous_patterns
        
    print(device2erroneous_pattern)
        
    return device2erroneous_pattern


def get_uneven_erroneous_pattern(model, total_error_pattern_num=50):
    model.total_error_pattern_num = total_error_pattern_num
    device2erroneous_pattern = defaultdict(list)

    all_path_num = 0
    for device, path_table in model.device2path_table.items():
        all_path_num += len(path_table.keys())

    p = total_error_pattern_num / all_path_num

    cnt = 0
    for device, path_table in model.device2path_table.items():
        paths = list(path_table.keys())
        for path in paths:
            if random.random() <= p:
                device2erroneous_pattern[device] += path
                cnt += 1

    print(total_error_pattern_num, cnt)
    return device2erroneous_pattern
