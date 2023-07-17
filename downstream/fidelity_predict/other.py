from simulator.noise_simulator import *
from qiskit import QuantumCircuit
from utils.backend_info import  single_qubit_fidelity, two_qubit_fidelity, readout_error
from upstream.randomwalk_model import RandomwalkModel

def naive_predict(circuit):
    fidelity = 1
    for instruction in circuit.data:
        # print(instruction)
        gate_type = instruction.operation.name
        # print(gate_type)
        operated_qubits = [_.index for _ in instruction.qubits]
        if gate_type == 'barrier':
            continue
        elif gate_type == 'measure':
            _redout_error = readout_error[operated_qubits[0]]
            fidelity *= (_redout_error[0][0] + _redout_error[1][1]) / 2
        elif gate_type in default_basis_single_gates:
            fidelity *= single_qubit_fidelity[operated_qubits[0]]
        elif gate_type in default_basis_two_gates:
            fidelity *= two_qubit_fidelity[tuple(operated_qubits)]
        else:
            raise Exception('unkown gate', instruction)
    return fidelity

def smart_predict(circuit: QuantumCircuit, gate_fidelity_function, parameters, model: RandomwalkModel) -> float:
    ''''''
    fidelity = 1

    instruction2sparse_vector = model.vectorize(circuit)

    for instruction in circuit:
        operated_qubits = [_.index for _ in instruction.qubits]
        gate_type = instruction.operation.name
        if gate_type == 'barrier':
            continue
        elif gate_type == 'measure':
            _redout_error = readout_error[operated_qubits[0]]
            fidelity *= (_redout_error[0][0] + _redout_error[1][1]) / 2
        else:
            sparse_vector = instruction2sparse_vector[id(instruction)]
            fidelity *= gate_fidelity_function(parameters, sparse_vector)

    return fidelity
