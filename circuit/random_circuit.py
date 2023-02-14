from cmath import pi
from qiskit import QuantumCircuit
import random
from simulator.hardware_info import basis_two_gates,basis_single_gates
from qiskit.converters import dag_to_circuit, circuit_to_dag
import math

'''
randomly generate various circuit for noise analysis
'''


def random_pi(): 
    rand = round(random.random(), 1)
    if rand == 0: return 0.1 * pi
    return rand * pi

# 没从从coupling map里面挑两比特门
def random_circuit(n_qubits, n_gates, two_qubit_prob = 0.5, reverse = True):
    if reverse:
        n_gates = n_gates//2
    circuit = QuantumCircuit(n_qubits)
    qubits = list(range(n_qubits))

    for qubit in qubits:
        gate_type = random.choice(basis_single_gates)
        if gate_type == 'h':
            circuit.h(random.choice(qubits))
        elif gate_type in ('rx', 'rz', 'ry'):
            getattr(circuit, gate_type)(random_pi(), qubit)
        
    coupling_map = [[q, q+1] for q in range(n_qubits-1)]
    
    for _ in range(n_gates):
        if random.random() < two_qubit_prob:
            gate_type = basis_two_gates[0]
            assert len(basis_two_gates) == 1
        else:
            gate_type = random.choice(basis_single_gates)
        
        operated_qubits = list(random.choice(coupling_map))
        random.shuffle(operated_qubits)
        control_qubit = operated_qubits[0]
        target_qubit = operated_qubits[1]
        if gate_type == 'cz':
            circuit.cz(control_qubit, target_qubit)
        if gate_type == 'cx':
            circuit.cx(control_qubit, target_qubit)
        elif gate_type == 'h':
            circuit.h(random.choice(qubits))
        elif gate_type in ('rx', 'rz', 'ry'):
            getattr(circuit, gate_type)(random_pi(), random.choice(qubits))
            # circuit_to_dag(circuit)

    if reverse:
        circuit = circuit.compose(circuit.inverse())

    return circuit


def random_circuit_various_input(n_qubits, n_gates, n_circuits, two_qubit_prob = 0.5):
    n_gates = n_gates//2

    center_circuit = QuantumCircuit(n_qubits)
    qubits = list(range(n_qubits))
    coupling_map = [[q, q+1] for q in range(n_qubits-1)]
    for _ in range(n_gates):
        if random.random() < two_qubit_prob:
            gate_type = basis_two_gates[0]
            assert len(basis_two_gates) == 1
        else:
            gate_type = random.choice(basis_single_gates)
        
        operated_qubits = list(random.choice(coupling_map))
        random.shuffle(operated_qubits)
        control_qubit = operated_qubits[0]
        target_qubit = operated_qubits[1]
        if gate_type == 'cz':
            center_circuit.cz(control_qubit, target_qubit)
        if gate_type == 'cx':
            center_circuit.cx(control_qubit, target_qubit)
        elif gate_type == 'h':
            center_circuit.h(random.choice(qubits))
        elif gate_type in ('rx', 'rz', 'ry'):
            getattr(center_circuit, gate_type)(random_pi(), random.choice(qubits))

    circuits = []
    for _ in range(n_circuits):
        circuit = QuantumCircuit(n_qubits)

        for qubit in qubits:
            gate_type = random.choice(basis_single_gates)
            if gate_type == 'h':
                circuit.h(random.choice(qubits))
            elif gate_type in ('rx', 'rz', 'ry'):
                getattr(circuit, gate_type)(random_pi(), qubit)
        
        circuit = circuit.compose(center_circuit)
        circuit = circuit.compose(circuit.inverse())
        circuits.append(circuit)

    return circuits

def one_layer_random_circuit(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    for qubit in range(n_qubits):
        gate_type = random.choice(basis_single_gates)
        getattr(circuit, gate_type)(pi * random.random(), qubit)
    return circuit

if __name__ == '__main__':
    # for i in range(10):
    #     print(random_circuit(5, 30))
    circuits = random_circuit_various_input(5, 60, 20)
    for circuit in circuits:
        print(circuit)
        print('\n')
