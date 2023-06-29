from cmath import pi
from collections import defaultdict
from qiskit import QuantumCircuit
import random
# from utils.backend_info import default_basis_two_gates, default_basis_single_gates
# from qiskit.converters import dag_to_circuit, circuit_to_dag
import math

'''
randomly generate various circuit for noise analysis
'''

        
def random_pi(): 
    rand = round(random.random(), 1)
    if rand == 0: return 0.1 * pi
    return rand * 2 *  pi

def random_gate(circuit: QuantumCircuit, qubits, two_qubit_prob, coupling_map, basis_single_gates, basis_two_gates, pre_couple, pre_single):
    if random.random() < two_qubit_prob:
        gate_type = basis_two_gates[0]
        assert len(basis_two_gates) == 1
    else:
        gate_type = random.choice(basis_single_gates)
    
    if len(coupling_map) != 0:
        operated_qubits = list(random.choice(coupling_map))
        control_qubit = operated_qubits[0]
        target_qubit = operated_qubits[1]
        
    if len(coupling_map) == 0 and gate_type in ('cx', 'cz', 'unitary'):
        gate_type = random.choice(basis_single_gates)
        print('WARNING: no coupling map')
    
    if gate_type == 'cz':
        # 没有控制和非控制的区别
        if pre_couple[control_qubit] == target_qubit and pre_couple[target_qubit] == control_qubit:
            random_gate(circuit, qubits, two_qubit_prob, coupling_map, basis_single_gates, basis_two_gates, pre_couple, pre_single)
        else:
            circuit.cz(control_qubit, target_qubit)
            pre_couple[control_qubit] = target_qubit 
            pre_couple[target_qubit] = control_qubit
            pre_single[control_qubit] = ''
            pre_single[target_qubit] = ''
            
    elif gate_type == 'cx':
        random.shuffle(operated_qubits)
        control_qubit = operated_qubits[0]
        target_qubit = operated_qubits[1]
        if pre_couple[control_qubit] == target_qubit and pre_couple[target_qubit] != control_qubit:
            random_gate(circuit, qubits, two_qubit_prob, coupling_map, basis_single_gates, basis_two_gates, pre_couple, pre_single)
        else:
            circuit.cx(control_qubit, target_qubit)
            pre_couple[control_qubit] = target_qubit
            pre_couple[target_qubit] = -1
            pre_single[control_qubit] = ''
            pre_single[target_qubit] = '' 
        
    elif gate_type in ('h',):
        selected_qubit = random.choice(qubits)
        circuit.h(selected_qubit)
        pre_single[selected_qubit] = ''
        if pre_couple[pre_couple[selected_qubit]] == selected_qubit:
            pre_couple[pre_couple[selected_qubit]] = -1 
        pre_couple[selected_qubit] = -1
    elif gate_type in ('rx', 'rz', 'ry'):
        selected_qubit = random.choice(qubits)
        if gate_type == pre_single[selected_qubit]:
            random_gate(circuit, qubits, two_qubit_prob, coupling_map, basis_single_gates, basis_two_gates, pre_couple, pre_single)
        else:
            getattr(circuit, gate_type)(random_pi(), selected_qubit)
            # getattr(circuit, gate_type)(pi, selected_qubit)
            pre_single[selected_qubit] = gate_type
            if pre_couple[pre_couple[selected_qubit]] == selected_qubit:
                pre_couple[pre_couple[selected_qubit]] = -1 
            pre_couple[selected_qubit] = -1
            
    elif gate_type in ('u',):
        selected_qubit = random.choice(qubits)
        if gate_type == pre_single[selected_qubit]:
            random_gate(circuit, qubits, two_qubit_prob, coupling_map, basis_single_gates, basis_two_gates, pre_couple, pre_single)
        else:
            getattr(circuit, gate_type)(random_pi(), random_pi(), random_pi(), selected_qubit)
            pre_single[selected_qubit] = gate_type
            if pre_couple[pre_couple[selected_qubit]] == selected_qubit:
                pre_couple[pre_couple[selected_qubit]] = -1 
            pre_couple[selected_qubit] = -1
    else:
        raise Exception('Unknown gate type', gate_type)
    
    return

def random_1q_layer(n_qubits, basis_single_gates):
    qubits = list(range(n_qubits))
    circuit = QuantumCircuit(n_qubits)
    
    for qubit in qubits:
        gate_type = random.choice(basis_single_gates)
        if gate_type in ('h',):
            circuit.h(qubits)
        elif gate_type in ('rx', 'rz', 'ry'):
            getattr(circuit, gate_type)(random_pi(), qubit)
        elif gate_type in ('u',):
            getattr(circuit, gate_type)(random_pi(), random_pi(), random_pi(), qubit)
        else:
            raise Exception('Unknown gate type', gate_type)
        
    return circuit

# 没从从coupling map里面挑两比特门
def random_circuit(n_qubits, n_gates, two_qubit_prob = 0.5, reverse = True, coupling_map = None, basis_single_gates = None, basis_two_gates = None,):
    if reverse:
        n_gates = n_gates//2
    circuit = QuantumCircuit(n_qubits)
    qubits = list(range(n_qubits))

    circuit = circuit.compose(random_1q_layer(n_qubits, basis_single_gates))
    
    n_gates -= len(qubits)

    pre_single = defaultdict(str)
    pre_couple = defaultdict(lambda:-1)
    cnt = 0
    while cnt < n_gates:
        random_gate(circuit, qubits, two_qubit_prob, coupling_map, basis_single_gates, basis_two_gates, pre_couple, pre_single)
        cnt += 1

    if reverse:
        circuit = circuit.compose(circuit.inverse())

    # print(circuit)
    return circuit


def _random_block(n_qubits, n_gates, two_qubit_prob,  coupling_map, basis_single_gates, basis_two_gates,):
    circuit = QuantumCircuit(n_qubits)
    qubits = list(range(n_qubits))
    
    n_gates = n_gates//2

    pre_single = defaultdict(str)
    pre_couple = defaultdict(lambda:-1)
    cnt = 0
    while cnt < n_gates:
        random_gate(circuit, qubits, two_qubit_prob, coupling_map, basis_single_gates, basis_two_gates, pre_couple, pre_single)
        cnt += 1
    
    circuit = circuit.compose(circuit.inverse())
    
    return circuit


def random_circuit_cycle(n_qubits, n_gates, two_qubit_prob = 0.5, reverse = True, coupling_map = None, basis_single_gates = None, basis_two_gates = None,):
    assert reverse  == True
    
    # n_gates -= n_qubits * 2  # 两层单比特
    
    block_size  = 20
    block = _random_block(n_qubits, block_size, two_qubit_prob,  coupling_map, basis_single_gates, basis_two_gates,)    
    
    block_num = n_gates // block_size
        
    circuit = QuantumCircuit(n_qubits)
    # qubits = list(range(n_qubits))
    
    layer_1q = random_1q_layer(n_qubits, basis_single_gates)
    circuit = circuit.compose(layer_1q)
    circuit.barrier()
    
    for i_block in range(block_num):    
        circuit = circuit.compose(block)
        circuit.barrier()
    
    circuit = circuit.compose(layer_1q.inverse())
    
    # print(circuit)     
    return circuit
