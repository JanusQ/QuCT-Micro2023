import os

from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
import matplotlib.pyplot as plt
from collections import OrderedDict
from circuit.quct.analysis.cricuit_operation import dynamic_decoupling, dynamic_decoupling_divide, get_layered_instructions, layer2circuit
from circuit.quct.dataset.random_circuit import random_circuit, random_circuit_various_input
from qiskit import QuantumCircuit, QuantumRegister


# import pickle

# 用data里面的顺序定义qc的门id。可能会优雅一些

def instruction2str(instruction):
    if isinstance(instruction, CircuitInstruction):
        qubits = [qubit.index for qubit in instruction.qubits]
        op_name = instruction.operation.name
    else:
        qubits = instruction['qubits']
        op_name = instruction['name']
    return f'{op_name}-{"-".join([str(_) for _ in qubits])}'



from circuit.quct.analysis.cricuit_operation import my_circuit_to_dag
# 之后还可以单独写一个获得每个比特前面的和后面的门的代码

# def draw_network_circuit(network_circuit):
#     # TODO: _node_id 不是唯一的吗?
#     edge_labels = nx.get_edge_attributes(network_circuit, 'type') # 获取边的name属性，
    
#     pos = nx.spring_layout(network_circuit, iterations=100)
#     # nx.draw_networkx_labels(new_network_circuit, pos, labels=node_labels,font_size=10)  # 将desc属性，显示在节点上
#     # node_labels = nx.get_node_attributes(new_network_circuit, 'name')  # 获取节点的desc属性

#     nx.draw_networkx_edge_labels(network_circuit, pos, edge_labels=edge_labels, font_size=10) # 将name属性，显示在边上

#     nx.draw(network_circuit, pos, with_labels=True, )
#     plt.show()

def qiskit_to_my_format_instruction(instruction):
    return {
        'name': instruction.operation.name,
        'qubits': [qubit.index for qubit in instruction.qubits],
        'params': instruction.operation.params,
    }


def qiskit_to_my_format_circuit(layer2qiskit_instructions):
    instructions = []
    instruction2layer = []
    layer2instructions = []
    for layer, layer_instructions in enumerate(layer2qiskit_instructions):
        layer_instructions = [qiskit_to_my_format_instruction(instruction) for instruction in layer_instructions]
        instructions += layer_instructions
        layer2instructions.append(layer_instructions)

        for layer_instructions in layer_instructions:
            instruction2layer.append(layer)

    for index, instruction in enumerate(instructions):
        instruction['id'] = index

    return layer2instructions, instruction2layer, instructions

def my_format_circuit_to_qiskit(qubit_num, layer2instructions):
    circuit = QuantumCircuit(qubit_num)

    for layer, layer_instructions in enumerate(layer2instructions):
        for instruction in layer_instructions:
            name = instruction['name']
            qubits = instruction['qubits']
            params = instruction['params']
        if name in ('rx', 'ry', 'rz'):
            assert len(params) == 1 and len(qubits) == 1
            circuit.__getattribute__(name)(params[0], qubits[0])
        elif name in ('cz', 'cx'):
            assert len(params) == 0 and len(qubits) == 2
            circuit.__getattribute__(name)(qubits[0], qubits[1])
        elif name in ('h'):
            circuit.__getattribute__(name)(qubits[0])
        else:
            circuit.__getattribute__(name)(*(params + qubits))

        circuit.barrier()
    
    return circuit

def parse_circuit(circuit, devide=False, require_decoupling=False):
    circuit_info = {}
    layer2instructions, instruction2layer, instructions, dagcircuit, nodes = get_layered_instructions(circuit) # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的
    # layer2instructions = dynamic_coupling(circuit.num_qubits, layer2instructions, 2)
    # print(circuit)
    if(devide):
        layer2instructions = divide_layer(layer2instructions)
        if require_decoupling:
            layer2instructions = dynamic_decoupling_divide(layer2instructions, 30, 60, 300)
            circuit = layer2circuit(layer2instructions, circuit.num_qubits)
    else:
        if require_decoupling:
            layer2instructions = dynamic_decoupling(layer2instructions, 60, 300)
            circuit = layer2circuit(layer2instructions, circuit.num_qubits)
    
    # print(circuit)
    # assert_devide(layer2instructions)
    
    layer2instructions, instruction2layer, instructions, dagcircuit, nodes = get_layered_instructions(circuit)
    # assert_devide(layer2instructions)

    layer2instructions, instruction2layer, instructions = qiskit_to_my_format_circuit(layer2instructions)  # 转成一个更不占内存的格式

    circuit_info['layer2instructions'] = layer2instructions
    circuit_info['instruction2layer'] = instruction2layer
    circuit_info['instructions'] = instructions
    circuit_info['num_qubits'] = circuit.num_qubits
    # circuit_info['dagcircuit'] = dagcircuit
    # circuit_info['nodes'] = nodes
    circuit_info['qiskit_circuit'] = circuit
    return circuit_info


def assert_devide(layer2instructions):
    for layer in layer2instructions:
        s = True
        d = True
        for instruction in layer:
            if len(instruction.qubits) == 1:
                s = False
            else:
                d = False
                
        assert s == True or d == True
        
        
def divide_layer(layer2instructions):
    copy_list = []
    for layer in layer2instructions:
        list_s = []
        list_t = []
        for instruction in layer:
            if len(instruction.qubits) == 1:
                list_s.append(instruction)
            else:
                list_t.append(instruction)
        if len(list_s) != 0 and len(list_t) != 0:
            copy_list.append(list_s)
            copy_list.append(list_t)
        else:
            copy_list.append(layer)
    return copy_list


def load_algorithms():
    from ._algorithm import dataset as _dataset

    dataset = [
        # {
            # id: name_#qubit_otherparms
            # qiskit_circuit: qiskit.QuantumCircuit
            # network_circuit: circuit from the parse_circuit function
            # gate2sparse_vector,
            # 
        # }
    ]   # load all the circuits
    dataset += _dataset

    for elm in dataset:
        circuit = elm['qiskit_circuit']
        circuit_info = parse_circuit(circuit) # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的
        elm.update(circuit_info)
    return dataset

def load_randomcircuits(n_qubits, n_gates = 40, two_qubit_prob = 0.5, n_circuits = 2000, reverse = True, devide = True, require_decoupling = True):
    dataset = [
        ({
            'id': f'rc_{n_qubits}_{n_gates}_{two_qubit_prob}_{_}',
            'qiskit_circuit': random_circuit(n_qubits, n_gates, two_qubit_prob, reverse = reverse) 
        })
        for _ in range(n_circuits)
    ]

    new_dataset = []
    for elm in dataset:
        # circuit = elm['qiskit_circuit']
        # print(circuit)
        circuit_info = parse_circuit(elm['qiskit_circuit'], devide, require_decoupling) # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的
        circuit_info['id'] = elm['id']
        circuit_info['qiskit_circuit'] = elm['qiskit_circuit']
        # elm.update(circuit_info)
        new_dataset.append(circuit_info)

    return new_dataset

def load_randomcircuits_various_input(n_qubits, n_gates ,center, n_circuits, devide = True, require_decoupling = True):
    dataset = []
    for i in range(center):
        dataset+= random_circuit_various_input(n_qubits, n_gates,n_circuits=n_circuits, two_qubit_prob = 0.5)
    
    new_dataset = []
    for elm in dataset:
        # circuit = elm['qiskit_circuit']
        # print(circuit)
        
        circuit_info = parse_circuit(elm, devide,  require_decoupling) # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的
        circuit_info['qiskit_circuit'] = elm
        circuit_info['qiskit_circuit_devide'] = my_format_circuit_to_qiskit(n_qubits, circuit_info['layer2instructions'])
        # elm.update(circuit_info)
        new_dataset.append(circuit_info)

    return new_dataset

# def save_dataset(dataset, path):
#     path = os.path.join('dataset/temp',path)
#     file = open(path,'wb')
#     return

if __name__ == '__main__':
    dataset = load_algorithms()
    print(len(dataset))
    print(dataset)