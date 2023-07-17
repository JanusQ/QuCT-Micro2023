import math

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.dagcircuit import DAGOpNode

from circuit.formatter import qiskit_to_my_format_circuit, layered_instructions_to_circuit,  get_layered_instructions
import numpy as np


def qiskit_to_layered_circuits(circuit, divide=False, decoupling=False):
    circuit_info = {}
    
    assert divide == False and decoupling == False, 'get_layered_instructions 问题太多了，主要是创建一个新的时候quanutm register对不上'
    # layer2instructions, instruction2layer, instructions, dagcircuit, nodes = get_layered_instructions(
    #     circuit)  # instructions的index之后会作为instruction的id, nodes和instructions的顺序是一致的

    # if (divide):
    #     layer2instructions = divide_layer(layer2instructions)
    #     if decoupling:
    #         layer2instructions = dynamic_decoupling_divide(
    #             layer2instructions, 30, 60, 300)
    #         circuit = layered_instructions_to_circuit(layer2instructions, circuit.num_qubits)
    # else:
    #     if decoupling:
    #         layer2instructions = dynamic_decoupling(
    #             layer2instructions, 60, 300)
    #         circuit = layered_instructions_to_circuit(layer2instructions, circuit.num_qubits)

    # circuit = layered_instructions_to_circuit(layer2instructions, circuit.num_qubits)
    layer2instructions, instruction2layer, instructions, dagcircuit, nodes = get_layered_instructions(
        circuit)

    layer2gates, gate2layer, gates = qiskit_to_my_format_circuit(
        layer2instructions)  # 转成一个更不占内存的格式

    circuit_info['layer2gates'] = layer2gates
    circuit_info['gate2layer'] = gate2layer
    circuit_info['gates'] = gates
    circuit_info['num_qubits'] = circuit.num_qubits
    # circuit_info['dagcircuit'] = dagcircuit
    # circuit_info['nodes'] = nodes
    circuit_info['qiskit_circuit'] = circuit
    return circuit_info


def assert_divide(layer2instructions):
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


def dynamic_decoupling(layer2instructions, t_t, t_dd):
    n_qubits = layer2instructions[0][0].qubits[0].register.size
    num_layer = len(layer2instructions)
    layer_type = get_layer_type(layer2instructions, n_qubits)

    for qubit in range(n_qubits):
        start = 0
        while start < num_layer:
            start, end = _get_ideal_section_in_one_qubit_dynamic_coupling_divide(
                start, layer_type[qubit])
            layer2instructions = _dynamic_decoupling_in_one_ideal_section(
                start, end, qubit, t_t, t_dd, layer_type[qubit], layer2instructions
            )
            start = end + 1
    return layer2instructions


def get_layer_type(layer2instructions, n_qubits):
    num_layer = len(layer2instructions)
    layer_type = [[0 for _ in range(num_layer)] for _ in range(n_qubits)]
    for index, layer in enumerate(layer2instructions):
        for ins in layer:
            for qubit in ins.qubits:
                layer_type[qubit.index][index] = 1
    return layer_type


def _get_ideal_section_in_one_qubit_dynamic_coupling(start, one_layer_type):
    num_layer = len(one_layer_type)
    end = start
    for i in range(start, num_layer):
        if one_layer_type[i] == 0:
            start = i
            break
    fin_loop = 0
    for i in range(start, num_layer):
        if one_layer_type[i] == 1:
            end = i
            break
        fin_loop = i
    if fin_loop == num_layer - 1:
        end = num_layer
    return start, end


def _dynamic_decoupling_in_one_ideal_section(start, end, qubit, t_t, t_dd, one_layer_type, layer2instructions):
    n_qubits = layer2instructions[0][0].qubits[0].register.size
    if end - start < 2:
        return layer2instructions
    while (end - start) > 1:
        start_gate_id = start
        jump = t_dd // 2 // t_t
        if (t_dd / 2) % t_t > (t_dd / 4):
            jump += 1
        end_gate_id = start_gate_id + jump
        if end_gate_id >= end:
            end_gate_id -= 1
        layer2instructions[start_gate_id].append(
            get_x_instruction(n_qubits, qubit)
        )
        layer2instructions[end_gate_id].append(
            get_x_instruction(n_qubits, qubit)
        )
        start = end_gate_id + 1
    return layer2instructions


def _get_ideal_section_in_one_qubit_dynamic_coupling_divide(start, one_layer2qubits):
    num_layer = len(one_layer2qubits)
    end = start
    for i in range(start, num_layer):
        if one_layer2qubits[i] == 0 or one_layer2qubits[i] == 3:
            start = i
            break
    fin_loop = 0
    for i in range(start, num_layer):
        if one_layer2qubits[i] == 2 or one_layer2qubits[i] == 1:
            end = i
            break
        fin_loop = i
    if fin_loop == num_layer - 1:
        end = num_layer
    return start, end


def _dynamic_decoupling_in_one_ideal_section_divide(start, end, qubit, t_s, t_t, t_dd, one_layer2qubits, layer2instructions):
    n_qubits = layer2instructions[0][0].qubits[0].register.size
    list0 = []
    for i in range(start, end):
        if one_layer2qubits[i] == 0:
            list0.append(i)
    if len(list0) < 2:
        return layer2instructions

    while len(list0) > 1:
        start_gate_id = list0.pop(0)
        listd = []
        for i in list0:
            t = 0
            for j in range(start_gate_id, i):
                if one_layer2qubits[j] == 0:
                    t += t_s
                else:
                    t += t_t
            listd.append({
                'id': i,
                't': abs(t - t_dd / 2)
            })
        min_t = listd[0]['t']
        end_gate_id = listd[0]['id']
        for d in listd:
            if min_t > d['t']:
                min_t = d['t']
                end_gate_id = d['id']
        layer2instructions[start_gate_id].append(
            get_x_instruction(n_qubits, qubit)
        )
        layer2instructions[end_gate_id].append(
            get_x_instruction(n_qubits, qubit)
        )
        list0 = [i for i in list0 if i > end_gate_id]
    return layer2instructions


def dynamic_decoupling_divide(layer2instructions, t_s, t_t, t_dd):
    n_qubits = layer2instructions[0][0].qubits[0].register.size
    num_layer = len(layer2instructions)
    layer2qubits = get_layer2qubits_divide(layer2instructions, n_qubits)
    for qubit in range(n_qubits):
        start = 0
        while start < num_layer:
            # 分段找完整的03段
            start, end = _get_ideal_section_in_one_qubit_dynamic_coupling_divide(
                start, layer2qubits[qubit])
            # 在这一段进行门保护
            layer2instructions = _dynamic_decoupling_in_one_ideal_section_divide(
                start, end, qubit, t_s, t_t, t_dd, layer2qubits[qubit], layer2instructions
            )
            start = end + 1
    return layer2instructions


def get_layer2qubits_divide(layer2instructions, n_qubits):
    """
    返回一个列表，列表中的每个元素是 每个比特对应的每层门类型表，0为单比特层无门，1为单比特门，2为双比特门，3为双比特层无门
    """
    layer2qubits = []
    num_layer = len(layer2instructions)
    layer_type = get_layer_type_divide(layer2instructions)
    for qubit in range(n_qubits):
        layer2qubits.append([0 for _ in range(num_layer)])
    for index, layer in enumerate(layer2instructions):
        if layer_type[index] == 1:
            for ins in layer:
                layer2qubits[ins.qubits[0].index][index] = 1
        else:
            for ins in layer:
                for qubit in ins.qubits:
                    layer2qubits[qubit.index][index] = 2
                for qubit in range(n_qubits):
                    if layer2qubits[qubit][index] == 0:
                        layer2qubits[qubit][index] = 3
    return layer2qubits

# def get_layer_type_divide(layer2instructions):
#     """
#     返回layer_circuits的每层的类型，是单比特门层则为1；否则则为2
#     """
#     return [len(layer[0].qubits) for layer in layer2instructions]


def get_x_instruction(n_qubits, index):
    """
    生成一个x门的instruction，输入电路的比特数和，x门插在哪个比特上
    """
    x_circuit = QuantumCircuit(1)
    x_circuit.rx(math.pi, 0)
    x_operation = x_circuit.data[0].operation
    register = QuantumRegister(size=n_qubits, name='q')
    return CircuitInstruction(x_operation,
                              [Qubit(register=register, index=index)]
                              )


def get_layer_type_divide(layer2instructions):
    """
    返回layer_circuits的每层的类型，是单比特门层则为1；否则则为2
    """
    if isinstance(layer2instructions[0][0], dict):
        get_qubits = lambda gate: gate['qubits']
    else:
        get_qubits = lambda gate: gate.qubits
    
    return [max([len(get_qubits(gate)) for gate in layer]) for layer in layer2instructions]


def get_couple_prop(circuit_info):
    ''' 
    返回两比特门的比例
    '''
    couple = 0
    for ins in circuit_info['gates']:
        if len(ins['qubits']) == 2:
            couple += 1
    return couple / (len(circuit_info['gates']))


def get_xeb_fidelity(circuit_info, single_qubit_error: np.array, two_qubit_error: np.array, measure0_fidelity: np.array, measure1_fidelity: np.array):
    fidelity = 1
    for instruction in circuit_info['gates']:
        if len(instruction['qubits']) == 2:
            q0_id, q1_id = instruction['qubits'][0], instruction['qubits'][1]
            if q0_id > q1_id:
                fidelity = fidelity * (1 - two_qubit_error[q1_id])
            else:
                fidelity = fidelity * (1 - two_qubit_error[q0_id])
        else:
            q0_id = instruction['qubits'][0]
            fidelity = fidelity * (1 - single_qubit_error[q0_id])
    return fidelity * np.product((measure0_fidelity + measure1_fidelity) / 2)


def get_circuit_duration(layer2instructions, single_qubit_gate_time=30, two_qubit_gate_time=60):
    layer_types = get_layer_type_divide(layer2instructions)

    duration = 0
    for layer_type in layer_types:
        if layer_type == 1:
            duration += single_qubit_gate_time
        elif layer_type == 2:
            duration += two_qubit_gate_time
        else:
            raise Exception(layer_type)

    return duration
